import contextlib
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import run_all
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from stvqa_degradation import operators
from stvqa_degradation.common import read_json,sha_file
from stvqa_degradation.pipeline import execute,generation_task
from stvqa_degradation.source_layout import shard_locations


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def fixture(base):
    source=base/'clean'; data=source/'data'; data.mkdir(parents=True)
    config=read_json(ROOT/'configs/full_stvqa.json')
    config.update(source_root=str(source),output_root=str(base/'output'),
                  workspace_root=str(base/'workspace'),workers=1,
                  calibration_images=25,single_calibration_images=2,ablation_images=4,
                  expected_images={'train':40},expected_questions={'train':54})
    rows=[]; original=[]
    rng=np.random.default_rng(19)
    for i in range(40):
        image=rng.integers(25,235,(80,112,3),dtype=np.uint8)
        buf=io.BytesIO()
        if i==0:
            Image.fromarray(image).save(buf,format='MPO',save_all=True,append_images=[Image.fromarray(255-image)])
        else: Image.fromarray(image).save(buf,format='JPEG')
        raw=buf.getvalue() if i!=39 else original[0]
        original.append(raw)
        qas=[dict(question=f'Question {i}/{j}?',answers=['example']*(j+1)) for j in range(1+(i%3==0))]
        rows.append(dict(image=dict(bytes=raw,path=f'original_{i}.jpg'),qas=qas))
    schema=pa.schema([
        ('image',pa.struct([('bytes',pa.binary()),('path',pa.string())])),
        ('qas',pa.list_(pa.struct([('question',pa.string()),('answers',pa.list_(pa.string()))])))])
    files=[]
    for i,indices in enumerate(np.array_split(np.arange(40),6)):
        path=data/f'train-{i:05d}-of-00006.parquet'
        table=pa.Table.from_pylist([rows[int(j)] for j in indices],schema=schema)
        pq.write_table(table,path,row_group_size=2)
        files.append(dict(name=path.name,size=path.stat().st_size,sha256=sha_file(path)))
    pin=base/'fixture_pin.json'
    pin.write_text(json.dumps(dict(source_repo='vikhyatk/st-vqa',source_revision='a'*40,source_split='train',files=files)),encoding='utf-8')
    config['source_manifest_path']=str(pin)
    return config,original


def image_from_record(r):
    return dict(dataset='stvqa',split=r['split'],image_id=r['image_id'],
        clean_image_path=r['source_image_path'],source_image_sha256=r['source_image_sha256'],
        clean_width=int(r['clean_width']),clean_height=int(r['clean_height']),
        source_format=r['source_format'],source_frames=int(r['source_frames']))


class OperatorTests(unittest.TestCase):
    def test_original_notebook_cell_and_params(self):
        provenance=read_json(ROOT/'operator_provenance.json')
        module=Path(operators.__file__).read_text(encoding='utf-8')
        cell=module.split('# Original notebook cell below, byte-for-byte unchanged.\n',1)[1]
        self.assertEqual(hashlib.sha256(cell.encode()).hexdigest(),provenance['original_cell_sha256'])
        params=json.dumps(operators.DEFAULT_PARAMS,sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(params).hexdigest(),provenance['default_params_sha256'])

    def test_repeatability_mix_constraints_and_noise(self):
        image=np.tile(np.arange(128,dtype=np.uint8)[None,:,None],(96,1,3))+60
        for condition in ('realistic_mix',)+operators.SINGLES:
            a,p=operators.degrade(image,condition,'L2',None,1234)
            b,q=operators.degrade(image,condition,'L2',None,1234)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(p,q)
            self.assertEqual(a.shape,image.shape)
            self.assertEqual(a.dtype,np.uint8)
        for seed in range(50):
            _,params=operators.degrade(image,'realistic_mix','L2',None,seed)
            ops=[r['operation'] for r in params]
            self.assertIn(len(ops),(2,3))
            self.assertLessEqual(sum(c in ops for c in ('motion_camera_shake','defocus_blur')),1)
            self.assertLessEqual(sum(c in ops for c in ('lowlight_sensor_noise','overexposure_glare')),1)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.base=Path(cls.temp.name)
        cls.config,cls.original=fixture(cls.base)
        with contextlib.redirect_stdout(io.StringIO()): cls.summary=execute(cls.config)
        cls.root=Path(cls.config['output_root'])

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_complete_all_rows_all_questions_and_metadata(self):
        self.assertEqual(self.summary['status'],'complete')
        self.assertEqual(self.summary['main_images'],40)
        self.assertEqual(self.summary['main_questions'],54)
        self.assertEqual(self.summary['stages']['calibration']['output_count'],117)
        self.assertEqual(self.summary['stages']['ablation']['output_count'],28)
        rows=read_csv(self.root/'main/final_main_manifest.csv')
        self.assertEqual({r['split'] for r in rows},{'train'})
        self.assertEqual(len({r['image_id'] for r in rows}),40)
        self.assertEqual({len(json.loads(r['answers'])) for r in rows},{1,2})
        self.assertTrue(all(r['gt_available']=='True' and r['dataset']=='stvqa' for r in rows))
        self.assertFalse((self.root/'main/manifests/val.csv').exists())
        self.assertFalse((self.root/'main/manifests/test.csv').exists())
        for r in rows:
            self.assertEqual(r['clean_width'],r['degraded_width'])
            self.assertEqual(r['clean_height'],r['degraded_height'])
            self.assertEqual(sha_file(r['degraded_image_path']),r['degraded_sha256'])
            self.assertEqual(sha_file(r['clean_image_path']),r['source_image_sha256'])
            self.assertIn(len(json.loads(r['params'])),(2,3))
        provenance=read_json(self.root/'metadata/source_provenance.json')
        self.assertEqual(provenance['exact_duplicate_rows_kept'],1)
        review=read_json(self.root/'calibration/manual_review/reviewed_samples.random_label_provenance.json')
        self.assertFalse(review['human_review_performed'])
        self.assertEqual(review['counts'],{'EASY':82,'DIFFICULT_BUT_ANSWERABLE':35})

    def test_original_bytes_mpo_and_frame_zero(self):
        rows=read_csv(self.root/'metadata/source_image_manifest.csv')
        for i,r in enumerate(rows):
            self.assertEqual(Path(r['clean_image_path']).read_bytes(),self.original[i])
            self.assertEqual(r['image_id'],f'stvqa_train_{i:06d}')
        mpo=rows[0]
        self.assertEqual(mpo['source_format'],'MPO')
        self.assertEqual(mpo['source_frames'],'2')
        self.assertEqual(mpo['source_frame_index'],'0')
        self.assertEqual(Path(mpo['clean_image_path']).suffix,'.mpo')
        record=next(r for r in read_csv(self.root/'main/image_manifest.csv') if r['image_id']==mpo['image_id'])
        with Image.open(mpo['clean_image_path']) as im:
            im.seek(0); clean=np.asarray(im.convert('RGB'),dtype=np.uint8)
        expected,_=operators.degrade(clean,'realistic_mix','L2',None,int(record['seed']))
        with Image.open(record['output_path']) as im: actual=np.asarray(im.convert('RGB'))
        np.testing.assert_array_equal(expected,actual)

    def test_resume_in_parallel_same_pngs_and_manifests(self):
        old_manifest=(self.root/'main/final_main_manifest.csv').read_bytes()
        before={str(p):sha_file(p) for p in (self.root/'main/images').rglob('*.png')}
        clean_times={str(p):p.stat().st_mtime_ns for p in (self.root/'source_clean/images').iterdir()}
        with contextlib.redirect_stdout(io.StringIO()): result=execute(dict(self.config,workers=2))
        self.assertEqual(result['stages']['main']['reused_count'],40)
        self.assertEqual(result['stages']['calibration']['reused_count'],117)
        self.assertEqual(old_manifest,(self.root/'main/final_main_manifest.csv').read_bytes())
        self.assertEqual(before,{str(p):sha_file(p) for p in (self.root/'main/images').rglob('*.png')})
        self.assertEqual(clean_times,{str(p):p.stat().st_mtime_ns for p in (self.root/'source_clean/images').iterdir()})

    def test_orphan_resume_and_corrupt_sidecar_rejected(self):
        r=read_csv(self.root/'main/image_manifest.csv')[0]
        side=Path(r['output_path']+'.json')
        old=side.read_bytes(); side.unlink()
        task=(image_from_record(r),'main',r['condition'],r['level'],r['config_sha256'],r['source_manifest_sha256'],self.config)
        regenerated,reused,_=generation_task(task)
        self.assertFalse(reused)
        self.assertEqual(regenerated['output_sha256'],r['output_sha256'])
        broken=json.loads(old); broken['seed']=0; side.write_text(json.dumps(broken))
        try:
            with self.assertRaisesRegex(RuntimeError,'mismatch'): generation_task(task)
        finally: side.write_bytes(old)

    def test_new_parallel_generation_matches_original_pngs(self):
        tasks=[]; expected=[]
        for r in read_csv(self.root/'main/image_manifest.csv')[:2]:
            tasks.append((image_from_record(r),'main',r['condition'],r['level'],r['config_sha256'],r['source_manifest_sha256'],self.config))
            expected.append(r['output_sha256'])
            Path(r['output_path']).unlink(); Path(r['output_path']+'.json').unlink()
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(generation_task,tasks))
        self.assertEqual([r[0]['output_sha256'] for r in results],expected)
        self.assertTrue(all(not r[1] for r in results))

    def test_changed_parquet_rejected_before_export(self):
        p=Path(self.config['source_root'])/'data/train-00000-of-00006.parquet'
        old=p.read_bytes(); changed=bytearray(old); changed[len(changed)//2]^=1; p.write_bytes(changed)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Shard SHA-256'): execute(self.config)
        finally: p.write_bytes(old)

    def test_changed_exported_clean_rejected(self):
        p=Path(read_csv(self.root/'metadata/source_image_manifest.csv')[0]['clean_image_path'])
        old=p.read_bytes(); p.write_bytes(b'broken clean')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'clean bytes changed'): execute(self.config)
        finally: p.write_bytes(old)


class ControllerTests(unittest.TestCase):
    def test_default_configuration_and_nested_paths(self):
        config=read_json(ROOT/'configs/full_stvqa.json')
        run_all.validate(config)
        self.assertEqual(config['scheduler']['cpus'],8)
        self.assertEqual(config['scheduler']['qos'],'mig2-35')
        config['output_root']=config['source_root']+'/bad'
        with self.assertRaisesRegex(ValueError,'separate'): run_all.validate(config)

    def test_submit_one_script_and_job_no_compute(self):
        with tempfile.TemporaryDirectory(prefix='test space ') as d:
            root=Path(d); config,_=fixture(root)
            with patch('subprocess.check_output',return_value='12345\n') as submit:
                with contextlib.redirect_stdout(io.StringIO()): run_all.submit(config)
            submit.assert_called_once()
            args=submit.call_args.args[0]
            self.assertIn('--partition=normal',args)
            self.assertIn('--gres=gpu:2g.35gb:1',args)
            self.assertIn('--qos=mig2-35',args)
            self.assertIn('--cpus-per-task=8',args)
            scripts=list((root/'workspace/logs').glob('*.sh'))
            self.assertEqual(len(scripts),1)
            self.assertIn('--in-allocation',scripts[0].read_text())
            self.assertEqual(read_json(next((root/'workspace/logs').glob('*_config.json'))),config)

    def test_no_login_compute_fallback(self):
        with patch.object(sys,'argv',['run_all.py','--in-allocation']),patch.object(sys,'platform','linux'),patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'refusing to run on login'): run_all.main()

    def test_missing_or_extra_shard_rejected_on_login(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config,_=fixture(root)
            extra=root/'clean/data/extra.parquet'; extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'extra/missing'): shard_locations(config)
            extra.unlink()
            (root/'clean/data/train-00000-of-00006.parquet').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'complete shard'): shard_locations(config)


if __name__=='__main__': unittest.main()
