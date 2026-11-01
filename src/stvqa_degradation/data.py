"""Stream local pinned HF Parquet shards and export original image bytes safely."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from .common import require,sha_file,lock_bytes,json_bytes,csv_bytes,atomic_bytes
from .source_layout import shard_locations

IMAGE_COLUMNS=['dataset','split','image_id','source_image_id','source_row_index',
               'source_shard','source_shard_row_index','clean_image_path',
               'source_image_sha256','clean_width','clean_height',
               'source_format','source_frame_index','source_frames']
QA_COLUMNS=['dataset','split','image_id','source_image_id','question_id','question',
            'answers','answer_count','gt_available','source_row_index','qa_index']


def select_images(images,n,seed):
    ordered=sorted(images,key=lambda r:(r['split'],r['image_id']))
    require(len(ordered)>=n,f'Need {n} source images; only {len(ordered)} available')
    indices=np.random.default_rng(seed).choice(len(ordered),size=n,replace=False)
    return sorted((ordered[int(i)] for i in indices),key=lambda r:(r['split'],r['image_id']))


def image_info(data):
    with Image.open(io.BytesIO(data)) as image:
        image.seek(0)
        fmt=image.format
        width,height=image.size
        frames=getattr(image,'n_frames',1)
        image.load()  # fully decode first frame now, not just trust the header
    require(min(width,height)>=7,'Source image too small for original SSIM')
    ext={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp','BMP':'.bmp','TIFF':'.tif','GIF':'.gif','MPO':'.mpo'}.get(fmt)
    if ext is None:
        choices=[e for e,name in Image.registered_extensions().items() if name==fmt and e[1:].isalnum()]
        require(bool(choices),f'Unsupported source image format: {fmt}')
        ext=sorted(choices,key=lambda e:(len(e),e))[0]
    return ext,width,height,fmt,frames


def discover(config,root):
    manifest,pin,folder=shard_locations(config)
    source_pin=sha_file(manifest)
    lock_bytes(root/'metadata/hf_source_manifest.json',json_bytes(pin))
    shard_rows={}; total_bytes=0
    for item in sorted(pin['files'],key=lambda r:r['name']):
        path=folder/item['name']
        print('[SOURCE] Verify shard SHA-256:',item['name'],flush=True)
        require(sha_file(path)==item['sha256'],f'Shard SHA-256 differs from pinned HF revision: {path}')
        parquet=pq.ParquetFile(path)
        require({'image','qas'}.issubset(parquet.schema_arrow.names),f'Missing image/qas columns: {path}')
        shard_rows[item['name']]=parquet.metadata.num_rows
        total_bytes+=item['size']
    require(sum(shard_rows.values())==config['expected_images']['train'],'Unexpected number of source rows across shards')
    print('[SOURCE] Rows:',sum(shard_rows.values()),'| revision:',pin['source_revision'],flush=True)
    # Exported clean is part of this output, so paired use does not depend on a deletable venv cache.
    export=root/'source_clean/images'
    export.mkdir(parents=True,exist_ok=True)
    existing=sum(p.stat().st_size for p in export.iterdir() if p.is_file())
    extra=max(0,total_bytes*1.1-existing)
    require(extra+100_000_000<shutil.disk_usage(root).free,'Not enough free space to export clean images')
    require(total_bytes/1e9<config['max_output_disk_gb'],'Configured output budget cannot hold source export')
    images=[]; questions=[]; names=set(); ordinal=0; reused=0
    for item in sorted(pin['files'],key=lambda r:r['name']):
        shard=folder/item['name']
        parquet=pq.ParquetFile(shard)
        shard_row=0
        # A small batch bounds memory; do not read a ~500 MB shard into pandas at once.
        for batch in parquet.iter_batches(batch_size=16,columns=['image','qas'],use_threads=False):
            for row in batch.to_pylist():
                ident=f'stvqa_train_{ordinal:06d}'
                image=row['image']
                require(isinstance(image,dict),f'Expected HF image bytes/path struct: {ident}')
                data=image.get('bytes')
                require(isinstance(data,(bytes,bytearray)) and len(data)>0,
                        f'Missing embedded image bytes: {ident}. No automatic external URL/path fetch.')
                data=bytes(data)
                ext,width,height,fmt,frames=image_info(data)
                path=export/(ident+ext)
                digest=hashlib.sha256(data).hexdigest()
                if path.exists():
                    require(sha_file(path)==digest,f'Existing exported clean bytes changed: {path}')
                    reused+=1
                else: atomic_bytes(path,data)
                names.add(path.name)
                images.append(dict(dataset='stvqa',split='train',image_id=ident,source_image_id=ident,
                    source_row_index=ordinal,source_shard=item['name'],source_shard_row_index=shard_row,
                    clean_image_path=str(path),source_image_sha256=digest,
                    clean_width=width,clean_height=height,source_format=fmt,source_frame_index=0,source_frames=frames))
                qas=row['qas']
                require(isinstance(qas,list) and bool(qas),f'Missing QA list: {ident}')
                for qa_index,qa in enumerate(qas):
                    question=qa.get('question'); answers=qa.get('answers')
                    require(isinstance(question,str) and bool(question.strip()),f'Missing question: {ident}/{qa_index}')
                    require(isinstance(answers,list) and bool(answers) and all(isinstance(a,str) for a in answers),
                            f'Invalid reference answers: {ident}/{qa_index}')
                    require(any(a.strip() for a in answers),f'All answers empty: {ident}/{qa_index}')
                    questions.append(dict(dataset='stvqa',split='train',image_id=ident,source_image_id=ident,
                        question_id=f'{ident}_q{qa_index:02d}',question=question,
                        answers=json.dumps(answers,ensure_ascii=False),answer_count=len(answers),
                        gt_available=True,source_row_index=ordinal,qa_index=qa_index))
                ordinal+=1; shard_row+=1
                if ordinal%1000==0 or ordinal==config['expected_images']['train']:
                    print(f'[SOURCE] Export/verify {ordinal}/{config["expected_images"]["train"]} images | {len(questions)} QA | reused {reused}',flush=True)
        require(shard_row==shard_rows[item['name']],f'Parquet row iteration incomplete: {shard}')
        # Detect source modified while it was being read/exported.
        require(sha_file(shard)==item['sha256'],f'Shard changed during export: {shard}')
    require(len(images)==config['expected_images']['train'],'Source image count mismatch')
    require(len(questions)==config['expected_questions']['train'],'Source question count mismatch')
    require({p.name for p in export.iterdir() if p.is_file()}==names,'Extra or missing exported clean image files')
    image_sha=lock_bytes(root/'metadata/source_image_manifest.csv',csv_bytes(images,IMAGE_COLUMNS))
    qa_sha=lock_bytes(root/'metadata/locked_eval_manifest.csv',csv_bytes(questions,QA_COLUMNS))
    lock_bytes(root/'source_clean/image_manifest.csv',csv_bytes(images,IMAGE_COLUMNS))
    lock_bytes(root/'source_clean/question_manifest.csv',csv_bytes(questions,QA_COLUMNS))
    hashes=Counter(r['source_image_sha256'] for r in images)
    provenance=dict(dataset='stvqa',source_repo=pin['source_repo'],source_revision=pin['source_revision'],
        source_split='train',source_root=str(folder),source_pin_sha256=source_pin,shard_sha256=pin['files'],
        source_image_manifest_sha256=image_sha,qa_manifest_sha256=qa_sha,
        counts={'train':dict(images=len(images),questions=len(questions))},
        exact_duplicate_hash_groups=sum(n>1 for n in hashes.values()),
        exact_duplicate_rows_kept=sum(n-1 for n in hashes.values()),
        duplicate_policy='Keep every HF row and all QAs, including exact byte duplicates; no resampling/deduplication',
        id_policy='Sorted shard filename, row order, global ordinal stvqa_train_%06d; matches prior importer',
        source_frame_policy='Preserve original bytes including MPO; decode only frame 0 for degradation',
        scope='Full provided HF train source only; not the complete official ST-VQA train/val/test collection')
    lock_bytes(root/'metadata/source_provenance.json',json_bytes(provenance))
    print('[AUDIT] Counts:',provenance['counts'],'| exact duplicate rows kept:',provenance['exact_duplicate_rows_kept'],flush=True)
    return images,questions,provenance
