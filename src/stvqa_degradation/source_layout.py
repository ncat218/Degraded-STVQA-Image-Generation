"""Small stdlib-only checks shared by login controller and compute importer."""
from pathlib import Path
import json
import re

REPO=Path(__file__).resolve().parents[2]


def shard_locations(config):
    manifest=Path(config['source_manifest_path'])
    if not manifest.is_absolute(): manifest=REPO/manifest
    pin=json.loads(manifest.read_text(encoding='utf-8'))
    if pin.get('source_repo')!='vikhyatk/st-vqa' or pin.get('source_split')!='train':
        raise ValueError('Expected the vikhyatk/st-vqa train source manifest')
    if not re.fullmatch('[a-f0-9]{40}',pin.get('source_revision','')):
        raise ValueError('Source revision must be a full commit hash')
    files=pin['files']
    if not files or len({f['name'] for f in files})!=len(files):
        raise ValueError('Missing or duplicate shard names in source manifest')
    for f in files:
        if Path(f['name']).name!=f['name'] or not f['name'].endswith('.parquet'):
            raise ValueError('Invalid shard filename')
        if not re.fullmatch('[a-f0-9]{64}',f['sha256']) or f['size']<=0:
            raise ValueError('Invalid pinned SHA-256 or shard size')
    source=Path(config['source_root'])
    candidates=[]
    for folder in (source,source/'data'):
        if folder.is_dir() and all((folder/f['name']).is_file() for f in files):
            candidates.append(folder)
    if len(candidates)!=1:
        raise FileNotFoundError(f'Need exactly one complete shard folder at {source} or {source}/data; found {candidates}')
    folder=candidates[0]
    actual={p.name for p in folder.glob('*.parquet') if p.is_file()}
    if actual!={f['name'] for f in files}:
        raise ValueError('Unexpected extra/missing parquet shards; do not mix source revisions')
    for f in files:
        path=folder/f['name']
        if path.stat().st_size!=f['size']:
            raise ValueError(f'Shard size differs from pinned source: {path}')
    return manifest,pin,folder
