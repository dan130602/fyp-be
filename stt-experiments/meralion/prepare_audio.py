"""Copy verified development recordings to a private inference batch without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

DATASET='advisory_execution_multilingual_v1_development'
REVISION='9f226d738dee62c60a2f43b0ab7b9a2585a132c3'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args()
    if args.dataset_root.name != DATASET:
        raise ValueError('Only the development dataset is allowed')
    manifest=json.loads((args.dataset_root/'dataset_manifest.json').read_text(encoding='utf-8'))
    ids=manifest['call_ids']
    if len(ids)!=175 or len(set(ids))!=175:
        raise ValueError('Expected exactly 175 distinct development calls')
    rows=[]
    copies=[]
    for cid in ids:
        if not cid.startswith('CALL_') or not cid[5:].isdigit():
            raise ValueError('Unexpected call ID')
        renders=list((args.dataset_root/'calls'/cid/'renders').glob('*/render_manifest.json'))
        if len(renders)!=1:
            raise ValueError(f'Expected one render manifest for {cid}')
        render=json.loads(renders[0].read_text(encoding='utf-8'))
        name=render.get('audio_file','audio.mp3')
        if name not in {'audio.wav','audio.mp3'}:
            raise ValueError(f'Unexpected audio_file for {cid}')
        audio=renders[0].parent/name
        candidates=list((args.dataset_root/'calls'/cid/'renders').glob('*/audio.wav'))+list((args.dataset_root/'calls'/cid/'renders').glob('*/audio.mp3'))
        if len(candidates)!=1 or candidates[0]!=audio:
            raise ValueError(f'Expected exactly the manifest audio for {cid}')
        payload=audio.read_bytes()
        if hashlib.sha256(payload.replace(b'\r\n',b'\n')).hexdigest()!=render['audio_sha256']:
            raise ValueError(f'Render hash mismatch for {cid}')
        digest=hashlib.sha256(payload).hexdigest()
        relative=f'audio/{cid}{audio.suffix}'
        target=args.batch/relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Refusing to overwrite different content: {target}')
        rows.append(dict(call_id=cid,audio=relative,audio_sha256=digest))
        copies.append((audio,target))
    document=dict(dataset_id=DATASET,revision=REVISION,calls=rows)
    existing=args.batch/'calls.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))!=document:
        raise ValueError('Existing batch manifest differs; use a new private batch directory')
    for audio,target in copies:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(audio,target)
    existing.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print('175 development recordings validated; batch copied without labels or audio conversion.')


if __name__=='__main__':
    main()
