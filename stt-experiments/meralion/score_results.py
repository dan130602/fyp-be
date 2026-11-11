"""Adapt private predictions, then call the project's unchanged offline scoring CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from formatting import format_fee
from transcript_adapter import transcript_document

DATASET='advisory_execution_multilingual_v1_development'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,default=Path(__file__).resolve().parents[2]/'evaluation-engine')
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--fee-formatting',action='store_true')
    parser.add_argument('--allow-partial',action='store_true',help='Permit a development smoke run, never publish it as a full run')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,120}',args.run_id):
        parser.error('Use a simple unique run ID')
    engine=args.engine.resolve()
    dataset=engine.parent/'evaluation-dataset/datasets'/DATASET
    ids=set(json.loads((dataset/'dataset_manifest.json').read_text(encoding='utf-8'))['call_ids'])
    if len(ids)!=175:
        raise ValueError('Expected the 175-call development manifest')
    sys.path.insert(0,str(engine))
    from transcribe_eval.collect import local_audio
    rows={}
    # Validate the complete input before creating a run.
    for path in sorted(args.results.glob('CALL_*.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        cid=row['call_id']
        if cid in rows or cid not in ids:
            raise ValueError('Unknown or duplicate development call')
        settings=row['settings']
        if settings.get('adapter') is not None or settings.get('training') is not False:
            raise ValueError('This adapter accepts only the unchanged-weight experiment')
        digest=hashlib.sha256(local_audio(dataset,cid,'auto').read_bytes()).hexdigest()
        if digest!=settings['audio_sha256']:
            raise ValueError(f'Audio identity differs: {cid}')
        rows[cid]=row
    if not rows or (not args.allow_partial and set(rows)!=ids):
        raise ValueError('Expected all 175 calls; --allow-partial is only for smoke testing')
    target=engine/'runs'/args.run_id
    target.mkdir(parents=True,exist_ok=False)
    calls=[]
    for cid,row in rows.items():
        text=format_fee(row['transcript']) if args.fee_formatting else row['transcript']
        relative=f'raw/transcripts/{cid}.json'
        dest=target/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(transcript_document(text),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        calls.append(dict(call_id=cid,status='COMPLETED',transcript_file=relative,audio_sha256=row['settings']['audio_sha256']))
    metadata=dict(run_id=args.run_id,dataset_id=DATASET,status='completed',calls=calls,
                  fee_formatting=args.fee_formatting,scope='development',speaker_attribution_available=False,
                  confidence_available=False,word_timestamps_available=False)
    (target/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    env={**os.environ,'EVAL_BUCKET':os.environ.get('EVAL_BUCKET','offline-placeholder'),'PYTHONIOENCODING':'utf-8'}
    return subprocess.run([sys.executable,'-B',str(engine/'scripts/score.py'),'--run',args.run_id],cwd=engine,env=env).returncode


if __name__=='__main__':
    raise SystemExit(main())
