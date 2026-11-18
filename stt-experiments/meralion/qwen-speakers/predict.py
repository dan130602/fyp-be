"""Original-audio GPU stages; explicit private paths, no reference input."""
from pathlib import Path
import argparse,os,shutil
import base_gpu as b
import turn_gpu

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','diarize','asr','auxiliary','align','labels']);p.add_argument('--batch',type=Path,required=True);p.add_argument('--models',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYANNOTE_METRICS_ENABLED='0')
 b.HERE=args.batch.resolve();turn_gpu.HERE=b.HERE;m=b.preflight();args.output=args.output.resolve()
 source=Path(__file__).resolve().parent
 if m.get('meralion_revision')!='9f226d738dee62c60a2f43b0ab7b9a2585a132c3' or m.get('qwen_config_sha256')!='d616c65d46c4b90bdc651b0a0963ea932732241140f337f9bb6b0335a9c8ef09':raise ValueError('Model pins differ from the validated pipeline')
 args.output.mkdir(parents=True,exist_ok=True);b.save(args.output/'input-manifest.json',m)
 if args.stage=='labels':
  from pipeline import refine_document
  for c in m['calls']:
   cid=c['call_id'];doc=b.read(args.output/'aligned'/f'{cid}.json');speakers=b.read(args.output/'speakers'/f'{cid}.json');aux=b.read(args.output/'turn_first'/f'{cid}.json')
   if any(x['audio_sha256']!=c['audio_sha256'] for x in [doc,speakers,aux]):raise ValueError('Source audio hashes differ')
   result,changes=refine_document(doc['qwen_refined'],speakers['exclusive_speaker_diarization'],speakers['speaker_diarization'],aux['document']['results']['items']);b.save(args.output/'refined'/f'{cid}.json',result);b.save(args.output/'label-audit'/f'{cid}.json',changes)
  return
 if args.models is None:p.error('--models is required for GPU model stages')
 args.root=args.models.resolve()
 if b.model_files(args.root/'model')!=b.read(source/'expected-asr-model.json')['files'] or b.model_files(args.root/'models/forced-aligner')!=b.read(source/'expected-aligner-model.json')['files']:raise ValueError('Pinned model files differ')
 if args.stage=='preflight':print('Original-audio preflight passed:',len(m['calls']),'calls. No inference started.');return
 if args.stage=='diarize':
  import clustering
  clustering.HERE=b.HERE;clustering.diarize(args,m);name='diarizer-model.json'
 elif args.stage=='asr':b.asr(args,m);name='asr-model.json'
 elif args.stage=='auxiliary':turn_gpu.asr(args,m);name='asr-model.json'
 else:b.align(args,m);name='aligner-model.json'
 if b.read(args.output/name)!=b.read(source/('expected-'+name)):raise ValueError('Effective model configuration differs from the validated pipeline')
if __name__=='__main__':main()
