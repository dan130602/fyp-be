"""Original-audio GPU stages. No gold, roles, training or audio writes."""
import argparse,hashlib,json,math,os,sys,time
from copy import deepcopy
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
RATE=16000

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert read(p)==d,f'Existing checkpoint differs: {p}';return
 tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def progress(stage,n,total,cid,start):
 elapsed=time.monotonic()-start;bar='#'*(24*n//total)+'-'*(24-24*n//total)
 print(f'[{bar}] {stage} {n}/{total} calls: {cid}; elapsed {elapsed/60:.1f} min; ETA {elapsed/max(n,1)*(total-n)/60:.1f} min',flush=True)
def waveform(path):
 import numpy as np,soundfile as sf
 from scipy.signal import resample_poly
 a,sr=sf.read(path,dtype='float32',always_2d=True);a=a.mean(axis=1)
 if sr!=RATE:
  g=math.gcd(sr,RATE);a=resample_poly(a,RATE//g,sr//g)
 a=np.ascontiguousarray(a,dtype=np.float32);assert len(a) and np.isfinite(a).all();return a
def model_files(folder):
 paths=sorted(list(folder.glob('*.safetensors'))+list(folder.glob('*.json')));assert any(p.suffix=='.safetensors' for p in paths),'Model checkpoint absent'
 return {p.name:sha(p) for p in paths}
def preflight():
 m=read(HERE/'manifest.json')
 if m.get('scope')!='development-only-original-audio' or not 0<len(m.get('calls',[]))<=175:raise ValueError('Expected a nonempty development-only original-audio batch of at most175 calls')
 if len({c['call_id'] for c in m['calls']})!=len(m['calls']):raise ValueError('Duplicate call IDs')
 for c in m['calls']:
  p=(HERE/c['audio']).resolve()
  if not p.is_relative_to(HERE.resolve()) or p.suffix.lower() not in ['.wav','.mp3']:raise ValueError('Audio must stay inside the private batch')
  if sha(p)!=c['audio_sha256']:raise ValueError('Audio hash mismatch: '+c['call_id'])
 return m

def timeline(annotation):
 if hasattr(annotation,'itertracks'):pairs=((t,l) for t,_,l in annotation.itertracks(yield_label=True))
 else:pairs=iter(annotation)
 rows=sorted([dict(start=float(t.start),end=float(t.end),speaker=str(l)) for t,l in pairs],key=lambda r:(r['start'],r['end'],r['speaker']))
 assert all(math.isfinite(r['start']) and math.isfinite(r['end']) and 0<=r['start']<r['end'] for r in rows);return rows
def diarize(args,m):
 import torch
 from huggingface_hub import snapshot_download
 from pyannote.audio import Pipeline
 from pyannote.audio.core.task import Specifications,Problem,Resolution
 from torch.torch_version import TorchVersion
 from model_cache import validate_snapshot
 model=Path(snapshot_download('pyannote/speaker-diarization-community-1',revision='3533c8cf8e369892e6b79ff1bf80f7b0286a54ee',local_files_only=True));validate_snapshot(model,'5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be')
 fingerprint={n:sha(model/n) for n in ('config.yaml','segmentation/pytorch_model.bin','embedding/pytorch_model.bin')};save(args.output/'diarizer-model.json',fingerprint)
 assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0),flush=True)
 with torch.serialization.safe_globals([Specifications,Problem,Resolution,TorchVersion]):pipeline=Pipeline.from_pretrained(str(model))
 pipeline.to(torch.device('cuda'));start=time.monotonic()
 for n,c in enumerate(m['calls'],1):
  target=args.output/'speakers'/f"{c['call_id']}.json"
  if target.exists():assert read(target)['audio_sha256']==c['audio_sha256']
  else:
   x=waveform(HERE/c['audio']);t=time.monotonic()
   with torch.inference_mode():pred=pipeline({'waveform':torch.from_numpy(x).unsqueeze(0),'sample_rate':RATE},num_speakers=2)
   save(target,dict(call_id=c['call_id'],audio_sha256=c['audio_sha256'],audio_seconds=len(x)/RATE,speaker_diarization=timeline(pred.speaker_diarization),exclusive_speaker_diarization=timeline(pred.exclusive_speaker_diarization),inference_seconds=time.monotonic()-t,model=fingerprint))
  progress('Pyannote',n,len(m['calls']),c['call_id'],start)
 save(args.output/'diarization-COMPLETE.json',dict(calls=len(m['calls'])))
def asr(args,m):
 import numpy as np,torch
 from meralion_3_asr import Meralion3ASR
 from core import turn_chunks,turn_document
 from chunk_experiment import chunk_bounds
 modelpath=args.root/'model';revision=m['meralion_revision'];metas=list((modelpath/'.cache/huggingface/download').glob('*.safetensors.metadata'))
 assert metas and all(p.read_text().splitlines()[0]==revision for p in metas),'MERaLiON download revision cannot be verified; do not substitute another model'
 fingerprint=model_files(modelpath);dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32;assert torch.cuda.is_available();torch.manual_seed(0)
 save(args.output/'asr-model.json',dict(revision=revision,files=fingerprint,dtype=str(dtype),backend='transformers',batch_size=1,resampling='mono mean + scipy resample_poly 16k',training=False))
 model=Meralion3ASR.from_pretrained(str(modelpath.resolve()),backend='transformers',device='cuda',batch_size=1,dtype=dtype);start=time.monotonic()
 for n,c in enumerate(m['calls'],1):
  done=[args.output/arm/f"{c['call_id']}.json" for arm in ['energy']]
  if all(p.exists() for p in done):
   for p in done:
    d=read(p);assert d['audio_sha256']==c['audio_sha256'] and d['model_revision']==revision
   progress('MERaLiON verified completed calls',n,len(m['calls']),c['call_id'],start);continue
  x=waveform(HERE/c['audio']);speaker=read(args.output/'speakers'/f"{c['call_id']}.json");assert speaker['audio_sha256']==c['audio_sha256']
  turn_audit=[]
  plans={'energy':[dict(start_sample=a,end_sample=b) for a,b in chunk_bounds(x,True)]}
  for arm,plan in plans.items():
   chunks=[]
   for k,b in enumerate(plan,1):
    p=args.output/'chunks'/arm/c['call_id']/f'{k:05d}.json';key=dict(audio_sha256=c['audio_sha256'],bounds=b,model=fingerprint,implementation=sha(Path(__file__)))
    if p.exists():d=read(p);assert d['key']==key
    else:
     t=time.monotonic();text=model.transcribe(x[b['start_sample']:b['end_sample']]);assert isinstance(text,str);d=dict(key=key,text=text,inference_seconds=time.monotonic()-t);save(p,d)
    chunks.append(dict(b,text=d['text'],inference_seconds=d['inference_seconds']));print(f"{c['call_id']} {arm} {k}/{len(plan)} clips checkpointed",flush=True)
   result=dict(call_id=c['call_id'],audio_sha256=c['audio_sha256'],chunks=chunks,transcript=' '.join(z['text'] for z in chunks if z['text']),model_revision=revision,audio_seconds=len(x)/RATE,empty_clips=sum(not z['text'].strip() for z in chunks),inference_seconds=sum(z['inference_seconds'] for z in chunks))
   if arm=='turn_first':
    result['document'],result['turns']=turn_document(chunks)
    if turn_audit:result['turn_boundary_audit']=turn_audit
   assert result['transcript'].strip(),'Call has no recognized text; retain checkpoints and inspect'
   save(args.output/arm/f"{c['call_id']}.json",result)
  progress('MERaLiON energy control',n,len(m['calls']),c['call_id'],start)
 assert fingerprint==model_files(modelpath),'Model files changed';save(args.output/'asr-COMPLETE.json',dict(calls=len(m['calls']),paths=1,weights_changed=False))
def align(args,m):
 import torch
 from qwen_asr import Qwen3ForcedAligner
 from alignment_core import document,canonical
 from guarded_speaker_recovery import recover_chunk
 from shared_recovery import recover,CONFIG
 from importlib.metadata import version
 modelpath=args.root/'models/forced-aligner';fingerprint=model_files(modelpath);assert sha(modelpath/'config.json')==m['qwen_config_sha256'],'Qwen config differs from frozen prior baseline'
 save(args.output/'aligner-model.json',dict(files=fingerprint,qwen_asr_version=version('qwen-asr'),language_argument='English (existing baseline setting)',shared_recovery=CONFIG));assert torch.cuda.is_available();model=Qwen3ForcedAligner.from_pretrained(str(modelpath),dtype=torch.bfloat16,device_map='cuda:0');model.model.eval();start=time.monotonic()
 for n,c in enumerate(m['calls'],1):
  done=args.output/'aligned'/f"{c['call_id']}.json"
  if done.exists():
   d=read(done);source=read(args.output/'energy'/f"{c['call_id']}.json");assert d['audio_sha256']==c['audio_sha256'] and d['text_only']['results']['transcripts'][0]['transcript']==source['transcript']
   progress('Qwen verified completed calls',n,len(m['calls']),c['call_id'],start);continue
  x=waveform(HERE/c['audio']);source=read(args.output/'energy'/f"{c['call_id']}.json");sp=read(args.output/'speakers'/f"{c['call_id']}.json");assert source['audio_sha256']==sp['audio_sha256']==c['audio_sha256'];items=[];refined=[];audit=[];changes=[]
  for k,chunk in enumerate(source['chunks'],1):
   if not chunk['text']:continue
   a,b=chunk['start_sample'],chunk['end_sample'];p=args.output/'alignment-chunks'/c['call_id']/f'{k:05d}.json';key=dict(audio_sha256=c['audio_sha256'],source_sha256=sha(args.output/'energy'/f"{c['call_id']}.json"),chunk=k,bounds=[a,b],model=fingerprint)
   if p.exists():d=read(p);assert d['key']==key
   else:
    t=time.monotonic()
    with torch.inference_mode():u=model.align(audio=(x[a:b],RATE),text=chunk['text'],language='English')[0] if canonical(chunk['text']) else []
    d=dict(key=key,units=[dict(text=z.text,start_time=float(z.start_time),end_time=float(z.end_time)) for z in u],inference_seconds=time.monotonic()-t);save(p,d)
   base,details,_=recover_chunk(chunk['text'],d['units'],a/RATE,(b-a)/RATE,sp['exclusive_speaker_diarization']);new,added=recover(chunk['text'],d['units'],a/RATE,(b-a)/RATE,sp['exclusive_speaker_diarization'],sp['speaker_diarization'],base,details)
   for z in added:z.update(chunk=k,global_item=len(items)+z['item'])
   items.extend(base);refined.extend(new);audit.extend(details);changes.extend(added);print(f"{c['call_id']} Qwen {k}/{len(source['chunks'])} chunks checkpointed",flush=True)
  plain=deepcopy(items)
  for i in plain:i.pop('speaker_label',None)
  result=dict(call_id=c['call_id'],audio_sha256=c['audio_sha256'],text_only=document(source['transcript'],plain),qwen_guarded=document(source['transcript'],items),qwen_refined=document(source['transcript'],refined),shared_changes=changes,audit=audit)
  check=deepcopy(result['qwen_refined'])
  for i in check['results']['items']:i.pop('speaker_label',None)
  assert check==result['text_only'];save(args.output/'aligned'/f"{c['call_id']}.json",result);progress('Qwen alignment + both label rules',n,len(m['calls']),c['call_id'],start)
 assert fingerprint==model_files(modelpath);save(args.output/'COMPLETE.json',dict(calls=len(m['calls']),source_manifest_sha256=sha(HERE/'manifest.json'),training=False,weights_changed=False,api_calls=False,paths=['text_only','qwen_guarded','qwen_refined']))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','diarize','asr','align']);p.add_argument('--root',type=Path,default=Path.home()/'meralion-pilot');p.add_argument('--output',type=Path,required=True);args=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYANNOTE_METRICS_ENABLED='0');m=preflight();print('PREFLIGHT PASSED: current-source audio hashes and packaged code verified.',flush=True)
 if args.stage!='preflight':globals()[args.stage](args,m)
if __name__=='__main__':main()
