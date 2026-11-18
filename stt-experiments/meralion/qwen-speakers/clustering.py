"""No-Qwen full current-source development run: no gold, roles, denoising or Qwen."""
import argparse,os,shutil,time
from pathlib import Path
import base_gpu as b
import json,inspect
HERE=b.HERE
def diarize(args,m):
 import torch
 from huggingface_hub import snapshot_download
 from pyannote.audio import Pipeline
 from pyannote.audio.core.task import Specifications,Problem,Resolution
 from torch.torch_version import TorchVersion
 from pyannote.audio.pipelines.clustering import KMeansClustering
 from model_cache import validate_snapshot
 model=Path(snapshot_download('pyannote/speaker-diarization-community-1',revision='3533c8cf8e369892e6b79ff1bf80f7b0286a54ee',local_files_only=True));validate_snapshot(model,'5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be')
 fp={n:b.sha(model/n) for n in ('config.yaml','segmentation/pytorch_model.bin','embedding/pytorch_model.bin')}
 with torch.serialization.safe_globals([Specifications,Problem,Resolution,TorchVersion]):pipeline=Pipeline.from_pretrained(str(model))
 original=type(pipeline.clustering).__name__;assert original!='KMeansClustering','Duplicate of existing clustering; stop rather than relabel it as a new test'
 replacement=KMeansClustering(metric=pipeline._embedding.metric);replacement.constrained_assignment=True;replacement.instantiate({});pipeline.clustering=replacement;pipeline.klustering='KMeansClustering';pipeline._expects_num_speakers=True
 config=dict(original_clustering=original,alternative='KMeansClustering',num_speakers=2,constrained_assignment=True,installed_clustering_module_sha256=b.sha(Path(inspect.getfile(KMeansClustering))),constructor=str(inspect.signature(KMeansClustering)),model=fp,training=False)
 b.save(args.output/'diarizer-model.json',config);assert torch.cuda.is_available();torch.manual_seed(0);pipeline.to(torch.device('cuda'));start=time.monotonic();print('FIXED CLUSTERING:',json.dumps(config),flush=True)
 for n,c in enumerate(m['calls'],1):
  p=args.output/'speakers'/f"{c['call_id']}.json"
  if p.exists():assert b.read(p)['audio_sha256']==c['audio_sha256']
  else:
   x=b.waveform(HERE/c['audio']);t=time.monotonic()
   with torch.inference_mode():pred=pipeline({'waveform':torch.from_numpy(x).unsqueeze(0),'sample_rate':16000},num_speakers=2)
   b.save(p,dict(call_id=c['call_id'],audio_sha256=c['audio_sha256'],audio_seconds=len(x)/16000,speaker_diarization=b.timeline(pred.speaker_diarization),exclusive_speaker_diarization=b.timeline(pred.exclusive_speaker_diarization),inference_seconds=time.monotonic()-t,model=config))
  b.progress('KMeans Pyannote',n,len(m['calls']),c['call_id'],start)
 assert fp=={n:b.sha(model/n) for n in fp};b.save(args.output/'diarization-COMPLETE.json',dict(calls=len(m['calls'])))
