"""Frozen v3->v6->v8 speaker-only refinement, with no benchmark input."""
from pathlib import Path
from copy import deepcopy
import sys,json,argparse
# Use the repository's existing utilities unchanged; never import references or scoring.
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'evaluation-engine'))
import anchor_v3,consensus_v6
from refine import refine as word_transfer
from contract_guard import equivalent
from transcribe_eval.normalize import NORMALIZER_VERSION

def refine_document(document,exclusive,regular,auxiliary_items):
 if NORMALIZER_VERSION!='1.4.0':raise RuntimeError('The validated contract requires unchanged normalizer1.4.0')
 if auxiliary_items is None:raise ValueError('The validated chain requires predicted auxiliary speaker-turn items')
 output=deepcopy(document);before=deepcopy(document['results']['items']);items=before;changes={}
 for name,method in [('anchor_v3',anchor_v3.refine),('consensus_v6',consensus_v6.refine),('v8',word_transfer)]:items,changes[name]=method(items,exclusive,regular,auxiliary_items)
 if len(items)!=len(before) or not equivalent(before,items):raise ValueError('Speaker proposal violated the hypothesis normalization contract')
 for a,b in zip(before,items):
  a,b=deepcopy(a),deepcopy(b);a.pop('speaker_label',None);b.pop('speaker_label',None)
  if a!=b:raise ValueError('Speaker refinement changed a non-speaker field')
 output['results']['items']=items
 return output,changes

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--transcript',type=Path,required=True);p.add_argument('--speakers',type=Path,required=True);p.add_argument('--auxiliary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'));doc=read(args.transcript);speakers=read(args.speakers);aux=read(args.auxiliary)
 source=doc.get('qwen_refined',doc);auxiliary=aux.get('document',aux)['results']['items']
 for prediction in [speakers,aux]:
  if doc.get('audio_sha256') and prediction.get('audio_sha256')!=doc['audio_sha256']:raise ValueError('Predicted inputs are not from the same recording')
 result,_=refine_document(source,speakers['exclusive_speaker_diarization'],speakers['speaker_diarization'],auxiliary)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
