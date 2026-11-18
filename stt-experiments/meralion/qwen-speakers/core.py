"""Audio-only clip boundaries and honest turn-level transcript assembly."""
import math,warnings
from alignment_core import token_spans,canonical,document
from chunk_experiment import chunk_bounds
RATE=16000

def turn_chunks(waveform,segments,audit=None):
 result=[];previous=0.0;n=len(waveform)
 for index,s in enumerate(segments):
  a,b=float(s['start']),float(s['end']);assert math.isfinite(a) and math.isfinite(b) and 0<=a<b and a>=previous-1e-7
  previous=b
  if a>=n/RATE:
   record=dict(turn_index=index,speaker=str(s['speaker']),start=a,end=b,audio_seconds=n/RATE,reason='wholly_after_audio_end',audio_samples_removed=0)
   if audit is not None:audit.append(record)
   warnings.warn('Audited out-of-audio predicted turn: '+str(record),RuntimeWarning)
   continue
  left=max(0,min(n,round(a*RATE)));right=max(0,min(n,round(b*RATE)))
  if right<=left:raise ValueError('Empty or out-of-range speaker turn; never silently drop it')
  for x,y in chunk_bounds(waveform[left:right],True):result.append(dict(start_sample=left+x,end_sample=left+y,speaker=str(s['speaker']),turn_index=index))
 assert all(0<=c['start_sample']<c['end_sample']<=n and c['end_sample']-c['start_sample']<=30*RATE for c in result)
 return result

def turn_document(chunks):
 items=[];turns=[];texts=[]
 for c in chunks:
  first=len(items);text=c['text'];texts.append(text)
  for _,_,token in token_spans(text):
   pron=bool(canonical(token)) or token in '$%+-\u00a3\u20ac\u00a5\u20b9';item=dict(type='pronunciation' if pron else 'punctuation',alternatives=[dict(content=token)])
   if pron:item['speaker_label']=c['speaker']
   items.append(item)
  turns.append(dict(start=c['start_sample']/RATE,end=c['end_sample']/RATE,speaker=c['speaker'],text=text,first_item=first,last_item_exclusive=len(items)))
 # No word timestamps/confidence are invented from a whole speaker turn.
 return document(' '.join(t for t in texts if t),items),turns
