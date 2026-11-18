"""Hypothesis-only label recovery; never changes text, times, or existing labels."""
from copy import deepcopy
import math
import unicodedata
CONFIG={'method':'shared-unit-single-speaker-v1','maximum_unit_seconds':2.0,'require_continuous_exclusive_coverage':True,'reject_competing_regular_speaker':True,'change_existing_labels':False,'fabricate_timestamps':False}
def canonical(text):return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())
def continuous(segments,label,start,end):
 cursor=start
 for s in segments:
  a,b=max(start,s['start']),min(end,s['end'])
  if b<=a:continue
  if s['speaker']!=label or a>cursor+1e-7:return False
  cursor=max(cursor,b)
 return cursor>=end-1e-7
def recover(text,units,offset,duration,exclusive,regular,items,audit):
 assert len(items)==len(audit)
 result=deepcopy(items);source=[(ch,i) for i,c in enumerate(text) for ch in canonical(c)];lex=[(u,canonical(u['text'])) for u in units if canonical(u['text'])]
 if ''.join(c for c,_ in source)!=''.join(w for _,w in lex):return result,[]
 owners=[set() for _ in text];cursor=0;valid=[];previous=-1.0
 for j,(u,w) in enumerate(lex):
  a,b=float(u['start_time']),float(u['end_time']);valid.append(math.isfinite(a) and math.isfinite(b) and 0<=a<b<=duration and a>=previous)
  if math.isfinite(a):previous=a
  for _,i in source[cursor:cursor+len(w)]:owners[i].add(j)
  cursor+=len(w)
 changes=[]
 for i,(item,d) in enumerate(zip(items,audit)):
  if item['type']!='pronunciation' or item.get('speaker_label') or d.get('reason')!='alignment_unit_spans_multiple_items':continue
  used={j for p in range(d['start_char'],d['end_char']) for j in owners[p]}
  if len(used)!=1:continue
  j=next(iter(used))
  if not valid[j]:continue
  u=lex[j][0];a,b=float(u['start_time'])+offset,float(u['end_time'])+offset
  if b-a>CONFIG['maximum_unit_seconds']:continue
  labels={s['speaker'] for s in exclusive if min(b,s['end'])-max(a,s['start'])>1e-9}
  if len(labels)!=1:continue
  label=next(iter(labels))
  if not continuous(exclusive,label,a,b):continue
  if any(s['speaker']!=label and min(b,s['end'])-max(a,s['start'])>1e-9 for s in regular):continue
  result[i]['speaker_label']=label;changes.append(dict(item=i,speaker_label=label,unit=j,start=a,end=b,method=CONFIG['method']))
 for old,new in zip(items,result):
  a,b=deepcopy(old),deepcopy(new);a.pop('speaker_label',None);b.pop('speaker_label',None);assert a==b
  if old.get('speaker_label'):assert old==new
 return result,changes
