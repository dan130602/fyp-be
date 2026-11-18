"""Predicted-input-only speaker recovery. No references, IDs, roles or scores.
Only fills missing speaker labels; preserves all other fields and existing labels.
"""
from collections import Counter
from copy import deepcopy
import math
CONFIG=dict(method='boundary-preserving-overlap-and-bridge-v2',overlap_purity=.8,minimum_word_coverage=.5,maximum_bridge_seconds=4.,maximum_bridge_items=32,minimum_bridge_coverage=.6,maximum_silence_gap_seconds=.35,competing_regular_tolerance_seconds=.02,anchor_margin_seconds=.04,change_existing_labels=False,change_text=False,change_times=False)

def valid_interval(item):
 try:a,b=float(item['start_time']),float(item['end_time'])
 except (KeyError,TypeError,ValueError):return None
 return (a,b) if math.isfinite(a) and math.isfinite(b) and 0<=a<b else None

def overlaps(segments,a,b):
 totals=Counter()
 for s in segments:
  amount=max(0.,min(b,s['end'])-max(a,s['start']))
  if amount:totals[s['speaker']]+=amount
 return totals

def stable(segments,regular,label,a,b):
 if not 0<=a<b:return False
 total=overlaps(segments,a,b)
 if any(s!=label and amount>1e-7 for s,amount in total.items()):return False
 if total[label] < CONFIG['minimum_bridge_coverage']*(b-a):return False
 if any(s!=label and amount>CONFIG['competing_regular_tolerance_seconds'] for s,amount in overlaps(regular,a,b).items()):return False
 cursor=a
 for s in segments:
  left,right=max(a,s['start']),min(b,s['end'])
  if right<=left:continue
  if left-cursor>CONFIG['maximum_silence_gap_seconds']:return False
  cursor=max(cursor,right)
 return b-cursor<=CONFIG['maximum_silence_gap_seconds']

def initial_refine(items,exclusive,regular):
 original=deepcopy(items);result=deepcopy(items);changes=[]
 for i,z in enumerate(original):
  if z['type']!='pronunciation' or z.get('speaker_label'):continue
  interval=valid_interval(z)
  if interval is None:continue
  a,b=interval;totals=overlaps(exclusive,a,b)
  if not totals:continue
  label,amount=totals.most_common(1)[0]
  if amount/(sum(totals.values()))>=CONFIG['overlap_purity'] and amount/(b-a)>=CONFIG['minimum_word_coverage']:
   result[i]['speaker_label']=label;changes.append(dict(item=i,speaker_label=label,method='dominant-overlap'))
 lexical=[i for i,z in enumerate(original) if z['type']=='pronunciation']
 cursor=0
 while cursor<len(lexical):
  if original[lexical[cursor]].get('speaker_label'):cursor+=1;continue
  end=cursor
  while end<len(lexical) and not original[lexical[end]].get('speaker_label'):end+=1
  indices=lexical[cursor:end]
  if cursor>0 and end<len(lexical) and len(indices)<=CONFIG['maximum_bridge_items']:
   left,right=original[lexical[cursor-1]],original[lexical[end]]
   label=left.get('speaker_label');lt,rt=valid_interval(left),valid_interval(right)
   if label and label==right.get('speaker_label') and lt and rt and lt[0]<=rt[0] and 0<=rt[0]-lt[1]<=CONFIG['maximum_bridge_seconds']:
    a=max(0.,lt[1]-CONFIG['anchor_margin_seconds']);b=rt[0]+CONFIG['anchor_margin_seconds']
    # Do not bridge over an already recovered contradictory hypothesis.
    if not any(result[i].get('speaker_label') not in (None,label) for i in indices) and stable(exclusive,regular,label,a,b):
     for i in indices:
      if not result[i].get('speaker_label'):
       result[i]['speaker_label']=label;changes.append(dict(item=i,speaker_label=label,method='stable-bridge',left_anchor=lexical[cursor-1],right_anchor=lexical[end]))
  cursor=end
 for old,new in zip(original,result):
  a,b=deepcopy(old),deepcopy(new);a.pop('speaker_label',None);b.pop('speaker_label',None)
  assert a==b
  if old.get('speaker_label'):assert old==new
 return result,changes


def boundaries(items):
 lexical=[z for z in items if z['type']=='pronunciation']
 return {i for i,(a,b) in enumerate(zip(lexical,lexical[1:])) if a.get('speaker_label') and b.get('speaker_label') and a['speaker_label']!=b['speaker_label']}

def preserve_boundaries(original,result,changes):
 lexical=[i for i,z in enumerate(original) if z['type']=='pronunciation']
 before=boundaries(original)
 for j,(left,right) in enumerate(zip(lexical,lexical[1:])):
  a,b=result[left].get('speaker_label'),result[right].get('speaker_label')
  if a and b and a!=b and j not in before:
   for i in [left,right]:
    if not original[i].get('speaker_label'):result[i].pop('speaker_label',None)
 assert boundaries(result)==before
 return result,[c for c in changes if result[c['item']].get('speaker_label')==c['speaker_label']]

def refine(items,exclusive,regular):
 result,changes=initial_refine(items,exclusive,regular)
 return preserve_boundaries(items,result,changes)
