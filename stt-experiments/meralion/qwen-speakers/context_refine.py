"""Prediction-only mutual text alignment and auxiliary single-speaker bridges.
Extends the frozen98% candidate without replacing its labels/text/timestamps.
"""
from copy import deepcopy
from prior_refine import refine as previous_refine,stream,CONFIG as PRIOR_CONFIG
from base_refine import preserve_boundaries
CONFIG=dict(PRIOR_CONFIG,method='mutual-context-and-auxiliary-turn-bridge-v2',minimum_mutual_unique_context_characters=12,minimum_mutual_side_context_characters=3,maximum_auxiliary_bridge_items=64,maximum_auxiliary_bridge_characters=256,preserve_prior_recovered_labels=True,require_mutual_character_alignment=True,one_sided_unique_context_characters=24)

def correspondence(items,auxiliary):
 from rapidfuzz.distance import Levenshtein
 q,spans,_=stream(items);a,_,owners=stream(auxiliary)
 forward={};reverse={};blocks=[]
 for o in Levenshtein.opcodes(q,a):
  if o.tag=='equal':
   for k in range(o.src_end-o.src_start):forward[o.src_start+k]=o.dest_start+k
   blocks.append(o)
 for o in Levenshtein.opcodes(a,q):
  if o.tag=='equal':
   for k in range(o.src_end-o.src_start):reverse[o.src_start+k]=o.dest_start+k
 matches={}
 for i,(start,end) in enumerate(spans):
  if start==end or not all(k in forward and reverse.get(forward[k])==k for k in range(start,end)):continue
  target=forward[start]
  if any(forward[k]!=target+k-start for k in range(start,end)):continue
  labels={auxiliary[owners[forward[k]]].get('speaker_label') for k in range(start,end)}
  if len(labels)!=1 or None in labels:continue
  for b in blocks:
   if not b.src_start<=start<end<=b.src_end:continue
   left=max(b.src_start,start-24);right=min(b.src_end,end+24)
   if right-left<CONFIG['minimum_mutual_unique_context_characters']:continue
   balanced=(start-left>=CONFIG['minimum_mutual_side_context_characters'] or left==0) and (right-end>=CONFIG['minimum_mutual_side_context_characters'] or right==len(q))
   one_sided=right-left>=CONFIG['one_sided_unique_context_characters'] and max(start-left,right-end)>=CONFIG['one_sided_unique_context_characters']
   if not (balanced or one_sided):continue
   context=q[left:right]
   if q.count(context)!=1 or a.count(context)!=1:continue
   matches[i]=dict(label=next(iter(labels)),aux_start=target,aux_end=target+end-start,context_characters=right-left)
   break
 return q,spans,a,owners,matches

def refine(items,exclusive,regular,auxiliary=None):
 frozen,old_changes=previous_refine(items,exclusive,regular,auxiliary)
 if auxiliary is None:return frozen,old_changes
 result=deepcopy(frozen);changes=[]
 q,spans,a,owners,matches=correspondence(items,auxiliary)
 for i,z in enumerate(result):
  if z['type']=='pronunciation' and not z.get('speaker_label') and i in matches:
   m=matches[i];z['speaker_label']=m['label'];changes.append(dict(item=i,speaker_label=m['label'],method='mutual-unique-context',context_characters=m['context_characters']))
 # Freeze matched original labels as anchors; no cascading or new-label anchors.
 lexical=[i for i,z in enumerate(frozen) if z['type']=='pronunciation'];cursor=0
 while cursor<len(lexical):
  if frozen[lexical[cursor]].get('speaker_label'):cursor+=1;continue
  end=cursor
  while end<len(lexical) and not frozen[lexical[end]].get('speaker_label'):end+=1
  indices=lexical[cursor:end]
  if cursor>0 and end<len(lexical) and len(indices)<=CONFIG['maximum_auxiliary_bridge_items']:
   li,ri=lexical[cursor-1],lexical[end];left,right=matches.get(li),matches.get(ri)
   if left and right:
    label=frozen[li].get('speaker_label')
    if label and label==frozen[ri].get('speaker_label')==left['label']==right['label']:
     qa,qb=spans[li][1],spans[ri][0];aa,ab=left['aux_end'],right['aux_start']
     if 0<=qb-qa<=CONFIG['maximum_auxiliary_bridge_characters'] and 0<=ab-aa<=CONFIG['maximum_auxiliary_bridge_characters']:
      aux_labels={auxiliary[owners[k]].get('speaker_label') for k in range(left['aux_start'],right['aux_end'])}
      if aux_labels=={label} and not any(result[i].get('speaker_label') not in (None,label) for i in indices):
       for i in indices:
        if not result[i].get('speaker_label'):
         result[i]['speaker_label']=label;changes.append(dict(item=i,speaker_label=label,method='auxiliary-same-turn-bridge',left_anchor=li,right_anchor=ri))
  cursor=end
 result,changes=preserve_boundaries(frozen,result,changes)
 for before,after in zip(frozen,result):
  x,y=deepcopy(before),deepcopy(after);x.pop('speaker_label',None);y.pop('speaker_label',None);assert x==y
  if before.get('speaker_label'):assert before==after
 return result,old_changes+changes
