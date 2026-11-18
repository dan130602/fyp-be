"""Only predicted inputs: auxiliary-turn anchors independent of Qwen labels.
Preserves the accepted98.25% candidate, its text/times and all existing labels.
"""
from bisect import bisect_left
from copy import deepcopy
from context_refine import refine as previous_refine,correspondence,CONFIG as PRIOR_CONFIG
from base_refine import preserve_boundaries
CONFIG=dict(PRIOR_CONFIG,method='independent-auxiliary-anchor-v3',minimum_anchor_unique_context_characters=24,maximum_anchor_lexical_span=64,maximum_anchor_character_span=256,require_two_anchors=True,require_single_auxiliary_speaker_between_anchors=True,require_qwen_anchor_label_agreement=False,preserve_prior_recovered_labels=True)

def refine(items,exclusive,regular,auxiliary=None):
 frozen,old_changes=previous_refine(items,exclusive,regular,auxiliary)
 if auxiliary is None:return frozen,old_changes
 result=deepcopy(frozen);changes=[]
 q,spans,a,owners,matches=correspondence(items,auxiliary)
 lexical=[i for i,z in enumerate(items) if z['type']=='pronunciation'];position={i:j for j,i in enumerate(lexical)}
 anchors=sorted(i for i,m in matches.items() if m['context_characters']>=CONFIG['minimum_anchor_unique_context_characters'])
 for i in lexical:
  if frozen[i].get('speaker_label'):continue
  k=bisect_left(anchors,i)
  if k==0 or k==len(anchors):continue
  li,ri=anchors[k-1],anchors[k]
  # A matching item itself needs an independent following anchor.
  if ri==i:
   if k+1==len(anchors):continue
   ri=anchors[k+1]
  if position[ri]-position[li]>CONFIG['maximum_anchor_lexical_span']:continue
  left,right=matches[li],matches[ri]
  if left['label']!=right['label']:continue
  qa,qb=spans[li][1],spans[ri][0];aa,ab=left['aux_end'],right['aux_start']
  if not (qa<=spans[i][0]<=spans[i][1]<=qb):continue
  if not 0<=qb-qa<=CONFIG['maximum_anchor_character_span'] or not 0<=ab-aa<=CONFIG['maximum_anchor_character_span']:continue
  labels={auxiliary[owners[j]].get('speaker_label') for j in range(left['aux_start'],right['aux_end'])}
  if labels!={left['label']}:continue
  label=left['label'];result[i]['speaker_label']=label
  changes.append(dict(item=i,speaker_label=label,method='independent-auxiliary-two-anchor',left_anchor=li,right_anchor=ri,anchor_context_characters=[left['context_characters'],right['context_characters']]))
 result,changes=preserve_boundaries(frozen,result,changes)
 for before,after in zip(frozen,result):
  x,y=deepcopy(before),deepcopy(after);x.pop('speaker_label',None);y.pop('speaker_label',None);assert x==y
  if before.get('speaker_label'):assert before==after
 return result,old_changes+changes
