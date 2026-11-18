"""Predicted text correspondence for word-level speaker labels; no gold/IDs/scores."""
from copy import deepcopy
from bisect import bisect_left
from context_refine import correspondence
from contract_guard import equivalent
CONFIG=dict(method='mutual-word-label-transfer-v8',minimum_unique_context_characters=12,maximum_bridge_characters=256,maximum_bridge_items=64,maximum_bridge_edit_fraction=.25,requires_auxiliary_turn_asr=True,change_words=False,change_times=False,hypothesis_normalization_origin_contract=True,no_gold_or_score_inputs=True)

def refine(items,exclusive,regular,auxiliary=None):
 if not auxiliary:return deepcopy(items),[]
 q,spans,a,owners,matches=correspondence(items,auxiliary)
 def unique(i):
  left,right=spans[i];target_left,target_right=matches[i]['aux_start'],matches[i]['aux_end']
  # Use the actually matching context on each side, not a fixed48-character
  # window that discards an otherwise sound match near one ASR text error.
  before=after=0
  while before<24 and left-before>0 and target_left-before>0 and q[left-before-1]==a[target_left-before-1]:before+=1
  while after<24 and right+after<len(q) and target_right+after<len(a) and q[right+after]==a[target_right+after]:after+=1
  context=q[left-before:right+after]
  return len(context)>=CONFIG['minimum_unique_context_characters'] and q.find(context)==q.rfind(context) and a.find(context)>=0 and a.find(context)==a.rfind(context)
 matches={i:m for i,m in matches.items() if unique(i)}
 result=deepcopy(items);changes=[]
 for i,m in matches.items():
  if result[i].get('speaker_label')!=m['label']:
   result[i]['speaker_label']=m['label'];changes.append(dict(item=i,speaker_label=m['label'],previous_label=items[i].get('speaker_label'),method='mutual-unique-word-transfer'))
 from rapidfuzz.distance import Levenshtein
 anchors=sorted(i for i,m in matches.items() if m['context_characters']>=24)
 lexical=[i for i,z in enumerate(items) if z['type']=='pronunciation'];position={i:k for k,i in enumerate(lexical)}
 for i in lexical:
  if i in matches:continue
  k=bisect_left(anchors,i)
  if k==0 or k==len(anchors):continue
  li,ri=anchors[k-1],anchors[k];left,right=matches[li],matches[ri]
  if left['label']!=right['label'] or position[ri]-position[li]>CONFIG['maximum_bridge_items']:continue
  qa,qb=spans[li][1],spans[ri][0];aa,ab=left['aux_end'],right['aux_start']
  if not (qa<=spans[i][0]<=spans[i][1]<=qb):continue
  if not 0<=qb-qa<=CONFIG['maximum_bridge_characters'] or not 0<=ab-aa<=CONFIG['maximum_bridge_characters']:continue
  if {auxiliary[owners[j]].get('speaker_label') for j in range(left['aux_start'],right['aux_end'])}!={left['label']}:continue
  if Levenshtein.normalized_distance(q[qa:qb],a[aa:ab])>CONFIG['maximum_bridge_edit_fraction']:continue
  if result[i].get('speaker_label')!=left['label']:
   result[i]['speaker_label']=left['label'];changes.append(dict(item=i,speaker_label=left['label'],previous_label=items[i].get('speaker_label'),method='same-auxiliary-turn-text-bridge',left_anchor=li,right_anchor=ri))
 for before,after in zip(items,result):
  x,y=deepcopy(before),deepcopy(after);x.pop('speaker_label',None);y.pop('speaker_label',None);assert x==y
 # A uniform production input contract, not a comparison to reference/scores.
 # Reject a whole proposal if label-induced phrase boundaries change normalized
 # words or any word's source-item mapping. No per-call quality oracle is used.
 if not equivalent(items,result):return deepcopy(items),[dict(method='proposal-rejected-hypothesis-normalization-contract',proposed_label_changes=len(changes))]
 return result,changes
