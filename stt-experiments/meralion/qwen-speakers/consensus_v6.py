"""Predicted-input-only label correction; no gold, IDs, roles or score inputs."""
from copy import deepcopy
from bisect import bisect_left
from context_refine import correspondence
from base_refine import boundaries
import unicodedata
CONFIG=dict(method='punctuation-aware-auxiliary-consensus-v6',minimum_component_matches=5,minimum_component_context_characters=32,minimum_inverse_agreement=.95,minimum_bridge_anchor_context=24,maximum_bridge_items=256,maximum_bridge_characters=1024,maximum_bridge_edit_fraction=.35,change_words=False,change_times=False,preserve_nonpunctuation_speaker_boundaries=True,requires_auxiliary_turn_asr=True)

def hard_breaks(items):
 result=set();lexical=-1
 for item in items:
  if item['type']=='pronunciation':lexical+=1
  else:
   mark=unicodedata.normalize('NFKC',item['alternatives'][0]['content']).replace('\u3002','.').replace('\u3001',',')
   if lexical>=0 and mark in {'.','?','!',';',':'}:result.add(lexical)
   elif lexical>=0 and mark==',':result.discard(lexical)
 return result

def effective_boundaries(items):return boundaries(items)-hard_breaks(items)

def preserve_boundaries(original,result,changes):
 lexical=[i for i,z in enumerate(original) if z['type']=='pronunciation'];hard=hard_breaks(original);before=effective_boundaries(original)
 for j,(left,right) in enumerate(zip(lexical,lexical[1:])):
  if j in hard:continue
  a,b=result[left].get('speaker_label'),result[right].get('speaker_label')
  if a and b and a!=b and j not in before:
   for i in [left,right]:
    if not original[i].get('speaker_label'):result[i].pop('speaker_label',None)
 assert effective_boundaries(result)==before
 return result,[c for c in changes if result[c['item']].get('speaker_label')==c['speaker_label']]

def refine(items,exclusive,regular,auxiliary=None):
 result=deepcopy(items);changes=[]
 if not auxiliary:return result,changes
 q,spans,a,owners,matches=correspondence(items,auxiliary)
 # Overlap-aware uniqueness rejects repeated text even when str.count would
 # count only one non-overlapping occurrence. Use full independent context.
 def unique_anchor(i):
  left,right=spans[i];context=q[max(0,left-24):min(len(q),right+24)]
  return len(context)>=24 and q.find(context)==q.rfind(context) and a.find(context)>=0 and a.find(context)==a.rfind(context)
 matches={i:m for i,m in matches.items() if unique_anchor(i)}
 lexical=[i for i,z in enumerate(items) if z['type']=='pronunciation'];hard=hard_breaks(items)
 labels={z.get('speaker_label') for z in auxiliary if z.get('speaker_label')}|{z.get('speaker_label') for z in items if z.get('speaker_label')}
 # Correct a whole labelled component by a two-label permutation. Its internal
 # boundary positions stay fixed; unlabelled neighbours isolate its edges.
 if len(labels)==2:
  label_list=sorted(labels);inverse={label_list[0]:label_list[1],label_list[1]:label_list[0]};k=0
  while k<len(lexical):
   if not items[lexical[k]].get('speaker_label'):k+=1;continue
   end=k
   while end<len(lexical) and items[lexical[end]].get('speaker_label'):
    end+=1
    if end-1 in hard:break
   component=lexical[k:end]
   evidence=[i for i in component if i in matches and matches[i]['context_characters']>=CONFIG['minimum_component_context_characters']]
   votes=sum(matches[i]['label']==inverse[items[i]['speaker_label']] for i in evidence)
   if len(evidence)>=CONFIG['minimum_component_matches'] and votes/len(evidence)>=CONFIG['minimum_inverse_agreement']:
    for i in component:
     label=inverse[items[i]['speaker_label']];result[i]['speaker_label']=label;changes.append(dict(item=i,speaker_label=label,previous_label=items[i]['speaker_label'],method='whole-component-auxiliary-consensus',support=votes,evidence=len(evidence)))
   k=end
 assert effective_boundaries(result)==effective_boundaries(items)
 frozen=deepcopy(result)
 for i,m in matches.items():
  if frozen[i]['type']=='pronunciation' and not frozen[i].get('speaker_label'):
   result[i]['speaker_label']=m['label'];changes.append(dict(item=i,speaker_label=m['label'],method='punctuation-guarded-mutual-context'))
 # Recover long alignment dropouts only between independent unique text anchors
 # inside one predicted auxiliary speaker span, with bounded text disagreement.
 from rapidfuzz.distance import Levenshtein
 anchors=sorted(i for i,m in matches.items() if m['context_characters']>=CONFIG['minimum_bridge_anchor_context'])
 position={i:k for k,i in enumerate(lexical)}
 for i in lexical:
  if frozen[i].get('speaker_label'):continue
  k=bisect_left(anchors,i)
  if k==0 or k==len(anchors):continue
  li,ri=anchors[k-1],anchors[k]
  if ri==i:
   if k+1==len(anchors):continue
   ri=anchors[k+1]
  if position[ri]-position[li]>CONFIG['maximum_bridge_items']:continue
  left,right=matches[li],matches[ri]
  if left['label']!=right['label']:continue
  qa,qb=spans[li][1],spans[ri][0];aa,ab=left['aux_end'],right['aux_start']
  if not (qa<=spans[i][0]<=spans[i][1]<=qb):continue
  if not 0<=qb-qa<=CONFIG['maximum_bridge_characters'] or not 0<=ab-aa<=CONFIG['maximum_bridge_characters']:continue
  if {auxiliary[owners[j]].get('speaker_label') for j in range(left['aux_start'],right['aux_end'])}!={left['label']}:continue
  if Levenshtein.normalized_distance(q[qa:qb],a[aa:ab])>CONFIG['maximum_bridge_edit_fraction']:continue
  label=left['label'];result[i]['speaker_label']=label;changes.append(dict(item=i,speaker_label=label,method='bounded-long-dropout-bridge',left_anchor=li,right_anchor=ri))
 # Guard additions relative to corrected components; no new normalization breaks.
 additions=[z for z in changes if z['method'] in ['bounded-long-dropout-bridge','punctuation-guarded-mutual-context']]
 result,kept=preserve_boundaries(frozen,result,additions)
 changes=[z for z in changes if z['method'] not in ['bounded-long-dropout-bridge','punctuation-guarded-mutual-context']]+kept
 assert effective_boundaries(result)==effective_boundaries(items)
 for x,y in zip(items,result):
  x,y=deepcopy(x),deepcopy(y);x.pop('speaker_label',None);y.pop('speaker_label',None);assert x==y
 return result,changes
