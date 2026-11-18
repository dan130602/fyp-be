"""Auxiliary predicted-transcript corroboration; no references or score inputs.
Deployment needs the extra speaker-turn ASR pass. Never replaces Qwen words.
"""
from copy import deepcopy
import unicodedata
from base_refine import refine as base_refine,preserve_boundaries,CONFIG as BASE_CONFIG
CONFIG=dict(BASE_CONFIG,method='boundary-preserving-predicted-transcript-corroboration-v1',minimum_unique_context_characters=24,minimum_context_each_side_characters=6,requires_extra_speaker_turn_ASR=True,auxiliary_uses_references=False)

def canonical(text):return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())
def stream(items):
 chars=[];ranges=[];owners=[]
 for i,z in enumerate(items):
  word=canonical(z['alternatives'][0]['content']) if z['type']=='pronunciation' else ''
  a=len(chars);chars.extend(word);owners.extend([i]*len(word));ranges.append((a,len(chars)))
 return ''.join(chars),ranges,owners

def refine(items,exclusive,regular,auxiliary=None):
 result,changes=base_refine(items,exclusive,regular)
 if auxiliary is None:return result,changes
 from rapidfuzz.distance import Levenshtein
 q,spans,_=stream(items);a,_,owners=stream(auxiliary)
 blocks=[o for o in Levenshtein.opcodes(q,a) if o.tag=='equal' and o.src_end-o.src_start>=CONFIG['minimum_unique_context_characters']]
 for i,z in enumerate(result):
  if z['type']!='pronunciation' or z.get('speaker_label'):continue
  start,end=spans[i]
  if start==end:continue
  for b in blocks:
   if not b.src_start<=start<end<=b.src_end:continue
   left=max(b.src_start,start-12);right=min(b.src_end,end+12)
   if right-left<CONFIG['minimum_unique_context_characters']:continue
   if start-left<CONFIG['minimum_context_each_side_characters'] and left>0:continue
   if right-end<CONFIG['minimum_context_each_side_characters'] and right<len(q):continue
   context=q[left:right]
   if q.count(context)!=1 or a.count(context)!=1:continue
   target_start=b.dest_start+start-b.src_start;target_end=target_start+end-start
   labels={auxiliary[owners[k]].get('speaker_label') for k in range(target_start,target_end)}
   if len(labels)!=1 or None in labels:continue
   label=next(iter(labels));z['speaker_label']=label
   changes.append(dict(item=i,speaker_label=label,method='auxiliary-predicted-text-context',equal_context_characters=right-left))
   break
 return preserve_boundaries(items,result,changes)
