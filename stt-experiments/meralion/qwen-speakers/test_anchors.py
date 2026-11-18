from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'evaluation-engine'))
import unittest,sys
from pathlib import Path
from copy import deepcopy
from anchor_v3 import refine
from context_refine import refine as previous
class IndependentAnchorSafety(unittest.TestCase):
 def fixture(self):
  words='alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima november oscar papa quebec romeo sierra tango uniform victor whiskey yankee zebra'.split()
  q=[dict(type='pronunciation',alternatives=[dict(content=w)],speaker_label='qwen_b') for w in words];a=deepcopy(q)
  for z in a:z['speaker_label']='aux_a'
  for i in range(8,13):q[i].pop('speaker_label');a[i]['alternatives'][0]['content']='zzzzzzzzzzzz'
  return q,a
 def test_independent_anchors_can_recover_interior(self):
  q,a=self.fixture();old,_=previous(q,[],[],a);self.assertNotIn('speaker_label',old[10])
  new,c=refine(q,[],[],a);self.assertEqual(new[10]['speaker_label'],'aux_a')
  self.assertIn('independent-auxiliary-two-anchor',[z['method'] for z in c])
  self.assertNotIn('start_time',new[10])
 def test_intervening_aux_speaker_rejects(self):
  q,a=self.fixture();a[10]['speaker_label']='aux_other'
  self.assertNotIn('speaker_label',refine(q,[],[],a)[0][10])
 def test_existing_labels_preserved(self):
  q,a=self.fixture();old,_=previous(q,[],[],a);new,_=refine(q,[],[],a)
  for x,y in zip(old,new):
   if x.get('speaker_label'):self.assertEqual(x,y)
 def test_transcript_fields_preserved(self):
  q,a=self.fixture();new,_=refine(q,[],[],a)
  for x,y in zip(q,new):
   x,y=deepcopy(x),deepcopy(y);x.pop('speaker_label',None);y.pop('speaker_label',None);self.assertEqual(x,y)
if __name__=='__main__':unittest.main()
