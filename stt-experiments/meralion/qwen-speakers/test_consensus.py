from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'evaluation-engine'))
import sys,unittest
from pathlib import Path
from copy import deepcopy
from consensus_v6 import refine
from base_refine import boundaries

def fixture():
 words='alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima november oscar papa quebec romeo sierra tango uniform victor whiskey yankee zebra'.split()
 q=[dict(type='pronunciation',alternatives=[dict(content=w)],speaker_label='a',start_time=str(i),end_time=str(i+.5)) for i,w in enumerate(words)]
 aux=deepcopy(q)
 for z in aux:z['speaker_label']='b'
 return q,aux
class ConsensusSafety(unittest.TestCase):
 def test_supported_component_is_corrected(self):
  q,a=fixture();r,c=refine(q,[],[],a);self.assertTrue(c);self.assertTrue(all(z['speaker_label']=='b' for z in r));self.assertEqual(q[0]['speaker_label'],'a')
 def test_mixed_evidence_rejected(self):
  q,a=fixture()
  for z in a[::2]:z['speaker_label']='a'
  self.assertEqual(refine(q,[],[],a)[0],q)
 def test_fields_frozen(self):
  q,a=fixture();r,_=refine(q,[],[],a)
  for x,y in zip(q,r):
   x,y=deepcopy(x),deepcopy(y);x.pop('speaker_label',None);y.pop('speaker_label',None);self.assertEqual(x,y)
 def test_ambiguous_repetitions_do_not_correct(self):
  q,a=fixture()
  for z in q+a:z['alternatives'][0]['content']='yes'
  self.assertEqual(refine(q,[],[],a)[0],q)
 def test_no_auxiliary_no_change(self):
  q,_=fixture();self.assertEqual(refine(q,[],[],[])[0],q)
 def test_short_component_rejected(self):
  q,a=fixture();self.assertEqual(refine(q[:3],[],[],a[:3])[0],q[:3])
 def test_boundaries_preserved(self):
  q,a=fixture()
  for i in range(10,15):q[i]['speaker_label']='b';a[i]['speaker_label']='a'
  r,c=refine(q,[],[],a);self.assertTrue(c);self.assertEqual(boundaries(q),boundaries(r))
if __name__=='__main__':unittest.main()
