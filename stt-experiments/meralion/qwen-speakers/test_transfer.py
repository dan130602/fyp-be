from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'evaluation-engine'))
from pathlib import Path
import sys,unittest
from copy import deepcopy
from refine import refine
from contract_guard import equivalent,signature

def word(text,label=None):
 z=dict(type='pronunciation',alternatives=[dict(content=text)])
 if label:z['speaker_label']=label
 return z
class TransferSafety(unittest.TestCase):
 def fixture(self):
  q=[word(w,'a') for w in 'alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima november oscar papa quebec romeo sierra tango uniform victor whiskey yankee zebra'.split()];a=deepcopy(q)
  for z in a[8:14]:z['speaker_label']='b'
  return q,a
 def test_word_transfer_changes_boundary_without_changing_normalized_text(self):
  q,a=self.fixture();r,c=refine(q,[],[],a);self.assertTrue(c);self.assertEqual(r[10]['speaker_label'],'b');self.assertTrue(equivalent(q,r))
 def test_no_words_times_or_original_objects_changed(self):
  q,a=self.fixture();original=deepcopy(q);q[10].update(start_time='1.0',end_time='1.5');original=deepcopy(q);r,c=refine(q,[],[],a);self.assertEqual(q,original)
  for x,y in zip(q,r):
   x,y=deepcopy(x),deepcopy(y);x.pop('speaker_label',None);y.pop('speaker_label',None);self.assertEqual(x,y)
 def test_repeated_text_rejected(self):
  q,a=self.fixture()
  for z in q+a:z['alternatives'][0]['content']='yes'
  self.assertEqual(refine(q,[],[],a)[0],q)
 def test_no_auxiliary_no_change(self):
  q,_=self.fixture();self.assertEqual(refine(q,[],[],[])[0],q)
 def test_date_contract_rejects_changed_grouping(self):
  q=[word(w,'a') for w in 'twenty seventh of june nineteen fifty nine'.split()];r=deepcopy(q);r[3]['speaker_label']='b';self.assertFalse(equivalent(q,r))
 def test_identical_labels_identical_signature(self):
  q,_=self.fixture();self.assertEqual(signature(q),signature(deepcopy(q)))
class NearbyASRErrorSafety(TransferSafety):
 def test_matching_context_survives_nearby_asr_error(self):
  q,a=self.fixture();a[9]['alternatives'][0]['content']='julieZ'
  r,c=refine(q,[],[],a);self.assertEqual(r[10]['speaker_label'],'b');self.assertEqual(r[9]['alternatives'][0]['content'],'juliet');self.assertTrue(equivalent(q,r))
if __name__=='__main__':unittest.main()
