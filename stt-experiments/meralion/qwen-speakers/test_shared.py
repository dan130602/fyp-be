import unittest
from shared_recovery import recover
class RecoveryTests(unittest.TestCase):
 def fixture(self):
  return dict(text='one two',units=[dict(text='one two',start_time=.2,end_time=.8)],offset=0,duration=1,exclusive=[dict(start=0,end=1,speaker='A')],regular=[dict(start=0,end=1,speaker='A')],items=[dict(type='pronunciation',alternatives=[dict(content=t)]) for t in ['one','two']],audit=[dict(start_char=0,end_char=3,reason='alignment_unit_spans_multiple_items'),dict(start_char=4,end_char=7,reason='alignment_unit_spans_multiple_items')])
 def test_shared_unit_recovers_without_fabricated_times(self):
  f=self.fixture();r,c=recover(**f);self.assertEqual(len(c),2);self.assertTrue(all(x['speaker_label']=='A' and 'start_time' not in x for x in r));self.assertTrue(all('speaker_label' not in x for x in f['items']))
 def test_competing_overlap_rejected(self):
  f=self.fixture();f['regular'].append(dict(start=.3,end=.5,speaker='B'));self.assertEqual(recover(**f)[1],[])
 def test_cross_speaker_unit_rejected(self):
  f=self.fixture();f['exclusive']=[dict(start=0,end=.5,speaker='A'),dict(start=.5,end=1,speaker='B')];self.assertEqual(recover(**f)[1],[])
 def test_speech_gap_rejected(self):
  f=self.fixture();f['exclusive']=[dict(start=0,end=.4,speaker='A'),dict(start=.5,end=1,speaker='A')];self.assertEqual(recover(**f)[1],[])
 def test_text_mismatch_rejected(self):
  f=self.fixture();f['units'][0]['text']='something else';self.assertEqual(recover(**f)[1],[])
 def test_invalid_time_rejected(self):
  for a,b in [(0,0),(.8,.2),(-.1,.5),(.2,1.1),(float('nan'),.8)]:
   f=self.fixture();f['units'][0].update(start_time=a,end_time=b);self.assertEqual(recover(**f)[1],[])
 def test_long_span_rejected(self):
  f=self.fixture();f['duration']=4;f['units'][0]['end_time']=3;self.assertEqual(recover(**f)[1],[])
 def test_existing_labels_immutable(self):
  f=self.fixture();f['items'][0]['speaker_label']='B';r,c=recover(**f);self.assertEqual(r[0],f['items'][0]);self.assertEqual(len(c),1)
if __name__=='__main__':unittest.main()
