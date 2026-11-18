import unittest
import numpy as np
from core import turn_chunks,turn_document,RATE
class CoreTests(unittest.TestCase):
 def test_turns_never_pad_across_speakers(self):
  x=np.ones(3*RATE,dtype='float32')*.1;s=[dict(start=.1,end=1,speaker='A'),dict(start=1.1,end=2.9,speaker='B')];c=turn_chunks(x,s);self.assertEqual([(r['start_sample'],r['end_sample'],r['speaker']) for r in c],[(1600,16000,'A'),(17600,46400,'B')])
 def test_long_turns_preserve_coverage_and_model_limit(self):
  c=turn_chunks(np.ones(61*RATE,dtype='float32')*.1,[dict(start=0,end=61,speaker='A')]);self.assertEqual(c[0]['start_sample'],0);self.assertEqual(c[-1]['end_sample'],61*RATE);self.assertTrue(all(a['end_sample']==b['start_sample'] for a,b in zip(c,c[1:])));self.assertTrue(all(r['end_sample']-r['start_sample']<=30*RATE for r in c))
 def test_overlapping_exclusive_turns_fail(self):
  with self.assertRaises(AssertionError):turn_chunks(np.zeros(2*RATE),[dict(start=0,end=1.2,speaker='A'),dict(start=1,end=2,speaker='B')])
 def test_turn_only_output_has_no_fake_word_times(self):
  doc,turns=turn_document([dict(start_sample=0,end_sample=RATE,text='hello world',speaker='A')]);self.assertEqual(len(doc['results']['items']),2);self.assertTrue(all('start_time' not in i and 'confidence' not in i['alternatives'][0] for i in doc['results']['items']));self.assertEqual(turns[0]['end'],1)
 def test_empty_recognized_turn_is_retained(self):
  doc,t=turn_document([dict(start_sample=0,end_sample=RATE,text='',speaker='A')]);self.assertEqual(len(t),1);self.assertEqual(t[0]['text'],'');self.assertEqual(doc['results']['items'],[])
if __name__=='__main__':unittest.main()
