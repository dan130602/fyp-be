import unittest,warnings
import numpy as np
from core import turn_chunks,RATE
class BoundaryTests(unittest.TestCase):
 def test_wholly_after_end_is_audited_no_audio_dropped(self):
  audit=[]
  with warnings.catch_warnings(record=True) as caught:
   p=turn_chunks(np.ones(RATE),[dict(start=0,end=.5,speaker='A'),dict(start=1.01,end=1.05,speaker='B')],audit=audit)
  self.assertEqual(p,[dict(start_sample=0,end_sample=8000,speaker='A',turn_index=0)]);self.assertEqual(audit[0]['audio_samples_removed'],0);self.assertEqual(len(caught),1)
 def test_exact_end_start_is_outside_audio(self):
  audit=[]
  with warnings.catch_warnings(record=True):p=turn_chunks(np.ones(RATE),[dict(start=1,end=1.1,speaker='A')],audit=audit)
  self.assertEqual(p,[]);self.assertEqual(len(audit),1)
 def test_crossing_end_preserves_real_audio(self):
  p=turn_chunks(np.ones(RATE),[dict(start=.9,end=1.1,speaker='A')]);self.assertEqual((p[0]['start_sample'],p[0]['end_sample']),(14400,16000))
 def test_empty_turn_within_audio_still_fails(self):
  with self.assertRaises(AssertionError):turn_chunks(np.ones(RATE),[dict(start=.5,end=.5,speaker='A')])
 def test_subsample_in_range_turn_still_fails(self):
  with self.assertRaises(ValueError):turn_chunks(np.ones(RATE),[dict(start=.1,end=.100001,speaker='A')])
 def test_negative_and_overlapping_input_still_fail(self):
  with self.assertRaises(AssertionError):turn_chunks(np.ones(RATE),[dict(start=-.1,end=.1,speaker='A')])
  with self.assertRaises(AssertionError):turn_chunks(np.ones(RATE),[dict(start=0,end=.6,speaker='A'),dict(start=.5,end=.9,speaker='B')])
if __name__=='__main__':unittest.main()
