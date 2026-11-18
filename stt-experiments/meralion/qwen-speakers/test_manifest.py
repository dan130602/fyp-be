import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import base_gpu as b
class ManifestSafety(unittest.TestCase):
 def fixture(self,p):
  audio=p/'sample.wav';audio.write_bytes(b'private-audio-fixture');m=dict(scope='development-only-original-audio',calls=[dict(call_id='CALL_001',audio=audio.name,audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest())]);(p/'manifest.json').write_text(json.dumps(m));return m
 def test_valid_private_manifest(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root)
   with patch.object(b,'HERE',root):self.assertEqual(len(b.preflight()['calls']),1)
 def test_duplicate_ids_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);m=self.fixture(root);m['calls']*=2;(root/'manifest.json').write_text(json.dumps(m))
   with patch.object(b,'HERE',root),self.assertRaises(ValueError):b.preflight()
 def test_different_audio_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root);(root/'sample.wav').write_bytes(b'different')
   with patch.object(b,'HERE',root),self.assertRaises(ValueError):b.preflight()
 def test_path_escape_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);m=self.fixture(root);m['calls'][0]['audio']='../outside.wav';(root/'manifest.json').write_text(json.dumps(m))
   with patch.object(b,'HERE',root),self.assertRaises(ValueError):b.preflight()
if __name__=='__main__':unittest.main()
