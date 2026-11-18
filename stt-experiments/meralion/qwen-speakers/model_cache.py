"""Check all required local checkpoint files before any GPU model load."""
from pathlib import Path
import hashlib
REQUIRED=('config.yaml','segmentation/pytorch_model.bin','embedding/pytorch_model.bin')
def validate_snapshot(root,expected_config_sha256=None):
    root=Path(root)
    missing=[name for name in REQUIRED if not (root/name).is_file() or (root/name).stat().st_size==0]
    if missing:raise FileNotFoundError('Incomplete pinned model snapshot: '+', '.join(missing))
    if expected_config_sha256:
        assert hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest()==expected_config_sha256,'Pinned config differs from pilot'
    return root
