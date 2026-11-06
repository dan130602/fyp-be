"""An immutable, explicitly identified scope for Codex's manual reviews.

The provider request and legacy Claude packets are unchanged. Codex reads the
same request plus this supplement; its separate hash is checked before ingest
and retained with the response and generation metadata for reconstruction.
"""

import hashlib
from pathlib import Path

CODEX_REVIEW_FORMAT = "codex-review/1"
CODEX_REVIEWER = "codex-harness"
POLICY_VERSION = "codex-review-v1"
POLICY_PATH = Path(__file__).resolve().parents[3] / "prompts/shared/codex_review.v1.md"


def codex_review_policy():
    raw = POLICY_PATH.read_bytes()
    return {"version": POLICY_VERSION, "sha256": hashlib.sha256(raw).hexdigest(), "instructions": raw.decode("utf-8")}
