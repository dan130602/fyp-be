"""Deterministic transcript and speaker-role conversion from canonical ground truth.

Cross-file validation against ground truth (`validate_transcript_pair`) is
not re-exported here: it lives at `fyp_evaluation.validation.transcript` and
must be imported from there directly. Re-exporting it from this __init__
would recreate the package-init cycle described in
`fyp_evaluation.validation.__init__` (this package's own __init__ importing
a module that imports back into this package while it is still loading).
"""

from .converter import (
    PLACEHOLDER_TIME, SPEAKER_LABELS, SPEAKER_ROLES, TRANSCRIPT_SEPARATOR,
    build_transcript_artifacts,
)

__all__ = [
    "PLACEHOLDER_TIME", "SPEAKER_LABELS", "SPEAKER_ROLES", "TRANSCRIPT_SEPARATOR",
    "build_transcript_artifacts",
]
