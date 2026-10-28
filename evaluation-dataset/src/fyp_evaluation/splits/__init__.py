"""Deterministic whole-call DEVELOPMENT / HELDOUT split of a release (offline, no model calls).

The split is retrospective: it is drawn after generation, so it is not a pre-registered hold-out. Stratification reads only the frozen,
committed G2A call specifications, never generated dialogue or verifier outcomes. The manifest `splits/split.v1.json` is the sole
authority for later export and evaluation; nothing may redraw assignments.
"""

DEVELOPMENT = "DEVELOPMENT"
HELDOUT = "HELDOUT"
SPLITS = (DEVELOPMENT, HELDOUT)
SPLIT_DIR = "splits"
REGISTRY_FILE = "development_exposure.v1.json"
MANIFEST_FILE = "split.v1.json"
REPORT_FILE = "split_report.v1.json"
SEED = 20261028
ALGORITHM = {"name": "constrained-multilabel-greedy-swap", "version": "1.0"}

# Held-out calls per language; development is the complement. These are the integer-rounded 70/30 allocations the user fixed.
HELDOUT_QUOTAS = {"ENGLISH": 23, "MANDARIN": 22, "EN_ZH_MIXED": 15, "SINGLISH": 15}


class SplitError(ValueError):
    """A hard constraint, fingerprint or contract of the split is violated."""
