"""Resolve the integer quotas of a multilingual release from authoring/quota_design.json into authoring/generation_plan.json.

    python scripts/resolve_multilingual_quotas.py --dataset-root datasets/advisory_execution_multilingual_v1            # print
    python scripts/resolve_multilingual_quotas.py --dataset-root datasets/advisory_execution_multilingual_v1 --check    # stored == recomputed
    python scripts/resolve_multilingual_quotas.py --dataset-root datasets/advisory_execution_multilingual_v1 --write    # replace the quotas

--write validates the whole resolved G0 set first and refuses on any error. After it, G1, G2A and G2B must be regenerated
with their own tools; nothing here edits a generated artifact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.planning.quotas_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
