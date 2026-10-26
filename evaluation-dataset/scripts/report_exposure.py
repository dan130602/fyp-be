"""Derive the exposure ledger of a multilingual release: which calls were sent to a real provider, and under which code.

Read-only for the audit store and provenance. Prints the ledger, or with --write stores it at authoring/exposure_log.json
(commit it after each live check; the audit store itself is gitignored). Only live attempts count.

    python scripts/report_exposure.py --dataset-root datasets/advisory_execution_multilingual_v1
    python scripts/report_exposure.py --dataset-root datasets/advisory_execution_multilingual_v1 --write
    python scripts/report_exposure.py --dataset-root datasets/advisory_execution_multilingual_v1 --check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.llm import exposure
from fyp_evaluation.llm.fsutil import write_text_atomic
from fyp_evaluation.llm.stage_runner import DEFAULT_AUDIT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_ROOT, help="Gitignored operational store (default: .llm_audit)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write authoring/exposure_log.json")
    mode.add_argument("--check", action="store_true", help="Exit 1 when the committed ledger differs from what the records show")
    args = parser.parse_args(argv)
    root = args.dataset_root
    dataset_id = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["dataset_id"]
    text = exposure.render(root, args.audit_dir, dataset_id)
    target = root / exposure.LEDGER
    if args.write:
        write_text_atomic(target, text)
        print(f"Wrote {target}")
        return 0
    if args.check:
        current = target.read_text(encoding="utf-8").replace("\r\n", "\n") if target.is_file() else None
        if current != text:
            print(f"{exposure.LEDGER} is missing or differs from the attempt records; run with --write", file=sys.stderr)
            return 1
        print("The exposure ledger matches the attempt records.")
        return 0
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
