"""Print read-only Markdown reports for a multilingual release (writes nothing unless --output is given).

    python scripts/report_multilingual_plan.py --dataset-root datasets/advisory_execution_multilingual_v1 --compatibility
    python scripts/report_multilingual_plan.py --dataset-root datasets/advisory_execution_multilingual_v1 --composition
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.planning.reports import compatibility_report, composition_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--compatibility", action="store_true", help="Edge-case x language-profile matrix from the catalog")
    kind.add_argument("--composition", action="store_true", help="Planned composition from planning_record.json")
    parser.add_argument("--output", type=Path, help="Write the report to this file instead of standard output")
    args = parser.parse_args(argv)
    text = compatibility_report(args.dataset_root) if args.compatibility else composition_report(args.dataset_root)
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
