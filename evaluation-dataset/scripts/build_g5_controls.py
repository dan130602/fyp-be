"""Build the three G5 semantic-control dataset copies (offline; no network, no credentials, production untouched).

    python scripts/build_g5_controls.py --dataset-root datasets/advisory_execution_pilot_v1 --out ../g5_controls

Each control is written to OUT/<name>/ (A_unchanged, B_omission_leak, C_contradiction). Verify one with its own audit dir:
    python scripts/verify_authoring_plans.py --dataset-root OUT/B_omission_leak --audit-dir OUT/audit_B --run --call-id CALL_021
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.plan_verifications.controls import MUTATIONS, build_control  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Empty directory outside the dataset release")
    parser.add_argument("--control", action="append", choices=sorted(MUTATIONS), help="Build only these controls (repeatable)")
    args = parser.parse_args(argv)
    source, out = args.dataset_root.resolve(), args.out.resolve()
    if out == source or source in out.parents:
        parser.error("--out must be outside the dataset release")
    for name in args.control or sorted(MUTATIONS):
        target = out / name
        if target.exists():
            parser.error(f"{target} already exists; choose an empty --out")
        build_control(source, target, name, audit_root=out / f"g4_audit_{name}")
        print(f"built {name}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
