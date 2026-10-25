"""G1 command line: plan, write, check reproducibility or validate written outputs.

Exit codes follow fyp-validate: 0 success, 1 invalid or differing outputs,
2 input/usage errors or an infeasible plan.
"""

import argparse
import sys
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .outputs import OUTPUTS, check_outputs, generate, validate_written, write_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset release directory")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Replace existing G1 outputs")
    mode.add_argument("--check", action="store_true", help="Regenerate in memory and compare with disk; write nothing")
    mode.add_argument("--validate", action="store_true", help="Validate the outputs on disk; write nothing")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            errors = validate_written(args.dataset_root, args.schema_dir)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print("G1 outputs on disk are valid.")
            return 0
        texts, summary = generate(args.dataset_root, args.schema_dir)
        if args.check:
            if differing := check_outputs(args.dataset_root, texts):
                print("Regenerated G1 outputs differ from disk: " + ", ".join(differing), file=sys.stderr)
                return 1
            print(f"G1 outputs reproduce byte-for-byte ({len(OUTPUTS)} files).")
            return 0
        for path in write_outputs(args.dataset_root, texts, overwrite=args.overwrite):
            print(f"Wrote {path}")
        if "by_language_profile" in summary:
            print(f"G1 plan: {summary['calls']} calls; language profiles {summary['language_profiles']}; "
                  f"{summary['total_compliance_fail_pairs']} FAIL labels; {summary['total_edge_case_assignments']} edge cases.")
        else:
            print(f"G1 plan: {summary['calls']} calls; composition {summary['call_composition']}.")
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
