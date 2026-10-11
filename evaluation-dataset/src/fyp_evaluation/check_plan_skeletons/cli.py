"""G2B command line: build skeletons, check reproducibility or validate files on disk.

Exit codes follow G1/G2A: 0 success, 1 invalid or differing outputs, 2 input/usage errors
(including a G0/G1/G2A state that is not valid).
"""

import argparse
import sys
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .inputs import load_inputs
from .outputs import check_outputs, generate, read_disk, write_skeletons
from .validate import validate_skeleton_set


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset release directory")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Replace the complete set of G2B outputs")
    mode.add_argument("--check", action="store_true", help="Regenerate in memory and compare with disk; write nothing")
    mode.add_argument("--validate", action="store_true", help="Validate the files on disk against the call specs; write nothing")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            inputs = load_inputs(args.dataset_root, args.schema_dir)
            if errors := validate_skeleton_set(inputs, read_disk(args.dataset_root), args.schema_dir):
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(f"G2B outputs on disk are valid ({len(inputs.call_ids)} skeletons rebuilt byte-for-byte from the call specs).")
            return 0
        inputs, texts, summary = generate(args.dataset_root, args.schema_dir)
        if args.check:
            differences = check_outputs(args.dataset_root, texts)
            if any(differences.values()):
                for kind, paths in differences.items():
                    for path in paths:
                        print(f"{kind}: {path}", file=sys.stderr)
                return 1
            print(f"G2B outputs reproduce byte-for-byte ({len(texts)} files).")
            return 0
        written = write_skeletons(args.dataset_root, texts, overwrite=args.overwrite)
        print(f"Wrote {len(written)} files: {summary['skeletons']} skeletons plus the record "
              f"({summary['planned_events']} events, {summary['constraints']} constraints).")
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
