"""G2A command line: compile call specs, check reproducibility or validate files on disk.

Exit codes follow G1 and fyp-validate: 0 success, 1 invalid or differing
outputs, 2 input/usage errors (including a G0/G1 state that is not valid).
"""

import argparse
import sys
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .compile import load_g1_sources
from .outputs import check_outputs, generate, read_disk, write_specs
from .validate import validate_spec_set


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset release directory")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Replace the complete set of G2A outputs")
    mode.add_argument("--check", action="store_true", help="Regenerate in memory and compare with disk; write nothing")
    mode.add_argument("--validate", action="store_true", help="Validate the files on disk against G1; write nothing")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            src = load_g1_sources(args.dataset_root, args.schema_dir)
            if errors := validate_spec_set(src, read_disk(args.dataset_root), args.schema_dir):
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(f"G2A outputs on disk are valid ({len(src.call_ids)} call specs, reconstruct G1 byte-for-byte).")
            return 0
        src, texts, summary = generate(args.dataset_root, args.schema_dir)
        if args.check:
            differences = check_outputs(args.dataset_root, texts)
            if any(differences.values()):
                for kind, paths in differences.items():
                    for path in paths:
                        print(f"{kind}: {path}", file=sys.stderr)
                return 1
            print(f"G2A outputs reproduce byte-for-byte ({len(texts)} files).")
            return 0
        written = write_specs(args.dataset_root, texts, overwrite=args.overwrite)
        print(f"Wrote {len(written)} files: {summary['call_specs']} call specs plus the record.")
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
