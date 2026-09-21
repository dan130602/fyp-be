"""Read-only command line validation."""

import argparse
import sys
from pathlib import Path

from .schemas import (
    SCHEMA_NAMES, check_schemas, load_json, validate, validate_answers,
    validate_checklist,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-dir", type=Path, default=Path("schemas"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schemas", help="Check all schema definitions")
    instance = commands.add_parser("json", help="Validate one JSON document")
    instance.add_argument("schema", choices=SCHEMA_NAMES)
    instance.add_argument("path", type=Path)
    checklist = commands.add_parser("checklist", help="Check internal checklist and field-rule links")
    checklist.add_argument("path", type=Path)
    checklist.add_argument("normalization", type=Path)
    answers = commands.add_parser("answers", help="Check a call document against its checklist")
    answers.add_argument("schema", choices=["call_config", "check_plan", "ground_truth"])
    answers.add_argument("path", type=Path)
    answers.add_argument("checklist", type=Path)
    answers.add_argument("normalization", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "schemas":
            check_schemas(args.schema_dir)
            print(f"All {len(SCHEMA_NAMES)} Draft 2020-12 schema definitions are valid.")
            return 0
        if args.command == "checklist":
            errors = validate_checklist(
                load_json(args.path), load_json(args.normalization), args.schema_dir,
            )
        elif args.command == "answers":
            errors = validate_answers(
                load_json(args.path), args.schema, load_json(args.checklist),
                load_json(args.normalization), args.schema_dir,
            )
        else:
            errors = validate(load_json(args.path), args.schema, args.schema_dir)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("Structurally valid. Approval and semantic verification are separate.")
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
