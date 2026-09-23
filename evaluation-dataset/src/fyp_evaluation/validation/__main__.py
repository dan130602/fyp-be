"""Contract validation and explicitly requested deterministic scaffolding."""

import argparse
import json
import sys
from pathlib import Path

from .schemas import (
    SCHEMA_NAMES, check_schemas, load_json, validate, validate_answers,
    validate_checklist,
)
from .call_config import validate_call_config
from .check_plan import validate_check_plan, validate_forbidden_patterns
from .evidence import validate_dialogue_against_plan
from .tokens import validate_critical_token_coverage
from fyp_evaluation.authoring.coverage import PILOT_CALL_IDS, compile_expected, load_matrix
from fyp_evaluation.authoring.call_configs import write_call_configs
from fyp_evaluation.authoring.check_plans import write_check_plans
from fyp_evaluation.authoring.compile import compile_ground_truth


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
    matrix = commands.add_parser("matrix", help="Validate approved CSV structure and typed answers")
    matrix.add_argument("matrix", type=Path)
    call_config = commands.add_parser("call-config", help="Validate a config against the approved matrix")
    call_config.add_argument("path", type=Path)
    call_config.add_argument("matrix", type=Path)
    call_config.add_argument("--complete", action="store_true", help="Also require populated facts and authored intents")
    scaffold = commands.add_parser("scaffold-call-configs", help="Write unfilled matrix-derived configs")
    scaffold.add_argument("matrix", type=Path)
    scaffold.add_argument("output_dir", type=Path)
    scaffold.add_argument("--overwrite", action="store_true", help="Explicitly replace existing config files")
    check_plan = commands.add_parser("check-plan", help="Check a plan against its approved config and matrix")
    check_plan.add_argument("path", type=Path)
    check_plan.add_argument("call_config", type=Path)
    check_plan.add_argument("matrix", type=Path)
    check_plan.add_argument("--complete", action="store_true")
    scaffold_plans = commands.add_parser("scaffold-check-plans", help="Write deterministic evidence placeholders")
    scaffold_plans.add_argument("matrix", type=Path)
    scaffold_plans.add_argument("output_dir", type=Path)
    scaffold_plans.add_argument("--overwrite", action="store_true")
    scaffold_plans.add_argument("--only", nargs="+", help="Select output call IDs without narrowing matrix validation")
    dialogue_plan = commands.add_parser("dialogue-plan", help="Validate exact evidence independently of annotations")
    dialogue_plan.add_argument("dialogue", type=Path)
    dialogue_plan.add_argument("check_plan", type=Path)
    tokens = commands.add_parser("critical-tokens", help="Validate extraction annotation coverage and report advisories")
    tokens.add_argument("dialogue", type=Path)
    tokens.add_argument("matrix", type=Path)
    compiler = commands.add_parser("compile-ground-truth", help="Compile validated canonical gold without inferring answers")
    for name in ("call_config", "check_plan", "dialogue", "output_path", "matrix"):
        compiler.add_argument(name, type=Path)
    compiler.add_argument("--overwrite", action="store_true")
    for command in (matrix, call_config, scaffold, check_plan, scaffold_plans, tokens, compiler):
        command.add_argument("checklist", type=Path)
        command.add_argument("normalization", type=Path)
        command.add_argument("--call-ids", nargs="+", default=PILOT_CALL_IDS,
                             help="Approved call columns in order (defaults to five pilot calls)")
    args = parser.parse_args()
    try:
        if args.command == "schemas":
            check_schemas(args.schema_dir)
            print(f"All {len(SCHEMA_NAMES)} Draft 2020-12 schema definitions are valid.")
            return 0
        if args.command == "dialogue-plan":
            dialogue, plan = load_json(args.dialogue), load_json(args.check_plan)
            if dialogue.get("call_id") != plan.get("call_id"):
                errors = validate_dialogue_against_plan(dialogue, plan)
            else:
                errors = validate(dialogue, "dialogue", args.schema_dir)
                errors.extend(validate(plan, "check_plan", args.schema_dir))
                if not errors:
                    errors = validate_dialogue_against_plan(dialogue, plan)
        elif args.command in {"critical-tokens", "compile-ground-truth"}:
            dialogue = load_json(args.dialogue)
            approved_checklist = load_json(args.checklist)
            normalization = load_json(args.normalization)
            approved_matrix = load_matrix(args.matrix, checklist=approved_checklist, call_ids=args.call_ids)
            if args.command == "critical-tokens":
                errors, advisory = validate_critical_token_coverage(
                    dialogue, approved_checklist, normalization, approved_matrix, schema_dir=args.schema_dir,
                )
                for record in advisory:
                    print(f"Advisory (non-blocking): turn {record['turn_id']}: {record['text']!r}: {record['message']}")
            else:
                result = compile_ground_truth(
                    load_json(args.call_config), load_json(args.check_plan), dialogue,
                    approved_checklist, normalization, approved_matrix, schema_dir=args.schema_dir,
                )
                output = args.output_path
                inputs = (args.call_config, args.check_plan, args.dialogue, args.matrix, args.checklist, args.normalization)
                if output.resolve() in {path.resolve() for path in inputs}:
                    raise ValueError("Canonical output must not overwrite an authoring input")
                if output.is_symlink():
                    raise ValueError(f"Refusing symlink output: {output}")
                if output.exists() and not args.overwrite:
                    raise FileExistsError(f"Output already exists: {output}; use --overwrite explicitly")
                payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as stream:
                    stream.write(payload)
                print(f"Compiled canonical ground truth: {output}. Approval and semantic review remain separate.")
                return 0
        elif args.command in {"matrix", "call-config", "scaffold-call-configs", "check-plan", "scaffold-check-plans"}:
            approved_checklist = load_json(args.checklist)
            normalization = load_json(args.normalization)
            errors = validate_checklist(approved_checklist, normalization, args.schema_dir)
            if not errors:
                approved_matrix = load_matrix(args.matrix, checklist=approved_checklist, call_ids=args.call_ids)
                if args.command == "matrix":
                    for call_id, cells in approved_matrix.items():
                        for item in approved_checklist:
                            try:
                                compile_expected(item, normalization, cells[item["id"]])
                            except ValueError as error:
                                errors.append(f"{call_id}/{item['id']}: {error}")
                elif args.command == "call-config":
                    errors = validate_call_config(
                        load_json(args.path), approved_checklist, normalization,
                        approved_matrix, args.schema_dir, complete=args.complete,
                    )
                elif args.command == "check-plan":
                    errors = validate_check_plan(
                        load_json(args.path), load_json(args.call_config), approved_checklist,
                        normalization, approved_matrix, args.schema_dir, complete=args.complete,
                    )
                elif args.command == "scaffold-check-plans":
                    # Approved configs live beside the authoritative matrix,
                    # independently of where the selected plans are written.
                    call_configs = {
                        call_id: load_json(args.matrix.parent / "call_configs" / f"{call_id}.json")
                        for call_id in approved_matrix
                    }
                    paths = write_check_plans(
                        args.output_dir, approved_checklist, normalization, approved_matrix,
                        call_configs, args.schema_dir, only=args.only, overwrite=args.overwrite,
                    )
                    print(f"Wrote {len(paths)} unfilled check-plan scaffolds; all pass complete=False.")
                    for path in paths:
                        print(path)
                    return 0
                else:
                    paths = write_call_configs(
                        args.output_dir, approved_checklist, normalization, approved_matrix,
                        args.schema_dir, overwrite=args.overwrite,
                    )
                    print(f"Wrote {len(paths)} unfilled call-config scaffolds; all pass complete=False.")
                    for path in paths:
                        print(path)
                    return 0
            if not errors:
                print("Matrix answers valid." if args.command == "matrix" else
                      f"{args.command} valid (complete={args.complete}); semantic review remains separate.")
                return 0
        elif args.command == "checklist":
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
        if (not errors and args.command in {"json", "answers"}
                and args.schema in {"check_plan", "ground_truth"}):
            errors = validate_forbidden_patterns(load_json(args.path))
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
