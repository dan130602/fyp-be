"""Command line for the quota resolver: print, check or write the resolved generation-plan quotas."""

import argparse
import json
import sys
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR, load_stage0, validate_stage0

from .quotas import QuotaError, apply_quotas, canonical_text

DESIGN = "authoring/quota_design.json"
PLAN = "authoring/generation_plan.json"


def resolved_plan(root):
    """(documents with the resolved plan, resolved plan text); raises QuotaError or ValueError."""
    root = Path(root)
    design_text = (root / DESIGN).read_bytes().decode("utf-8")
    design = json.loads(design_text)
    documents = load_stage0(root)
    plan = apply_quotas(documents["generation_plan"], design, design_text, documents["edge_case_catalog"])
    documents["generation_plan"] = plan
    return documents, canonical_text(plan)


def summary_lines(documents):
    plans = documents["generation_plan"]["language_profile_plans"]
    lines = [f"{documents['generation_plan']['num_calls']} calls: " + ", ".join(f"{p['language_profile']} {p['num_calls']}" for p in plans)]
    for p in plans:
        fails = sum(int(k) * v for k, v in p["fail_count_per_call_counts"].items())
        lines.append(f"  {p['language_profile']}: FAIL hist {list(p['fail_count_per_call_counts'].values())} = {fails} labels; "
                     f"edge hist {list(p['edge_cases_per_call_counts'].values())} = {sum(p['edge_case_counts'].values())} cases; "
                     f"composition {list(p['call_composition_counts'].values())}; durations {p['duration_counts']}; "
                     f"T03 cap {p['target_exposure_caps']['T03']}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Fail unless the stored quotas equal the recomputed ones; write nothing")
    mode.add_argument("--write", action="store_true", help="Replace the quotas in generation_plan.json")
    args = parser.parse_args(argv)
    try:
        documents, text = resolved_plan(args.dataset_root)
        if errors := validate_stage0(documents, args.schema_dir):
            print("The resolved G0 documents are invalid:\n" + "\n".join(errors), file=sys.stderr)
            return 1
        stored = (args.dataset_root / PLAN).read_bytes().decode("utf-8").replace("\r\n", "\n")
        if args.check:
            if stored != text:
                print("generation_plan.json differs from the quotas recomputed from quota_design.json", file=sys.stderr)
                return 1
            print("Stored quotas equal the recomputed quotas.")
            return 0
        if args.write:
            (args.dataset_root / PLAN).write_text(text, encoding="utf-8", newline="\n")
            print(f"Wrote {args.dataset_root / PLAN}" + ("" if stored != text else " (unchanged)"))
        print("\n".join(summary_lines(documents)))
        return 0
    except (OSError, ValueError, QuotaError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
