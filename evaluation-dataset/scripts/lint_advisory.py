"""Advisory-only lint of the unplanned employee recommendation in the accepted G6 dialogues (never a gate, never a verdict).

    python scripts/lint_advisory.py --dataset-root datasets/advisory_execution_pilot_v1 [--call-id CALL_001 ...]

Lists signals for a human to read (missing product name, rationale-like wording, order detail, placement against the derived window,
adjacency to objective/loss-tolerance evidence, a client reaction that is more than a mild acknowledgement). Meaning, rationale and
naturalness are judged by G7. Always exits 0 unless the inputs cannot be read (2).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fyp_evaluation.dialogues.advisory import advisory_lint, advisory_recommendation  # noqa: E402
from fyp_evaluation.dialogues.inputs import load_inputs  # noqa: E402
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--call-id", action="append", dest="call_ids")
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(args.dataset_root, args.schema_dir)
        for call_id in args.call_ids or inputs.call_ids:
            plan = inputs.plans[call_id]
            data = advisory_recommendation(inputs.specs[call_id], inputs.configs[call_id], plan["planned_events"])
            if data is None:
                continue
            path = args.dataset_root / "authoring" / "dialogues" / f"{call_id}.json"
            report = advisory_lint(data, json.loads(path.read_text(encoding="utf-8")))
            found = report["recommendation"]
            print(f"{call_id}: " + (f"turn {found['turn_id']}: {found['sentence']!r}" if found else "no recommendation sentence found"))
            for signal in report["signals"]:
                print(f"    lint: {signal}")
            if not report["signals"]:
                print("    lint: no signals")
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
