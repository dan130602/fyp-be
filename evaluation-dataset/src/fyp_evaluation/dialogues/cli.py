"""G6 command line: the shared stage CLI bound to `DialogueRunner`.

Identical modes, modifiers, guards and exit codes to G3/G4. G6 supports Batch (bulk generation: `--submit`, `--collect`) and, for pilots,
one synchronous request per call (`--sync --call-id X`); both use the same request builder, validation and acceptance. A synchronous
request whose outcome is unknown (timeout) is recorded as possibly billed and blocks the call until `--sync --acknowledge-duplicate-risk`.
Offline modes never need credentials.

One G6-only option: `--repair-feedback FILE` (with `--dry-run` or `--submit`) adds call-specific repair guidance, derived from the
call's current INVALID G7 report, to that call's request (see `request.load_repair_feedback`) and records its source on the attempt.
"""

import argparse
import json
import sys
from pathlib import Path

from fyp_evaluation.call_configs.cli import main as stage_main

from .request import load_repair_feedback
from .run import DialogueRunner


def main(argv=None, **kwargs):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--repair-feedback", type=Path)
    pre.add_argument("--dataset-root", type=Path)
    pre.add_argument("--revalidate-length-only", action="store_true")
    pre.add_argument("--reinstate-accepted", action="store_true")
    pre.add_argument("--attempt-id")
    known, rest = pre.parse_known_args(argv)
    if known.revalidate_length_only and known.reinstate_accepted:
        print("--revalidate-length-only and --reinstate-accepted are separate actions", file=sys.stderr)
        return 2
    if known.revalidate_length_only or known.reinstate_accepted:
        flag, label, description, method = (
            ("--revalidate-length-only", "length-only revalidation", "Offline reuse of one completed, length-only rejected G6 response; "
             "no network", "revalidate_length_only") if known.revalidate_length_only else
            ("--reinstate-accepted", "reinstatement", "Offline reinstatement of an earlier accepted G6 dialogue that a reopen superseded; "
             "no network", "reinstate_accepted"))
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--dataset-root", type=Path, required=True)
        parser.add_argument("--call-id", required=True)
        parser.add_argument("--attempt-id", required=True)
        parser.add_argument("--reason", required=True)
        from fyp_evaluation.llm.stage_runner import DEFAULT_AUDIT_ROOT
        from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

        parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_ROOT)
        parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
        args = parser.parse_args([arg for arg in argv if arg != flag])
        try:
            runner_class = kwargs.get("runner_class", DialogueRunner)
            runner = runner_class(args.dataset_root, audit_root=args.audit_dir, schema_dir=args.schema_dir,
                                  selected_call_ids=[args.call_id])
            result = getattr(runner, method)(args.call_id, args.attempt_id, args.reason)
        except (OSError, ValueError, KeyError) as error:
            print(f"{label}: {error}", file=sys.stderr)
            return 1
        print(json.dumps({"call_id": args.call_id, "attempt_id": result["attempt_id"], "state": result["state"],
                          "source_attempt_id": args.attempt_id, "estimated_cost_usd": result["estimated_cost_usd"]}))
        return 0 if result["state"] == "ACCEPTED" else 1
    if known.attempt_id:
        print("--attempt-id applies only to --revalidate-length-only or --reinstate-accepted", file=sys.stderr)
        return 2
    if known.repair_feedback is not None:
        if known.dataset_root is None:
            print("--repair-feedback needs --dataset-root", file=sys.stderr)
            return 2
        if not any(mode in rest for mode in ("--submit", "--sync", "--dry-run")):
            print("--repair-feedback applies to --submit, --sync or --dry-run", file=sys.stderr)
            return 2
        try:
            feedback = load_repair_feedback(known.repair_feedback, known.dataset_root)
        except (OSError, ValueError, KeyError) as error:
            print(f"repair feedback: {error}", file=sys.stderr)
            return 2
        argv = ["--dataset-root", str(known.dataset_root), *rest]
        kwargs.setdefault("runner_class", lambda root, **kw: DialogueRunner(root, repair_feedback=feedback, **kw))
    kwargs.setdefault("runner_class", DialogueRunner)
    return stage_main(argv, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
