"""G3 command line: a thin wrapper over `ConfigRunner`.

Exit codes follow G1/G2A/G2B: 0 success, 1 invalid/differing outputs or a recorded failed outcome, 2 input/usage/
precondition/upstream-invalid/credentials/fatal-transport errors. Offline modes never need credentials.
"""

import argparse
import json
import sys
from pathlib import Path

from fyp_evaluation.llm.errors import CredentialsError, TransportError
from fyp_evaluation.llm.store import LockHeldError
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .config import DEFAULT_AUDIT_ROOT
from .run import ConfigRunner, RunError

MODES = ("dry_run", "sync", "submit", "collect", "status", "validate", "check", "attach_batch", "mark_not_submitted")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset release directory")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_ROOT, help="Gitignored operational store (default: .llm_audit)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Gate, feasibility pass and request preview; no network, writes nothing")
    mode.add_argument("--sync", action="store_true", help="One paid live request for exactly one --call-id")
    mode.add_argument("--submit", action="store_true", help="Submit all eligible calls (or the --call-id subset) as one Batch")
    mode.add_argument("--collect", action="store_true", help="Poll open batches and ingest finished results; safe to repeat")
    mode.add_argument("--status", action="store_true", help="Per-call and submission state; no network")
    mode.add_argument("--validate", action="store_true", help="Validate artifacts, provenance and record on disk; no network")
    mode.add_argument("--check", action="store_true", help="Rebuild artifacts from stored accepted responses and compare byte for byte")
    mode.add_argument("--attach-batch", nargs=2, metavar=("SUBMISSION_ID", "BATCH_ID"), help="Attach a confirmed provider batch to a SUBMISSION_UNKNOWN submission")
    mode.add_argument("--mark-not-submitted", metavar="SUBMISSION_ID", help="Declare a SUBMISSION_UNKNOWN submission never created (needs --acknowledge-duplicate-risk)")
    parser.add_argument("--call-id", action="append", dest="call_ids", metavar="CALL_ID", help="Restrict to a call (repeatable)")
    parser.add_argument("--retry-rejected", action="store_true", help="Allow new attempts for calls whose latest attempt was REJECTED")
    parser.add_argument("--reopen", action="store_true", help="Deliberately supersede an accepted config (needs --call-id and --reason)")
    parser.add_argument("--reason", help="Reason recorded in provenance when reopening")
    parser.add_argument("--acknowledge-duplicate-risk", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="With --validate: every call must have an accepted config")
    parser.add_argument("--print-request", action="store_true", help="With --dry-run: print each canonical request body")
    return parser


def _selected_mode(args):
    return next(name for name in MODES if getattr(args, name))


def _usage_errors(parser, args, mode):
    if args.reopen and (not args.reason or not args.call_ids):
        parser.error("--reopen requires --call-id and --reason")
    if args.reason and not args.reopen:
        parser.error("--reason is only used with --reopen")
    if args.reopen and mode not in ("sync", "submit", "dry_run"):
        parser.error("--reopen applies to --sync, --submit or --dry-run")
    if mode == "sync" and not (args.call_ids and len(args.call_ids) == 1):
        parser.error("--sync takes exactly one --call-id")
    if args.acknowledge_duplicate_risk and mode not in ("mark_not_submitted", "sync"):
        parser.error("--acknowledge-duplicate-risk is only used with --mark-not-submitted (or --sync, for a stage whose synchronous "
                     "outcome can be unknown)")
    if args.print_request and mode != "dry_run":
        parser.error("--print-request is only used with --dry-run")
    if args.require_complete and mode != "validate":
        parser.error("--require-complete is only used with --validate")


def _say(text=""):
    print(text)


def _run(runner, args, mode):
    reopen_ids = set(args.call_ids or []) if args.reopen else set()
    reason = args.reason if args.reopen else None
    if mode == "dry_run":
        report = runner.dry_run(args.call_ids, retry_rejected=args.retry_rejected, reopen_call_ids=reopen_ids, print_request=args.print_request)
        for row in report["calls"]:
            _say(f"{row['call_id']} {row['attempt_id']} {row['request_sha256'][:12]} ~{row['approx_input_tokens']} tokens "
                 f"{'eligible' if row['eligible'] else 'NOT eligible: ' + row['reason']}")
            if args.print_request:
                _say(row["canonical_request"])
        feasibility = report["feasibility"]
        _say(f"feasibility: {feasibility['checked'] - len(feasibility['infeasible'])}/{feasibility['checked']} calls have a valid synthetic response")
        for call_id, errors in feasibility["infeasible"].items():
            _say(f"  INFEASIBLE {call_id}: {errors}")
        _say("estimate: " + json.dumps(report["estimate"], sort_keys=True))
        return 1 if feasibility["infeasible"] else 0
    if mode == "sync":
        extra = {}
        if args.acknowledge_duplicate_risk:
            if not getattr(runner, "sync_acknowledgement", False):
                print("--acknowledge-duplicate-risk with --sync is not supported by this stage", file=sys.stderr)
                return 2
            extra["acknowledge_possible_duplicate"] = True
        attempt = runner.run_sync(args.call_ids[0], retry_rejected=args.retry_rejected, reopen_reason=reason, **extra)
        _say(f"{attempt['call_id']} {attempt['attempt_id']} {attempt['state']}"
             + (f": {attempt['rejection_reason']}" if attempt["rejection_reason"] else "")
             + (f" cost~${attempt['estimated_cost_usd']}" if attempt.get("estimated_cost_usd") is not None else ""))
        for error in attempt["validation_errors"]:
            _say(f"  - {error}")
        return 0 if attempt["state"] == "ACCEPTED" else 1
    if mode == "submit":
        report = runner.submit(args.call_ids, retry_rejected=args.retry_rejected, reopen_reason=reason)
        if not report["submission_id"]:
            _say("nothing to submit: no eligible calls")
            for call_id, why in report["skipped"].items():
                _say(f"  {call_id}: {why}")
            return 0
        _say(f"{report['submission_id']} {report['state']}: {report['submitted']} calls"
             + (f", batch {report['batch_id']}" if report.get("batch_id") else ""))
        if report.get("advice"):
            _say(report["advice"])
        return 0 if report["state"] == "SUBMITTED" else 1
    if mode == "collect":
        report = runner.collect()
        for label in ("collected", "pending", "anomalies"):
            for item in report[label]:
                _say(f"{label}: {item}")
        for call_id, attempt_id, state in report.get("finalized", []):
            _say(f"  finalized {call_id} {attempt_id} {state}")
        for call_id, attempt_id, state in report.get("outcomes", []):
            _say(f"  {call_id} {attempt_id} {state}")
        if not (report["collected"] or report["pending"]):
            _say("nothing to collect")
        return 1 if report["fatal"] or report["anomalies"] else 0
    if mode == "status":
        report = runner.status()
        for call_id, row in report["calls"].items():
            _say(f"{call_id} {row['status']} attempts={row['attempts']} rejected={row['rejected']} transport_failed={row['transport_failed']}"
                 f" dependency={row['dependency']}")
        for sub in report["submissions"]:
            _say(f"{sub['submission_id']} {sub['state']} calls={sub['calls']} batch={sub['batch_id']} status={sub['last_status']} anomalies={sub['anomalies']}")
        _say("counts: " + json.dumps(report["counts"], sort_keys=True))
        return 0
    if mode == "validate":
        errors = runner.validate(require_complete=args.require_complete)
        if errors:
            _say("\n".join(errors))
            return 1
        _say(f"{runner.spec.stage} outputs on disk are valid.")
        return 0
    if mode == "check":
        difference = runner.check()
        if any(difference.values()):
            for kind, paths in difference.items():
                for path in paths:
                    print(f"{kind}: {path}", file=sys.stderr)
            return 1
        _say(f"{runner.spec.stage} outputs reproduce byte-for-byte from the stored accepted responses.")
        return 0
    if mode == "attach_batch":
        report = runner.attach_batch(*args.attach_batch)
        _say(f"{report['submission_id']} attached to {report['batch_id']} ({report['state']})")
        return 0
    report = runner.mark_not_submitted(args.mark_not_submitted, acknowledge_duplicate_risk=args.acknowledge_duplicate_risk)
    _say(f"{report['submission_id']} {report['state']}; its calls are eligible again")
    return 0


def main(argv=None, *, transport=None, transport_factory=None, clock=None, runner_class=ConfigRunner):
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = _selected_mode(args)
    _usage_errors(parser, args, mode)
    kwargs = {"audit_root": args.audit_dir, "schema_dir": args.schema_dir, "transport": transport, "transport_factory": transport_factory}
    if clock is not None:
        kwargs["clock"] = clock
    if getattr(runner_class, "supports_selection", False) and args.call_ids:
        kwargs["selected_call_ids"] = list(args.call_ids)  # only these calls need an accepted, current, valid upstream
    try:
        return _run(runner_class(args.dataset_root, **kwargs), args, mode)
    except CredentialsError as error:
        print(f"credentials: {error}", file=sys.stderr)
        return 2
    except (RunError, LockHeldError, TransportError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
