"""G7 command line: a thin wrapper over `DialogueVerificationRunner`.

Exit codes follow G1-G6: 0 success, 1 invalid/differing outputs or a recorded failed outcome (or a stopped run), 2
input/usage/precondition/credentials/fatal errors. Offline modes never need credentials and never write.
A completed report whose verdict is INVALID or UNCLEAR is a successful G7 outcome (exit 0); it only blocks G8.
There is no Batch path: G7 sends one synchronous Flex request per call.
"""

import argparse
import json
import sys
from pathlib import Path

from fyp_evaluation.llm.errors import CredentialsError, TransportError
from fyp_evaluation.llm.stage_runner import DEFAULT_AUDIT_ROOT
from fyp_evaluation.llm.store import LockHeldError
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .run import DialogueVerificationRunner, RunError, describe

MODES = ("dry_run", "run", "status", "validate", "check", "prepare_review", "ingest_review")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset release directory")
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_ROOT, help="Gitignored operational store (default: .llm_audit)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="G7A preflight, feasibility pass and request preview; no network, writes nothing")
    mode.add_argument("--run", action="store_true", help="Recover interrupted state, then verify eligible dialogues with paid Flex requests")
    mode.add_argument("--status", action="store_true", help="Per-call lifecycle, verdict, quality review and G8 eligibility; no network")
    mode.add_argument("--validate", action="store_true", help="Validate reports, provenance and record on disk; no network")
    mode.add_argument("--prepare-review", action="store_true",
                      help="Claude-assisted review: write the brief and its hash for one --call-id (multilingual releases); no network")
    mode.add_argument("--ingest-review", type=Path, metavar="FILE",
                      help="Claude-assisted review: validate and accept a review file for one --call-id; no network, no credentials")
    mode.add_argument("--check", action="store_true", help="Rebuild reports from stored accepted responses and compare byte for byte")
    parser.add_argument("--call-id", action="append", dest="call_ids", metavar="CALL_ID", help="Restrict to a call (repeatable)")
    parser.add_argument("--output", type=Path, help="With --prepare-review: write the review packet to this file (default: stdout)")
    parser.add_argument("--limit", type=int, help="With --run: verify at most N calls")
    parser.add_argument("--retry-rejected", action="store_true", help="Allow new attempts for calls whose latest report was REJECTED")
    parser.add_argument("--reopen", action="store_true", help="Deliberately re-verify an accepted report (needs --call-id and --reason)")
    parser.add_argument("--reason", help="Reason recorded in provenance when reopening")
    parser.add_argument("--acknowledge-possible-duplicate", action="store_true",
                        help="With --run and one --call-id: accept that an ambiguous earlier attempt may have been billed, and try again")
    parser.add_argument("--require-complete", action="store_true", help="With --validate: every call must have an accepted report")
    parser.add_argument("--print-request", action="store_true", help="With --dry-run: print each canonical request body")
    return parser


def _usage_errors(parser, args, mode):
    if args.reopen and (not args.reason or not args.call_ids):
        parser.error("--reopen requires --call-id and --reason")
    if args.reason and not args.reopen:
        parser.error("--reason is only used with --reopen")
    if (args.reopen or args.retry_rejected) and mode not in ("run", "dry_run", "prepare_review", "ingest_review"):
        parser.error("--reopen and --retry-rejected apply to --run, --dry-run, --prepare-review or --ingest-review")
    if mode in ("prepare_review", "ingest_review") and not (args.call_ids and len(args.call_ids) == 1):
        parser.error("--prepare-review and --ingest-review take exactly one --call-id")
    if args.output and mode != "prepare_review":
        parser.error("--output is only used with --prepare-review")
    if args.acknowledge_possible_duplicate and (mode != "run" or not args.call_ids or len(args.call_ids) != 1):
        parser.error("--acknowledge-possible-duplicate is only used with --run and exactly one --call-id")
    if args.limit is not None and (mode != "run" or args.limit < 1):
        parser.error("--limit is only used with --run and must be at least 1")
    if args.print_request and mode != "dry_run":
        parser.error("--print-request is only used with --dry-run")
    if args.require_complete and mode != "validate":
        parser.error("--require-complete is only used with --validate")


def _say(text=""):
    print(text)


def _run(runner, args, mode):
    reopen_ids = set(args.call_ids or []) if args.reopen else set()
    if mode == "dry_run":
        report = runner.dry_run(args.call_ids, retry_rejected=args.retry_rejected, reopen_call_ids=reopen_ids, print_request=args.print_request)
        for row in report["calls"]:
            if row["request_sha256"] is None:
                _say(f"{row['call_id']} no request NOT eligible: {row['reason']}")
                continue
            _say(f"{row['call_id']} {row['request_sha256'][:12]} {row['characters']} chars {row['utf8_bytes']} UTF-8 bytes "
                 f"~{row['approx_input_tokens']} tokens (approximate) {'eligible' if row['eligible'] else 'NOT eligible: ' + row['reason']}")
            if args.print_request:
                _say(row["canonical_request"])
        feasibility = report["feasibility"]
        _say(f"feasibility: {feasibility['checked'] - len(feasibility['infeasible'])}/{feasibility['checked']} calls accept a synthetic report")
        for call_id, errors in feasibility["infeasible"].items():
            _say(f"  INFEASIBLE {call_id}: {errors}")
        _say("estimate: " + json.dumps(report["estimate"], sort_keys=True))
        return 1 if feasibility["infeasible"] else 0
    if mode == "run":
        report = runner.run(args.call_ids, limit=args.limit, retry_rejected=args.retry_rejected,
                            reopen_reason=args.reason if args.reopen else None,
                            acknowledge_duplicate=args.acknowledge_possible_duplicate)
        for call_id, attempt_id, state in report["recovered"]:
            _say(f"recovered {call_id} {attempt_id} {state}")
        if report["acknowledged"]:
            _say(f"acknowledged possible duplicate billing for {args.call_ids[0]} {report['acknowledged']}")
        for attempt in report["attempts"]:
            _say(describe(attempt))
            for error in attempt["validation_errors"][:10]:
                _say(f"  - {error}")
        if not report["attempts"]:
            _say("nothing to verify: no eligible calls")
            for call_id, reason in report["skipped"].items():
                _say(f"  skipped {call_id}: {reason}")
        if report["stopped"]:
            _say(f"STOPPED: {report['stopped']}")
        failed = any(a["state"] != "ACCEPTED" for a in report["attempts"])
        return 1 if failed or report["stopped"] else 0
    if mode == "prepare_review":
        packet = runner.prepare_review(args.call_ids[0], retry_rejected=args.retry_rejected, reopen=args.reopen)
        text = json.dumps(packet, ensure_ascii=False, indent=1)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
            _say(f"{args.call_ids[0]} review packet written to {args.output} (request_sha256 {packet['request_sha256']})")
        else:
            _say(text)
        return 0
    if mode == "ingest_review":
        attempt = runner.ingest_review(args.call_ids[0], args.ingest_review, reopen_reason=args.reason if args.reopen else None,
                                    retry_rejected=args.retry_rejected)
        _say(describe(attempt))
        for error in attempt["validation_errors"][:10]:
            _say(f"  - {error}")
        return 0 if attempt["state"] == "ACCEPTED" else 1
    if mode == "status":
        report = runner.status()
        for call_id, row in report["calls"].items():
            flag = " QUALITY_REVIEW_REQUIRED" if row["quality_review"] == "REVIEW_REQUIRED" else ""
            _say(f"{call_id} {row['status']} verdict={row['verdict']} quality_review={row['quality_review']} "
                 f"g8_eligible={row['g8_eligible']} attempts={row['attempts']} rejected={row['rejected']} "
                 f"transport_failed={row['transport_failed']} dependency={row['dependency']} sends={row['provider_sends']} "
                 f"est_cost_usd={row['estimated_cost_usd']}{flag}")
        _say("counts: " + json.dumps(report["counts"], sort_keys=True))
        _say("usage (estimated from reported usage; not a provider balance): " + json.dumps(report["usage"], sort_keys=True))
        _say("verdicts: " + json.dumps(report["verdicts"], sort_keys=True) + f" tier_not_echoed={report['tier_not_echoed']}")
        _say("quality review (advisory; REVIEW_REQUIRED means a serious naturalness finding to read): "
             + json.dumps(report["quality_reviews"], sort_keys=True))
        _say(f"g8_eligible (VALID verdict, current and intact): {report['g8_eligible']}/{len(report['calls'])}")
        return 0
    if mode == "validate":
        errors = runner.validate(require_complete=args.require_complete)
        if errors:
            _say("\n".join(errors))
            return 1
        _say("G7 outputs on disk are valid.")
        return 0
    difference = runner.check()
    if any(difference.values()):
        for kind, paths in difference.items():
            for path in paths:
                print(f"{kind}: {path}", file=sys.stderr)
        return 1
    _say("G7 outputs reproduce byte-for-byte from the stored accepted responses.")
    return 0


def main(argv=None, *, transport=None, transport_factory=None, clock=None, runner_class=DialogueVerificationRunner):
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = next(name for name in MODES if getattr(args, name))
    _usage_errors(parser, args, mode)
    kwargs = {"audit_root": args.audit_dir, "schema_dir": args.schema_dir, "transport": transport, "transport_factory": transport_factory}
    if clock is not None:
        kwargs["clock"] = clock
    if getattr(runner_class, "supports_selection", False) and getattr(args, "call_ids", None):
        kwargs["selected_call_ids"] = list(args.call_ids)  # only these calls need their upstream accepted, current and valid
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
