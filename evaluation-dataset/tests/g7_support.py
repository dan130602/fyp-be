"""Shared helpers for the G7 tests. No network, no credentials; production files are never written."""

import json
import shutil
from pathlib import Path

from fyp_evaluation.dialogue_verifications.inputs import load_inputs
from fyp_evaluation.dialogue_verifications.run import DialogueVerificationRunner, synthesize_response
from fyp_evaluation.llm.fake_gemini_transport import FakeGeminiTransport, call_id_of, interaction_body

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"
CALL_IDS = [f"CALL_{n:03d}" for n in range(1, 51)]
G7_OUTPUTS = ("authoring/dialogue_verification",)

_INPUTS = None


def shared_inputs():
    """The validated upstream state (G0-G6 of the committed release), loaded once per test process (read-only)."""
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = load_inputs(RELEASE, SCHEMAS)
    return _INPUTS


class Clock:
    def __init__(self):
        self.tick = 0

    def __call__(self):
        self.tick += 1
        return f"2026-10-15T00:{self.tick // 60 % 60:02d}:{self.tick % 60:02d}Z"


def report_body(report, **kwargs):
    return interaction_body(json.dumps(report), **kwargs)


def good_responder(runner, overrides=None, **body_kwargs):
    """Interactions bodies carrying the synthetic all-MATCH report of each requested call (or an override)."""
    overrides = overrides or {}

    def respond(body):
        call_id = call_id_of(body)
        ctx = runner.builder.context(call_id)
        report = overrides[call_id](ctx) if call_id in overrides else synthesize_response(ctx)
        return report_body(report, interaction_id=f"int_{call_id}", **body_kwargs)
    return respond


def make_runner(tmp_path, *, overrides=None, responder=None, transport=None, errors=None, inputs=None, **body_kwargs):
    """A runner whose outputs go to an EMPTY tmp root while sharing the validated (read-only) upstream state of the real release."""
    root = tmp_path / "release"
    root.mkdir(parents=True, exist_ok=True)
    holder = {}
    fake = transport or FakeGeminiTransport(lambda body: holder["responder"](body), errors=errors)
    runner = DialogueVerificationRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=fake, clock=Clock(),
                                        inputs=inputs or shared_inputs())
    runner.sleep = lambda seconds: holder.setdefault("slept", []).append(seconds)
    runner.slept = holder
    holder["responder"] = responder or good_responder(runner, overrides, **body_kwargs)
    return runner, fake


def copy_release(destination):
    """Copy file contents only (keeps the accepted G0-G6 outputs, never any G7 output)."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(RELEASE.rglob("*")):
        relative = path.relative_to(RELEASE).as_posix()
        if relative.startswith(G7_OUTPUTS) or relative.endswith(".stage_write.lock"):
            continue
        target = destination / path.relative_to(RELEASE)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination


def make_world(tmp_path, *, overrides=None, errors=None, **body_kwargs):
    """A runner over an isolated FULL copy of the release (G0-G6 included), so upstream files can be changed or removed."""
    root = copy_release(tmp_path / "dataset")
    holder = {}
    fake = FakeGeminiTransport(lambda body: holder["responder"](body), errors=errors)
    runner = DialogueVerificationRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=fake, clock=Clock(),
                                        inputs=load_inputs(root, SCHEMAS))
    runner.sleep = lambda seconds: holder.setdefault("slept", []).append(seconds)
    runner.slept = holder
    holder["responder"] = good_responder(runner, overrides, **body_kwargs)
    return runner, fake


def fresh_runner(runner, **kwargs):
    """A new runner over the same root/audit (a later process), reloading the inputs from disk."""
    other = DialogueVerificationRunner(runner.root, audit_root=runner.audit_root, schema_dir=SCHEMAS, clock=Clock(),
                                       inputs=load_inputs(runner.root, SCHEMAS), **kwargs)
    other.sleep = lambda seconds: None
    return other


def runner_factory(inputs=None, sleeper=None):
    """A `runner_class` for the CLI that injects the shared inputs (the CLI path is otherwise unchanged)."""
    def build(root, **kwargs):
        runner = DialogueVerificationRunner(root, inputs=inputs or shared_inputs(), **kwargs)
        runner.sleep = sleeper or (lambda seconds: None)
        return runner
    return build


def quote_of(ctx, event_id, limit=60):
    """A (turn_id, quote) citation of an event's rendered speech."""
    return {"turn_id": ctx.turn_of_event[event_id], "quote": ctx.planned_text(event_id)[:limit]}


def mismatch_report(ctx, question_id, citation, *, code="EXTRACTION_STATE", scope="G6_ONLY", assessment="MISMATCH", evidence_on_issue=True):
    """The synthetic report with one question flagged and linked to one issue (optionally citing `citation`)."""
    report = synthesize_response(ctx)
    check = next(c for c in report["checks"] if c["question_id"] == question_id)
    cited = [citation] if citation else []
    check.update(assessment=assessment, issue_ids=["I1"], rationale="The dialogue contradicts the target.", evidence=cited)
    report["issues"] = [{"issue_id": "I1", "assessment": assessment, "code": code, "question_ids": [question_id], "assignment_ids": [],
                         "evidence": cited if evidence_on_issue else [], "explanation": "The cited speech conveys the opposite of the target.",
                         "repair_scope": scope}]
    return report


def snapshot(*directories):
    result = {}
    for directory in directories:
        directory = Path(directory)
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    result[str(path)] = path.read_bytes()
    return result
