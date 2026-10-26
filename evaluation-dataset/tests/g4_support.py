"""Shared helpers for the G4 tests. No network, no credentials."""

import json
from pathlib import Path

from fyp_evaluation.check_plans.feasibility import synthesize_response
from fyp_evaluation.check_plans.inputs import load_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body
from release_copy import copy_tree

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"
CALL_IDS = [f"CALL_{n:03d}" for n in range(1, 51)]

_INPUTS = None


def shared_inputs():
    """The validated upstream state (G0-G2B + complete G3), loaded once per test process."""
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = load_inputs(RELEASE, SCHEMAS)
    return _INPUTS


class Clock:
    def __init__(self):
        self.tick = 0

    def __call__(self):
        self.tick += 1
        return f"2026-10-10T00:00:{self.tick % 60:02d}Z"


def bad_response(ctx):
    """A response that always fails validation (an unknown event is added and one is missing)."""
    response = synthesize_response(ctx)
    response["events"] = response["events"][1:]
    return response


def good_responder(runner, overrides=None, **offsets):
    """Provider-shaped answers built from the synthetic valid response of each requested call."""
    overrides = overrides or {}

    def respond(body):
        call_id = body["metadata"]["call_id"]
        ctx = runner.builder.context(call_id)
        response = overrides[call_id](ctx) if call_id in overrides else synthesize_response(ctx, **offsets)
        return response_body(json.dumps(response), response_id=f"resp_{call_id}")
    return respond


def make_runner(tmp_path, *, overrides=None, responder=None, transport=None, **offsets):
    """A runner over an EMPTY output root sharing the validated upstream state."""
    root = tmp_path / "release"
    root.mkdir(parents=True, exist_ok=True)
    holder = {}
    fake = transport or FakeTransport(lambda body: holder["responder"](body))
    runner = CheckPlanRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=fake, clock=Clock(), inputs=shared_inputs())
    holder["responder"] = responder or good_responder(runner, overrides, **offsets)
    return runner, fake


def entry_of(response, event_id):
    return next(e for e in response["events"] if e["event_id"] == event_id)


def snapshot(*directories):
    result = {}
    for directory in directories:
        directory = Path(directory)
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    result[str(path)] = path.read_bytes()
    return result


G4_OUTPUTS = ("authoring/check_plans", "authoring/check_plan_provenance", "authoring/check_plan_record")
# Later stages' accepted outputs describe the real G4 plans, so an isolated copy that regenerates G4 must not carry them.
LATER_STAGE_OUTPUTS = ("authoring/plan_verification", "authoring/dialogue")


def copy_release(destination, *, keep_g3=True, keep_renders=False):
    """Copy file contents only; never any G4 output (nor any G5/G6 output, which would otherwise show the isolated copy as
    already verified) and no rendered audio unless `keep_renders`."""
    skip = G4_OUTPUTS + LATER_STAGE_OUTPUTS + (() if keep_g3 else ("authoring/call_config",))
    return copy_tree(RELEASE, destination, skip_prefixes=skip, keep_renders=keep_renders)


def g3_digest():
    """A digest of every accepted G3 output, to prove G4 never touches them."""
    import hashlib

    digest = hashlib.sha256()
    for path in sorted((RELEASE / "authoring").rglob("*")):
        if path.is_file() and ("call_config" in path.as_posix()):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes().replace(bytes([13, 10]), bytes([10])))
    return digest.hexdigest()
