"""Shared helpers for the G3 tests. No network, no credentials."""

import json
from pathlib import Path

from fyp_evaluation.call_configs.feasibility import synthesize_response
from fyp_evaluation.call_configs.inputs import load_inputs
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"
CALL_IDS = [f"CALL_{n:03d}" for n in range(1, 51)]
GENERATED_MODEL_OUTPUTS = ("authoring/call_config", "authoring/check_plans", "authoring/check_plan_provenance", "authoring/check_plan_record")

_INPUTS = None


def shared_inputs():
    """The validated upstream state, loaded once per test process (the gate takes a few seconds)."""
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = load_inputs(RELEASE, SCHEMAS)
    return _INPUTS


class Clock:
    """Deterministic, strictly increasing timestamps."""

    def __init__(self):
        self.tick = 0

    def __call__(self):
        self.tick += 1
        return f"2026-10-09T00:00:{self.tick % 60:02d}Z"


BAD = {"employee": {"name": "   "}}  # an override that always fails validation (empty text)


def good_responder(runner, overrides=None):
    """Provider-shaped answers built from the synthetic valid response of each requested call."""
    overrides = overrides or {}

    def respond(body):
        call_id = body["metadata"]["call_id"]
        response = synthesize_response(runner.builder.context(call_id))
        response.update(overrides.get(call_id, {}))
        return response_body(json.dumps(response), response_id=f"resp_{call_id}")
    return respond


def make_runner(tmp_path, *, overrides=None, responder=None, transport=None):
    """A runner over an EMPTY output root (G3 only writes its own files) sharing the validated upstream state."""
    root = tmp_path / "release"
    root.mkdir(exist_ok=True)
    holder = {}
    fake = transport or FakeTransport(lambda body: holder["responder"](body))
    runner = ConfigRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=fake, clock=Clock(), inputs=shared_inputs())
    holder["responder"] = responder or good_responder(runner, overrides)
    return runner, fake


def copy_release(destination):
    """Copy file contents only (shutil.copytree would copy a OneDrive ReadOnly directory attribute)."""
    import shutil

    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(RELEASE.rglob("*")):
        relative = path.relative_to(RELEASE).as_posix()
        if relative.startswith(GENERATED_MODEL_OUTPUTS):  # accepted G3/G4 outputs are not part of the frozen G0-G2B inputs
            continue
        target = destination / path.relative_to(RELEASE)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination


def snapshot(*directories):
    """{relative path: bytes} of every file under the directories (for no-mutation assertions)."""
    result = {}
    for directory in directories:
        directory = Path(directory)
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    result[str(path)] = path.read_bytes()
    return result
