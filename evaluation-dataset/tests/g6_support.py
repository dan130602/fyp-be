"""Shared helpers for the G6 tests. No network, no credentials; production files are never written."""

import json
import shutil
from pathlib import Path

from fyp_evaluation.dialogues.feasibility import synthesize_dialogue
from fyp_evaluation.dialogues.inputs import load_inputs
from fyp_evaluation.dialogues.run import DialogueRunner
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"
CALL_IDS = [f"CALL_{n:03d}" for n in range(1, 51)]
G6_OUTPUTS = ("authoring/dialogues", "authoring/dialogue_provenance", "authoring/dialogue_record")

_INPUTS = None


def shared_inputs():
    """The validated upstream state (G0-G5 gate of the committed release), loaded once per test process (read-only)."""
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = load_inputs(RELEASE, SCHEMAS)
    return _INPUTS


class Clock:
    def __init__(self):
        self.tick = 0

    def __call__(self):
        self.tick += 1
        return f"2026-10-14T00:{self.tick // 60 % 60:02d}:{self.tick % 60:02d}Z"


def copy_release(destination):
    """Copy file contents only (keeps the accepted G0-G5 outputs, never any G6 output)."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(RELEASE.rglob("*")):
        relative = path.relative_to(RELEASE).as_posix()
        if relative.startswith(G6_OUTPUTS) or relative.endswith(".stage_write.lock"):
            continue
        target = destination / path.relative_to(RELEASE)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination


def good_responder(runner, overrides=None):
    """Provider-shaped answers built from the synthetic valid dialogue of each requested call (or an override)."""
    overrides = overrides or {}

    def respond(body):
        call_id = body["metadata"]["call_id"]
        ctx = runner.builder.context(call_id)
        response = overrides[call_id](ctx) if call_id in overrides else synthesize_dialogue(ctx)
        return response_body(json.dumps(response), response_id=f"resp_{call_id}")
    return respond


def make_runner(tmp_path, *, overrides=None):
    """A runner over an EMPTY output root sharing the validated (read-only) upstream state of the real release."""
    root = tmp_path / "release"
    root.mkdir(parents=True, exist_ok=True)
    holder = {}
    fake = FakeTransport(lambda body: holder["responder"](body))
    runner = DialogueRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=fake, clock=Clock(), inputs=shared_inputs())
    holder["responder"] = good_responder(runner, overrides)
    return runner, fake


def make_world(tmp_path, *, overrides=None):
    """A runner over an isolated FULL copy of the release (G0-G5 included), so upstream files can be changed or removed."""
    root = copy_release(tmp_path / "dataset")
    holder = {}
    fake = FakeTransport(lambda body: holder["responder"](body))
    runner = DialogueRunner(root, audit_root=tmp_path / "audit", schema_dir=SCHEMAS, transport=fake, clock=Clock(),
                            inputs=load_inputs(root, SCHEMAS))
    holder["responder"] = good_responder(runner, overrides)
    return runner, fake


def snapshot(*directories):
    result = {}
    for directory in directories:
        directory = Path(directory)
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    result[str(path)] = path.read_bytes()
    return result


def attempt_state(runner, call_id, attempt="A001"):
    return runner.store.read_attempt(call_id, attempt)["state"]


def submit_and_finish(runner, fake, call_ids, **finish):
    report = runner.submit(call_ids)
    assert report["state"] == "SUBMITTED", report
    fake.finish_batch(report["batch_id"], **finish)
    return report
