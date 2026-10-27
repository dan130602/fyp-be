"""Offline fixtures for the multilingual G3-G5 tests (Phase 3A). No network, no credentials, no paid calls.

Builds on `g4_multilingual_support` (artificial G3 configs over the validated G0-G2B state) and adds artificial G4 plans, G5
verification inputs and small helpers. Nothing is written to the repository; scratch copies of the release never include audio.
"""

import json
from pathlib import Path

import g3_support
import g4_support
import g5_support
import g6_support
import g7_support
from fyp_evaluation import register_guide
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.check_plans.inputs import load_inputs as load_g4_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.dialogue_verifications.inputs import load_inputs as load_g7_inputs
from fyp_evaluation.dialogue_verifications.run import DialogueVerificationRunner
from fyp_evaluation.dialogues.inputs import load_inputs as load_g6_inputs
from fyp_evaluation.dialogues.run import DialogueRunner
from fyp_evaluation.llm.fake_gemini_transport import FakeGeminiTransport
from fyp_evaluation.llm.fake_transport import FakeTransport
from fyp_evaluation.plan_verifications.inputs import load_inputs as load_g5_inputs
from fyp_evaluation.plan_verifications.run import PlanVerificationRunner

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.check_plans.compile import compile_artifact as compile_plan
from fyp_evaluation.check_plans.feasibility import synthesize_response as synthesize_plan
from fyp_evaluation.check_plans.request import RequestBuilder as PlanRequestBuilder
from fyp_evaluation.plan_verifications.inputs import VerificationInputs
from g4_multilingual_support import PROFILES, RELEASE, SCHEMAS, calls_of, context, plan_inputs, upstream_inputs  # noqa: F401  (re-exported)
from release_copy import copy_tree

_GENERATION = {"attempt_id": "A001", "model": "fixture", "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}
_CACHE = {}


def plan_texts(call_ids):
    """{call_id: text of an artificial G4 plan} for the given calls (deterministic; never accepted by G4 or written to disk)."""
    inputs = plan_inputs()
    builder = PlanRequestBuilder(inputs, SCHEMAS)
    texts = {}
    for call_id in call_ids:
        key = ("plan_text", call_id)
        if key not in _CACHE:
            ctx = builder.context(call_id)
            generation = {**_GENERATION, "response_schema_sha256": builder.response_schema_sha256, **register_guide.extra_hashes(inputs.specs[call_id])}
            _CACHE[key] = serialize_json(compile_plan(inputs, ctx, synthesize_plan(ctx), generation, builder.response_schema))
        texts[call_id] = _CACHE[key]
    return texts


def verification_inputs(call_ids):
    """`VerificationInputs` (G5's inputs) over the artificial G3 configs and G4 plans of `call_ids`."""
    return VerificationInputs(plan_inputs(), plan_texts(call_ids), {}, "0" * 64)


# Generated G3-G9 outputs, run records, exposure ledger and allowlist: a scratch release starts from the frozen G0-G2B state only, even
# after a real (live) run has written outputs into the committed release.
GENERATED = tuple(f"authoring/{name}" for name in (
    "call_config", "check_plan_provenance", "check_plan_record", "check_plans", "plan_verification", "dialogue", "ground_truth",
    "live_allowlist", "exposure_log"))


def scratch_release(destination):
    """A scratch copy of the multilingual release: the frozen G0-G2B state only (no generated outputs, no audio)."""
    return copy_tree(RELEASE, destination, skip_prefixes=GENERATED)


def first_call(profile, index=0):
    return calls_of(profile)[index]


def write_allowlist(root, batches):
    path = Path(root) / "authoring" / "live_allowlist.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "1.0", "batches": batches}, indent=2) + "\n", encoding="utf-8")
    return path


class Chain:
    """G3 -> G4 -> G5 over a scratch copy of the multilingual release, all on fake transports (offline, no credentials)."""

    def __init__(self, root, audit_root, call_ids):
        self.root, self.audit_root, self.call_ids = Path(root), Path(audit_root), list(call_ids)
        self.g3 = self.g4 = self.g5 = self.g6 = self.g7 = None

    def run_g3(self):
        holder = {}
        runner = ConfigRunner(self.root, audit_root=self.audit_root, schema_dir=SCHEMAS, clock=g3_support.Clock(),
                              transport=FakeTransport(lambda body: holder["respond"](body)), inputs=upstream_inputs(self.root))
        holder["respond"] = g3_support.good_responder(runner)
        self.g3 = runner
        return {call_id: runner.run_sync(call_id)["state"] for call_id in self.call_ids}

    def run_g4(self, overrides=None):
        holder = {}
        runner = CheckPlanRunner(self.root, audit_root=self.audit_root, schema_dir=SCHEMAS, clock=g4_support.Clock(),
                                 transport=FakeTransport(lambda body: holder["respond"](body)), selected_call_ids=self.call_ids,
                                 inputs=load_g4_inputs(self.root, SCHEMAS, self.call_ids))
        holder["respond"] = g4_support.good_responder(runner, overrides)
        self.g4 = runner
        return {call_id: runner.run_sync(call_id)["state"] for call_id in self.call_ids}

    def open_g5(self, overrides=None, errors=None):
        holder = {}
        runner = PlanVerificationRunner(self.root, audit_root=self.audit_root, schema_dir=SCHEMAS, clock=g5_support.Clock(),
                                        transport=FakeGeminiTransport(lambda body: holder["respond"](body), errors=errors),
                                        selected_call_ids=self.call_ids, inputs=load_g5_inputs(self.root, SCHEMAS, self.call_ids))
        runner.sleep = lambda seconds: None
        holder["respond"] = g5_support.good_responder(runner, overrides)
        self.g5 = runner
        return runner

    def run_g5(self, overrides=None):
        runner = self.open_g5(overrides)
        return {call_id: runner.run_sync(call_id)["state"] for call_id in self.call_ids}


    def _selection(self):
        """The calls to run G6/G7 for when the chain covers only some of the release (selected-call mode), else None (complete mode)."""
        return self.call_ids if len(self.call_ids) < len(upstream_inputs(self.root).call_ids) else None

    def open_g6(self, overrides=None):
        """A G6 runner over this root's current G0-G5 state (no request is sent)."""
        holder = {}
        selected = self._selection()
        runner = DialogueRunner(self.root, audit_root=self.audit_root, schema_dir=SCHEMAS, clock=g6_support.Clock(), selected_call_ids=selected,
                                transport=FakeTransport(lambda body: holder["respond"](body)), inputs=load_g6_inputs(self.root, SCHEMAS, selected))
        holder["respond"] = g6_support.good_responder(runner, overrides)
        self.g6 = runner
        return runner

    def run_g6(self, overrides=None, call_ids=None):
        """G6 is Batch-only: submit, finish the fake batch, collect. Needs a complete G3-G5 (G6 has no selected-call mode)."""
        runner = self.open_g6(overrides)
        chosen = list(call_ids or self.call_ids)
        report = runner.submit(chosen)
        assert report["state"] == "SUBMITTED", report
        runner.transport().finish_batch(report["batch_id"])
        runner.collect()
        return {call_id: runner.store.read_attempt(call_id, "A001")["state"] for call_id in chosen}

    def open_g7(self, overrides=None, errors=None):
        holder = {}
        selected = self._selection()
        runner = DialogueVerificationRunner(self.root, audit_root=self.audit_root, schema_dir=SCHEMAS, clock=g7_support.Clock(),
                                            transport=FakeGeminiTransport(lambda body: holder["respond"](body), errors=errors),
                                            selected_call_ids=selected, inputs=load_g7_inputs(self.root, SCHEMAS, selected))
        runner.sleep = lambda seconds: None
        holder["respond"] = g7_support.good_responder(runner, overrides)
        self.g7 = runner
        return runner

    def run_g7(self, overrides=None, call_ids=None):
        runner = self.open_g7(overrides)
        return {call_id: runner.run_sync(call_id)["state"] for call_id in (call_ids or self.call_ids)}


def copy_chain(source_root, destination, audit_root=None, call_ids=None):
    """A writable copy of an accepted chain root (authoring files only; the release has no audio) and a Chain over it
    (all 250 calls unless `call_ids` names the selected ones)."""
    root = copy_tree(source_root, destination)
    return Chain(root, Path(destination).parent / "audit_copy", list(call_ids or upstream_inputs().call_ids))


def chain(tmp_path, call_ids):
    return Chain(scratch_release(tmp_path / "dataset"), tmp_path / "audit", call_ids)


__all__ = ["Chain", "chain", "PROFILES", "RELEASE", "SCHEMAS", "calls_of", "context", "first_call", "parse_json_text", "plan_inputs", "plan_texts",
           "scratch_release", "text_hash", "upstream_inputs", "verification_inputs", "write_allowlist"]
