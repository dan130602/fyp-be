"""G5 lifecycle on isolated dataset copies (fake transports only): G3/G4 version changes, version-matched historical
reports, punctuation-sensitive grounding, history corruption, STALE_UPSTREAM, and the semantic control copies."""

import json

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.check_plans.feasibility import synthesize_response as g4_synthesize
from fyp_evaluation.llm.fake_gemini_transport import FakeGeminiTransport, call_id_of
from fyp_evaluation.llm.fake_transport import response_body
from fyp_evaluation.plan_verifications.compile import compile_artifact
from fyp_evaluation.plan_verifications.controls import CONTROL_CALL, LEAK_SENTENCE, build_control
from fyp_evaluation.plan_verifications.run import PlanVerificationRunner, synthesize_response
from g5_support import RELEASE, SCHEMAS, Clock, mismatch_report, report_body
from test_lifecycle import World


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class G5World(World):
    """The G3/G4 lifecycle world plus a G5 runner on the same isolated copy (separate audit directories)."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.g5_overrides, self.g5_hooks = {}, []
        self.fake5 = FakeGeminiTransport(self._g5_respond)
        self.g5 = None

    def _g5_respond(self, body):
        for hook in self.g5_hooks:
            hook()
        self.g5_hooks = []
        call_id = call_id_of(body)
        ctx = self.g5.builder.context(call_id)
        return report_body(self.g5_overrides.get(call_id, synthesize_response)(ctx))

    def new_g5(self):
        self.g5 = PlanVerificationRunner(self.root, audit_root=self.tmp / "audit_g5", schema_dir=SCHEMAS, transport=self.fake5,
                                         clock=Clock())
        self.g5.sleep = lambda seconds: None
        return self.g5

    def report_path(self, call_id):
        return self.root / "authoring/plan_verifications" / f"{call_id}.json"

    def report_text(self, call_id):
        return self.report_path(call_id).read_bytes().decode("utf-8").replace("\r\n", "\n")

    def g5_provenance(self, call_id):
        return parse_json_text((self.root / "authoring/plan_verification_provenance" / f"{call_id}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def world(tmp_path):
    return G5World(tmp_path)


def _corrupt_and_expect(g5_world, path, mutate, expected):
    """Temporarily corrupt one committed file; a FRESH G5 runner must fail explicitly for the historical report."""
    original = path.read_bytes()
    data = parse_json_text(original.decode("utf-8"))
    mutate(data)
    path.write_bytes(serialize_json(data).encode("utf-8"))
    try:
        runner = PlanVerificationRunner(g5_world.root, audit_root=g5_world.tmp / "audit_g5", schema_dir=SCHEMAS,
                                        inputs=g5_world.g5.inputs, clock=Clock())
        errors = runner.validate()
        assert any(expected in e for e in errors), errors
    finally:
        path.write_bytes(original)


# ============================================================================== R2: material G3/G4 change
def test_material_change_keeps_the_historical_report_bound_to_its_own_g3_g4_versions(world):
    call = world.fee_call
    assert world.g4.run_sync(call)["state"] == "ACCEPTED"  # G4 v1
    g5 = world.new_g5()
    assert g5.run([call])["attempts"][0]["state"] == "ACCEPTED"  # G5 v1
    v1_text = world.report_text(call)
    g3_v1, g4_v1 = parse_json_text(v1_text)["call_config_sha256"], parse_json_text(v1_text)["check_plan_sha256"]

    world.reopen_g3(call)  # G3 v2: the dependent G4 plan and G5 report become stale
    fresh = world.new_g5()  # a new process: the stale G4 plan is tolerated per call, not a global failure
    assert fresh.inputs.g4_stale == {call: ["call_config"]}
    assert fresh.dependency_status([call])[call] == {"status": "stale", "changed": ["call_config"], "error": None}
    assert fresh.status()["calls"][call]["status"] == "ACCEPTED_STALE"
    assert fresh.g6_eligibility([call])[call]["eligible"] is False
    ok, reason = fresh.eligibility(call, reopen=True)
    assert not ok and "regenerate G4 first" in reason

    assert world.new_g4().run_sync(call, reopen_reason="G3 changed")["state"] == "ACCEPTED"  # G4 v2
    assert fresh.dependency_status([call])[call]["changed"] == ["call_config", "check_plan"]
    v2 = fresh.run([call], reopen_reason="G3 and G4 changed")["attempts"][0]
    assert v2["state"] == "ACCEPTED" and v2["attempt_id"] == "A002"
    current = parse_json_text(world.report_text(call))
    assert (current["call_config_sha256"], current["check_plan_sha256"]) != (g3_v1, g4_v1)
    (history,) = world.g5_provenance(call)["accepted_history"]
    assert history["artifact_sha256"] == text_hash(v1_text)
    assert fresh._rebuild_text(call, history) == v1_text  # rebuilt against G3 v1 + G4 v1, not the current versions
    assert fresh.validate() == [] and fresh.check() == {"missing": [], "changed": [], "extra": []}
    assert fresh.validator.errors(history["parsed_response"], fresh.builder.context(call)) != []  # v1 report vs v2 context fails
    assert fresh.g6_eligibility([call])[call]["eligible"] is True

    # missing or corrupted history fails explicitly (never falls back to the current versions)
    g4_prov = world.root / "authoring/check_plan_provenance" / f"{call}.json"
    g3_prov = world.root / "authoring/call_config_provenance" / f"{call}.json"
    g5_prov = world.root / "authoring/plan_verification_provenance" / f"{call}.json"
    _corrupt_and_expect(world, g4_prov, lambda d: d.update(accepted_history=[]), "cannot be reconstructed from G4's accepted history")
    _corrupt_and_expect(world, g4_prov, lambda d: d["accepted_history"][0]["parsed_response"]["events"][0].update(text="Altered."),
                        "corrupted history")
    _corrupt_and_expect(world, g3_prov, lambda d: d.update(accepted_history=[]), "cannot be reconstructed from G3")

    def disagree(data):
        accepted_id = data["accepted"]["attempt_id"]
        next(a for a in data["attempts"] if a["attempt_id"] == accepted_id)["upstream_sha256"]["check_plan"] = g4_v1
    _corrupt_and_expect(world, g5_prov, disagree, "disagree about the G3/G4 versions")


# ============================================================================== R2: punctuation-only G4 change
def _c01_act(ctx):
    return next(e for e in ctx.skeleton["planned_events"] if e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"]
                and e.get("role") == "ACT")


def test_punctuation_only_g4_change_changes_the_stored_exact_quote_and_history_stays_version_matched(world):
    v1_line = "Before we go further—could you  remind me what you are hoping to achieve with these investments?"
    v2_line = "Before we go further-could you remind me what you are hoping to achieve with these investments?"
    call = next(c for c in world.inputs.call_ids if any(e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"]
                                                         and e.get("role") == "ACT" for e in world.inputs.skeletons[c]["planned_events"]))

    def g4_with(line):
        def respond(body):
            ctx = world.new_g4_ctx(call)
            response = g4_synthesize(ctx)
            event_id = _c01_act(ctx)["event_id"]
            next(e for e in response["events"] if e["event_id"] == event_id)["text"] = line
            return response_body(json.dumps(response))
        return respond
    world.new_g4_ctx = lambda c: world.g4.builder.context(c)
    world.fake4.responder = g4_with(v1_line)
    assert world.g4.run_sync(call)["state"] == "ACCEPTED"

    def flag_c01(ctx):
        report = synthesize_response(ctx)
        event_id = next(e["event_id"] for e in ctx.events if e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"]
                        and e.get("role") == "ACT")
        next(c for c in report["checks"] if c["question_id"] == "C01")["evidence"] = [
            {"source": "G4", "ref": event_id, "quote": "further-could you remind me"}]
        return report
    world.g5_overrides[call] = flag_c01
    g5 = world.new_g5()
    assert g5.run([call])["attempts"][0]["state"] == "ACCEPTED"
    v1_text = world.report_text(call)
    assert "further—could you  remind me" in parse_json_text(v1_text)["checks"][0]["evidence"][0]["quote"]

    world.fake4.responder = g4_with(v2_line)
    assert world.new_g4().run_sync(call, reopen_reason="punctuation only")["state"] == "ACCEPTED"
    assert g5.run([call], reopen_reason="G4 punctuation changed")["attempts"][0]["state"] == "ACCEPTED"
    assert parse_json_text(world.report_text(call))["checks"][0]["evidence"][0]["quote"] == "further-could you remind me"
    (history,) = world.g5_provenance(call)["accepted_history"]
    assert g5._rebuild_text(call, history) == v1_text
    generation = {key: history[key] for key in g5.GENERATION_KEYS}
    against_v2 = serialize_json(compile_artifact(g5.inputs, g5.builder.context(call), history["parsed_response"], generation,
                                                 g5.builder.response_schema))
    assert text_hash(against_v2) != history["artifact_sha256"]  # only the version-matched context reproduces history
    assert g5.validate() == [] and g5.check() == {"missing": [], "changed": [], "extra": []}


# ============================================================================== stale responses
def test_response_for_a_plan_that_changed_mid_request_is_stale_and_kept(world):
    call = world.fee_call
    assert world.g4.run_sync(call)["state"] == "ACCEPTED"
    g5 = world.new_g5()
    world.g5_hooks.append(lambda: world.new_g4().run_sync(call, reopen_reason="changed while G5 was waiting"))
    attempt = g5.run([call])["attempts"][0]
    assert attempt["state"] == "STALE_UPSTREAM" and attempt["stale_upstream"] == {"changed": ["check_plan"]}
    assert (g5.audit_root / g5.inputs.dataset_id / "G5/attempts" / call / "A001/raw_response.json").is_file()
    assert not world.report_path(call).exists() and g5.call_state(call)["rejected"] == 0
    again = g5.run([call])["attempts"][0]  # an explicit new attempt, against the new plan
    assert again["state"] == "ACCEPTED" and again["attempt_id"] == "A002"
    assert parse_json_text(world.report_text(call))["check_plan_sha256"] == text_hash(world.plan_text(call))


# ============================================================================== R4: semantic controls
def test_semantic_control_copies_are_structurally_valid_and_flow_through_g5(tmp_path):
    production_plan = (RELEASE / "authoring/check_plans" / f"{CONTROL_CALL}.json").read_bytes()
    built = {name: build_control(RELEASE, tmp_path / name, name, schema_dir=SCHEMAS, audit_root=tmp_path / f"g4_{name}")
             for name in ("A_unchanged", "B_omission_leak", "C_contradiction")}
    assert (RELEASE / "authoring/check_plans" / f"{CONTROL_CALL}.json").read_bytes() == production_plan

    def g5_for(name, override=None):
        holder = {}
        fake = FakeGeminiTransport(lambda body: report_body((override or synthesize_response)(holder["runner"].builder.context(CONTROL_CALL))))
        runner = PlanVerificationRunner(built[name], audit_root=tmp_path / f"g5_{name}", schema_dir=SCHEMAS, transport=fake, clock=Clock())
        holder["runner"] = runner
        return runner

    runner_a = g5_for("A_unchanged")
    assert runner_a.run([CONTROL_CALL])["attempts"][0]["state"] == "ACCEPTED"
    assert parse_json_text((built["A_unchanged"] / "authoring/plan_verifications" / f"{CONTROL_CALL}.json").read_text(encoding="utf-8"))["verdict"] == "VALID"

    def leak(ctx):
        assert LEAK_SENTENCE.strip() in ctx.planned_text("EV_001")
        return mismatch_report(ctx, "C05", "EV_001", "you could lose some or all of the money you invest")
    runner_b = g5_for("B_omission_leak", leak)
    runner_b.run([CONTROL_CALL])
    report_b = parse_json_text((built["B_omission_leak"] / "authoring/plan_verifications" / f"{CONTROL_CALL}.json").read_text(encoding="utf-8"))
    assert report_b["verdict"] == "INVALID" and report_b["issues"][0]["code"] == "OMISSION_LEAKAGE"
    assert report_b["issues"][0]["evidence"][0]["quote"] == "you could lose some or all of the money you invest"

    def contradiction(ctx):
        assert "I do not want to" in ctx.planned_text("EV_015")
        return mismatch_report(ctx, "T05", "EV_015", "I do not want to", code="EXTRACTION_STATE")
    runner_c = g5_for("C_contradiction", contradiction)
    runner_c.run([CONTROL_CALL])
    report_c = parse_json_text((built["C_contradiction"] / "authoring/plan_verifications" / f"{CONTROL_CALL}.json").read_text(encoding="utf-8"))
    assert report_c["verdict"] == "INVALID" and report_c["checks"][18]["assessment"] == "MISMATCH"
    for runner in (runner_a, runner_b, runner_c):
        assert runner.validate() == []
