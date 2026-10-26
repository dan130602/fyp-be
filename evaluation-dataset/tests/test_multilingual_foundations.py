"""Phase 3A-0/3A-1 foundations: versioned schemas, language-profile provenance, G3 v2 brief/validator, the paid-run guard,
exposure persistence and the register-guide dependency. Offline only: fake or fake-like transports, no credentials.
"""

import json
import shutil
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

import g3_support
import multilingual_support as M
from fyp_evaluation import register_guide
from fyp_evaluation.call_configs.compile import build_record as build_g3_record
from fyp_evaluation.call_configs.feasibility import make_validator, synthesize_response
from fyp_evaluation.call_configs.request import RequestBuilder as G3RequestBuilder, build_brief
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.llm import exposure, live_guard
from fyp_evaluation.llm.errors import AmbiguousError
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body
from fyp_evaluation.llm.stage_runner import RunError

PROFILE_CALLS = {profile: M.first_call(profile) for profile in M.PROFILES}


def validator_for(name):
    return Draft202012Validator(json.loads((M.SCHEMAS / "stage3" / f"{name}.schema.json").read_text(encoding="utf-8")))


# ============================================================================================ versioned schemas and provenance
def test_v2_schemas_accept_multilingual_artifacts_and_the_historical_schema_rejects_them():
    upstream = M.upstream_inputs()
    text = M.plan_inputs(call_ids=["CALL_001"]).config_texts["CALL_001"]
    artifact = json.loads(text)
    assert artifact["schema_version"] == "2.0" and artifact["source"]["language_profiles_sha256"]
    assert not list(validator_for("call_config.v2").iter_errors(artifact))
    assert list(validator_for("call_config").iter_errors(artifact))
    record = build_g3_record(upstream, {"CALL_001": text}, {"CALL_001": {"accepted_history": []}})
    assert record["schema_version"] == "2.0" and record["upstream"]["language_profiles_sha256"]
    assert not list(validator_for("call_config_record.v2").iter_errors(record))
    assert list(validator_for("call_config_record").iter_errors(record))
    del record["upstream"]["language_profiles_sha256"]
    assert list(validator_for("call_config_record.v2").iter_errors(record))


def test_every_stage_records_the_language_profile_hash_but_the_pilot_does_not():
    expected = M.text_hash((M.RELEASE / "language_profiles.json").read_text(encoding="utf-8"))
    assert M.upstream_inputs().source_hashes()["language_profiles_sha256"] == expected
    assert M.plan_inputs().source_hashes()["language_profiles_sha256"] == expected
    assert M.verification_inputs(["CALL_001"]).source_hashes()["language_profiles_sha256"] == expected
    assert "language_profiles_sha256" not in g3_support.shared_inputs().source_hashes()


def test_the_response_schema_v2_adds_spoken_names_and_the_historical_one_is_untouched():
    v1 = json.loads((M.SCHEMAS / "stage3/call_config_response.schema.json").read_text(encoding="utf-8"))
    v2 = json.loads((M.SCHEMAS / "stage3/call_config_response.v2.schema.json").read_text(encoding="utf-8"))
    assert "spoken_names" not in v1["properties"]["client"]["properties"]
    for speaker in ("client", "employee"):
        assert "spoken_names" in v2["properties"][speaker]["properties"] and "spoken_names" in v2["properties"][speaker]["required"]


# ============================================================================================ G3 v2 brief and validation
@pytest.fixture(scope="module")
def g3_builder():
    return G3RequestBuilder(M.upstream_inputs(), M.SCHEMAS)


@pytest.mark.parametrize("profile", M.PROFILES)
def test_the_g3_brief_replaces_the_single_language_value_with_the_language_block(g3_builder, profile):
    upstream, call_id = M.upstream_inputs(), PROFILE_CALLS[profile]
    brief = build_brief(upstream, g3_builder.context(call_id))
    assert "language" not in brief["metadata"]
    assert brief["language"]["language_profile"] == profile
    assert set(brief["language"]["speakers"]) == {"CLIENT", "EMPLOYEE"} and set(brief["required_name_languages"]) == {"client", "employee"}
    assert all("language_variant" in e for e in brief["edge_assignments"])
    expected = {"ENGLISH": ["en"], "SINGLISH": ["en"], "MANDARIN": ["zh"], "EN_ZH_MIXED": ["en", "zh"]}[profile]
    assert brief["required_name_languages"]["client"] == expected


def test_the_g3_request_uses_the_v2_prompt_schema_and_format_and_the_pilot_keeps_v1(g3_builder):
    body = g3_builder.build("CALL_001", "A001")
    assert body["text"]["format"]["name"] == "g3_call_config_v2"
    assert "template v2" in body["input"][0]["content"] and "spoken_names" in json.dumps(body["text"]["format"]["schema"])
    pilot = G3RequestBuilder(g3_support.shared_inputs(), M.SCHEMAS).build("CALL_001", "A001")
    assert pilot["text"]["format"]["name"] == "g3_call_config_v1" and "template v1" in pilot["input"][0]["content"]


def errors_for(g3_builder, call_id, mutate):
    ctx = g3_builder.context(call_id)
    response = synthesize_response(ctx)
    mutate(response)
    return make_validator(g3_builder).errors(response, ctx)


def test_spoken_names_are_required_per_language_and_formatted(g3_builder):
    mandarin, mixed, singlish = PROFILE_CALLS["MANDARIN"], PROFILE_CALLS["EN_ZH_MIXED"], PROFILE_CALLS["SINGLISH"]
    assert not errors_for(g3_builder, mandarin, lambda r: None)
    assert any("client.spoken_names.zh: required" in e for e in errors_for(g3_builder, mandarin, lambda r: r["client"]["spoken_names"].update(zh=None)))
    assert any("client.spoken_names.en: must be null" in e for e in errors_for(g3_builder, mandarin, lambda r: r["client"]["spoken_names"].update(en="Alex Tan")))
    assert any("2-4 Han characters" in e for e in errors_for(g3_builder, mandarin, lambda r: r["client"]["spoken_names"].update(zh="Alex")))
    assert any("2-4 Han characters" in e for e in errors_for(g3_builder, mandarin, lambda r: r["employee"]["spoken_names"].update(zh="林")))
    assert any("employee.spoken_names.en: required" in e for e in errors_for(g3_builder, mixed, lambda r: r["employee"]["spoken_names"].update(en=None)))
    assert any("must equal client.name" in e for e in errors_for(g3_builder, mixed, lambda r: r["client"]["spoken_names"].update(en="Someone Else")))
    assert any("spoken_names.zh: must be null" in e for e in errors_for(g3_builder, singlish, lambda r: r["client"]["spoken_names"].update(zh="陈亚历")))
    assert any("planning name must be written in English letters" in e for e in errors_for(g3_builder, mandarin, lambda r: r["client"].update(name="陈亚历")))


def test_the_scenario_prose_stays_english_and_names_no_chinese_product(g3_builder):
    call_id = PROFILE_CALLS["MANDARIN"]
    ctx = g3_builder.context(call_id)
    assert any("must be written in English" in e for e in errors_for(g3_builder, call_id, lambda r: r["client"].update(background="一位长期客户")))
    product = ctx.zh_product_names[0]
    assert any("Chinese product name" in e for e in errors_for(g3_builder, call_id, lambda r: r["final_product"].update(description=f"A fund called {product}.")))


def test_all_250_multilingual_g3_responses_validate_and_the_old_response_shape_is_rejected(g3_builder):
    validator, upstream = make_validator(g3_builder), M.upstream_inputs()
    for call_id in upstream.call_ids:
        ctx = g3_builder.context(call_id)
        assert not validator.errors(synthesize_response(ctx), ctx), call_id
    ctx = g3_builder.context("CALL_001")
    old_shape = synthesize_response(ctx)
    del old_shape["client"]["spoken_names"]
    assert any("spoken_names" in e for e in validator.errors(old_shape, ctx))


# ============================================================================================ the paid-run guard
class LiveLikeTransport:
    """Not one of the fake transports: the guard and the exposure record treat it as a real provider."""

    def __init__(self, inner, on_send=None):
        self.inner, self.on_send = inner, on_send

    def create_response(self, body):
        if self.on_send:
            self.on_send(body)
        return self.inner.create_response(body)


def g3_runner(tmp_path, *, live=False, on_send=None, inputs=None):
    root = tmp_path / "release"
    root.mkdir(exist_ok=True)
    holder = {}
    fake = FakeTransport(lambda body: holder["respond"](body))
    transport = LiveLikeTransport(fake, on_send) if live else fake
    runner = ConfigRunner(root, audit_root=tmp_path / "audit", schema_dir=M.SCHEMAS, transport=transport, clock=g3_support.Clock(),
                          inputs=inputs or M.upstream_inputs())
    holder["respond"] = g3_support.good_responder(runner)
    return runner


CLEAN = {"code_commit": "c0ffee1", "code_dirty_paths": []}


def test_a_live_run_needs_an_allowlisted_batch_for_its_stage_and_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: dict(CLEAN))
    runner = g3_runner(tmp_path, live=True)
    with pytest.raises(RunError, match="not all in one G3 batch"):
        runner.run_sync("CALL_001")  # no allowlist file at all
    assert not runner.store.attempt_ids("CALL_001"), "nothing is written before the guard passes"
    M.write_allowlist(runner.root, {"B1": {"stages": ["G4"], "call_ids": ["CALL_001"]}})
    with pytest.raises(RunError, match="not all in one G3 batch"):
        runner.run_sync("CALL_001")  # a batch for another stage does not authorise G3
    M.write_allowlist(runner.root, {"B1": {"stages": ["G3"], "call_ids": ["CALL_002"]}})
    with pytest.raises(RunError, match="CALL_001"):
        runner.run_sync("CALL_001")  # a batch with other calls does not authorise this call
    M.write_allowlist(runner.root, {"B1": {"stages": ["G3"], "call_ids": ["CALL_001"]}})
    assert runner.run_sync("CALL_001")["state"] == "ACCEPTED"


def test_a_batch_submission_must_be_covered_by_one_allowlisted_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: dict(CLEAN))
    runner = g3_runner(tmp_path, live=True)
    M.write_allowlist(runner.root, {"B1": {"stages": ["G3"], "call_ids": ["CALL_001", "CALL_002"]}})
    with pytest.raises(RunError, match="CALL_003"):
        runner._paid_guard(["CALL_001", "CALL_003"], runner.transport())
    runner._paid_guard(["CALL_001", "CALL_002"], runner.transport())
    assert runner._allowlist_batch == "B1"


def test_a_live_run_refuses_uncommitted_code_but_ignores_unrelated_changes(tmp_path, monkeypatch):
    runner = g3_runner(tmp_path, live=True)
    M.write_allowlist(runner.root, {"B1": {"stages": ["G3"], "call_ids": ["CALL_001"]}})
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: {"code_commit": "c0ffee1", "code_dirty_paths": ["src/fyp_evaluation/x.py"]})
    with pytest.raises(RunError, match="uncommitted code"):
        runner.run_sync("CALL_001")
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: {"code_commit": None, "code_dirty_paths": []})
    with pytest.raises(RunError, match="code commit"):
        runner.run_sync("CALL_001")
    assert live_guard.CODE_PATHS == ("src", "prompts", "schemas", "scripts"), "only code directories matter; data and runtime files never block"


def test_the_guard_never_touches_fake_transports_or_the_historical_release(tmp_path, monkeypatch):
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: pytest.fail("a fake transport must not even look at git"))
    assert g3_runner(tmp_path).run_sync("CALL_001")["state"] == "ACCEPTED"  # fake transport, no allowlist, dirty tree: fine
    pilot = ConfigRunner(tmp_path / "pilot", audit_root=tmp_path / "audit_pilot", schema_dir=M.SCHEMAS, inputs=g3_support.shared_inputs(),
                         transport=LiveLikeTransport(FakeTransport(lambda body: None)), clock=g3_support.Clock())
    pilot._paid_guard(["CALL_001"], pilot.transport())  # the historical release is never subject to the allowlist
    assert pilot._exposure() is None


def test_the_live_guard_is_loaded_from_the_committed_file_shape():
    assert live_guard.is_fake_transport(None) and live_guard.is_fake_transport(FakeTransport(lambda b: None))
    assert not live_guard.is_fake_transport(LiveLikeTransport(None))
    with pytest.raises(ValueError, match="no calls"):
        live_guard.authorised_batch(M.RELEASE, "G3", [])


# ============================================================================================ exposure persistence and the ledger
def test_the_exposure_record_is_persisted_in_the_prepared_attempt_before_anything_is_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: dict(CLEAN))
    seen = {}

    def on_send(body):
        attempt = runner.store.read_attempt("CALL_001", "A001")  # read from disk, mid-send
        seen.update(state=attempt["state"], exposure=attempt.get("exposure"))

    runner = g3_runner(tmp_path, live=True, on_send=on_send)
    M.write_allowlist(runner.root, {"B1": {"stages": ["G3"], "call_ids": ["CALL_001"]}})
    runner.run_sync("CALL_001")
    assert seen["state"] == "PREPARED"
    assert seen["exposure"] == {"run_kind": "live", **CLEAN, "allowlist_batch": "B1"}


def test_the_exposure_record_survives_an_interrupted_attempt_and_reaches_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: dict(CLEAN))

    def interrupted(body):
        raise AmbiguousError("connection lost after the request left")

    runner = g3_runner(tmp_path, live=True, on_send=interrupted)
    M.write_allowlist(runner.root, {"B1": {"stages": ["G3"], "call_ids": ["CALL_001"]}})
    attempt = runner.run_sync("CALL_001")
    assert attempt["state"] == "TRANSPORT_FAILED" and attempt["possibly_billed"] is True
    on_disk = runner.store.read_attempt("CALL_001", "A001")
    assert on_disk["exposure"]["run_kind"] == "live" and on_disk["exposure"]["code_commit"] == "c0ffee1"
    ledger = exposure.build(runner.root, tmp_path / "audit", runner.inputs.dataset_id)
    row = ledger["live_attempts"][0]
    assert (row["stage"], row["call_id"], row["attempt_id"], row["state"]) == ("G3", "CALL_001", "A001", "TRANSPORT_FAILED")
    assert row["possibly_billed"] is True and row["code_commit"] == "c0ffee1" and row["allowlist_batch"] == "B1"
    assert ledger["exposed_calls"] == {"CALL_001": ["G3"]} and ledger["summary"]["possibly_billed"] == 1
    assert exposure.render(runner.root, tmp_path / "audit", runner.inputs.dataset_id) == exposure.render(runner.root, tmp_path / "audit",
                                                                                                          runner.inputs.dataset_id)


def test_offline_attempts_are_labelled_offline_and_never_count_as_exposure(tmp_path):
    runner = g3_runner(tmp_path)
    assert runner.run_sync("CALL_002")["state"] == "ACCEPTED"
    attempt = runner.store.read_attempt("CALL_002", "A001")
    assert attempt["exposure"] == {"run_kind": "offline", "code_commit": None, "code_dirty_paths": [], "allowlist_batch": None}
    ledger = exposure.build(runner.root, tmp_path / "audit", runner.inputs.dataset_id)
    assert ledger["live_attempts"] == [] and ledger["exposed_calls"] == {}
    committed = json.loads((runner.root / "authoring/call_config_provenance/CALL_002.json").read_text(encoding="utf-8"))
    assert committed["attempts"][0]["exposure"]["run_kind"] == "offline"


def test_the_historical_release_records_no_exposure_field(tmp_path):
    runner = g3_support.make_runner(tmp_path)[0]
    runner.run_sync("CALL_001")
    assert "exposure" not in runner.store.read_attempt("CALL_001", "A001")


# ============================================================================================ the register guide as an upstream input
def calls_by_profile_with_plans(profile):
    return PROFILE_CALLS[profile]


def test_the_guide_hash_applies_to_singlish_calls_only_and_resolves_to_its_version():
    inputs = M.plan_inputs()
    sing, mand = inputs.specs[PROFILE_CALLS["SINGLISH"]], inputs.specs[PROFILE_CALLS["MANDARIN"]]
    assert register_guide.applies(sing) and not register_guide.applies(mand) and not register_guide.applies({"metadata": {"language": "en-SG"}})
    assert list(register_guide.extra_hashes(sing)) == ["register_guide_sha256"] and register_guide.extra_hashes(mand) == {}
    assert register_guide.version_of(register_guide.guide_sha256()) == "v1"
    assert register_guide.version_of("0" * 64) is None
    assert register_guide.brief_block(mand) is None and register_guide.brief_block(sing)["version"] == "v1"


def test_the_g4_attempt_and_artifact_record_the_guide_hash_only_for_singlish_calls():
    inputs = M.plan_inputs()
    runner = CheckPlanRunner(M.RELEASE, schema_dir=M.SCHEMAS, inputs=inputs)
    for profile, expected in (("SINGLISH", True), ("MANDARIN", False), ("ENGLISH", False), ("EN_ZH_MIXED", False)):
        call_id = PROFILE_CALLS[profile]
        assert ("register_guide_sha256" in runner._upstream_hashes(call_id)) is expected
        text = M.plan_texts([call_id])[call_id]
        assert ("register_guide_sha256" in json.loads(text)["source"]) is expected
    brief = json.loads(runner.builder.build(PROFILE_CALLS["SINGLISH"], "A001")["input"][1]["content"])
    assert brief["register_guide"]["version"] == "v1" and "caricature" in brief["register_guide"]["text"]
    assert "register_guide" not in json.loads(runner.builder.build(PROFILE_CALLS["MANDARIN"], "A001")["input"][1]["content"])


@pytest.fixture()
def changed_guide(tmp_path, monkeypatch):
    """A scratch guide directory whose current version has different text (the repository file is never touched)."""
    directory = tmp_path / "guides"
    directory.mkdir()
    for path in register_guide.GUIDE_DIR.glob("register_singlish.*.md"):
        shutil.copyfile(path, directory / path.name)
    monkeypatch.setattr(register_guide, "GUIDE_DIR", directory)
    return directory


def g4_scratch(tmp_path, call_id):
    """A scratch dataset holding an accepted-looking artificial G4 plan (and its G3 config) for one call."""
    root = M.scratch_release(tmp_path / "dataset")
    inputs = M.plan_inputs(root, [call_id])
    (root / "authoring/call_configs").mkdir(parents=True, exist_ok=True)
    (root / "authoring/call_configs" / f"{call_id}.json").write_bytes(inputs.config_texts[call_id].encode("utf-8"))
    runner = CheckPlanRunner(root, audit_root=tmp_path / "audit", schema_dir=M.SCHEMAS, inputs=inputs)
    from fyp_evaluation.check_plans.compile import compile_artifact
    from fyp_evaluation.check_plans.feasibility import synthesize_response as synth
    ctx = runner.builder.context(call_id)
    generation = {"attempt_id": "A001", "model": "fixture", "prompt_template_sha256": "0" * 64,
                  "response_schema_sha256": runner.builder.response_schema_sha256}
    plan = compile_artifact(inputs, ctx, synth(ctx), generation, runner.builder.response_schema)
    (root / "authoring/check_plans").mkdir(parents=True, exist_ok=True)
    (root / "authoring/check_plans" / f"{call_id}.json").write_bytes(M.serialize_json(plan).encode("utf-8"))
    return runner


def test_a_guide_change_makes_an_accepted_singlish_plan_stale_and_leaves_other_profiles_current(tmp_path, changed_guide):
    sing, mand = PROFILE_CALLS["SINGLISH"], PROFILE_CALLS["MANDARIN"]
    for call_id, affected in ((sing, True), (mand, False)):
        runner = g4_scratch(tmp_path / call_id, call_id)
        assert runner.dependency_status([call_id])[call_id] == {"status": "current", "changed": [], "error": None}
        (changed_guide / "register_singlish.v1.md").write_text(register_guide.guide_text() + "\nA new sentence.\n", encoding="utf-8")
        status = runner.dependency_status([call_id])[call_id]
        assert (status["status"] == "stale" and status["changed"] == ["register_guide_sha256"]) if affected else status["status"] == "current"
        shutil.copyfile(register_guide.GUIDE_DIR.parent.parent / "prompts" / "shared" / "register_singlish.v1.md", changed_guide / "register_singlish.v1.md") \
            if False else (changed_guide / "register_singlish.v1.md").write_text(
                (M.RELEASE.parents[1] / "prompts/shared/register_singlish.v1.md").read_text(encoding="utf-8"), encoding="utf-8")


def test_a_guide_change_during_an_attempt_closes_it_stale_upstream(tmp_path, changed_guide):
    call_id = PROFILE_CALLS["SINGLISH"]
    runner = g4_scratch(tmp_path, call_id)
    for name in ("check_plans", "check_plan_provenance"):  # start from no accepted plan: this call is attempted afresh
        shutil.rmtree(runner.root / "authoring" / name, ignore_errors=True)

    def change_the_guide(body):
        (changed_guide / "register_singlish.v1.md").write_text(register_guide.guide_text() + "\nChanged mid-attempt.\n", encoding="utf-8")

    from fyp_evaluation.check_plans.feasibility import synthesize_response as synth
    holder = {}
    fake = FakeTransport(lambda body: holder["respond"](body))
    runner._transport = LiveLikeTransport(fake, change_the_guide)  # fake-like in effect, but the guard is bypassed below
    runner._allowlist_batch = None
    holder["respond"] = lambda body: response_body(json.dumps(synth(runner.builder.context(call_id))))
    runner._paid_guard = lambda call_ids, transport: None
    attempt = runner.run_sync(call_id)
    assert attempt["state"] == "STALE_UPSTREAM" and attempt["stale_upstream"]["changed"] == ["register_guide_sha256"]
    assert not (runner.root / "authoring/check_plans" / f"{call_id}.json").exists(), "a stale response is never accepted"


def test_the_recorded_guide_hash_resolves_back_to_its_exact_text_for_reconstruction(tmp_path, changed_guide):
    recorded = register_guide.guide_sha256()
    original = register_guide.guide_text()
    (changed_guide / "register_singlish.v2.md").write_text("A later, different guide.\n", encoding="utf-8")
    assert register_guide.version_of(recorded) == "v1" and register_guide.guide_text("v1") == original
    assert register_guide.version_of(register_guide.guide_sha256("v2")) == "v2"


# ============================================================================================ the guard covers G4 and G5 as well
def test_g4_and_g5_live_runs_are_refused_without_an_allowlisted_batch_before_any_attempt_exists(tmp_path, monkeypatch):
    from fyp_evaluation.plan_verifications.run import PlanVerificationRunner
    from fyp_evaluation.plan_verifications.inputs import load_inputs as load_g5_inputs
    from fyp_evaluation.check_plans.inputs import load_inputs as load_g4_inputs

    monkeypatch.setattr(live_guard, "code_state", lambda *a, **k: dict(CLEAN))
    call_id = M.first_call("SINGLISH")
    chain = M.chain(tmp_path, [call_id])
    assert chain.run_g3() == {call_id: "ACCEPTED"}
    g4 = CheckPlanRunner(chain.root, audit_root=tmp_path / "audit_live4", schema_dir=M.SCHEMAS, transport=LiveLikeTransport(None),
                         selected_call_ids=[call_id], inputs=load_g4_inputs(chain.root, M.SCHEMAS, [call_id]))
    with pytest.raises(RunError, match="not all in one G4 batch"):
        g4.run_sync(call_id)
    with pytest.raises(RunError, match="not all in one G4 batch"):
        g4.submit([call_id])
    assert not g4.store.attempt_ids(call_id)
    assert chain.run_g4() == {call_id: "ACCEPTED"}
    g5 = PlanVerificationRunner(chain.root, audit_root=tmp_path / "audit_live5", schema_dir=M.SCHEMAS, transport=LiveLikeTransport(None),
                                selected_call_ids=[call_id], inputs=load_g5_inputs(chain.root, M.SCHEMAS, [call_id]))
    with pytest.raises(RunError, match="not all in one G5 batch"):
        g5.run([call_id])
    assert not g5.store.attempt_ids(call_id)
    M.write_allowlist(chain.root, {"B7": {"stages": ["G4", "G5"], "call_ids": [call_id]}})
    g5._paid_guard([call_id], g5.transport())  # an allowlisted batch for this stage and call passes the guard
    assert g5._allowlist_batch == "B7"
