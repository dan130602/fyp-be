"""G4 selected-call readiness: only the selected calls need an accepted, current, valid G3 config.

Everything runs on isolated copies of the historical pilot with fake transports. The G3 outputs of the copy are produced by
accepting just two calls through G3's own fake-transport flow, so the partial state is a genuine one (provenance, record and
artifacts consistent), not a hand-edited one.
"""

import pytest

import g3_support
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.check_plans.inputs import G3StateError, load_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner, RunError
from fyp_evaluation.llm.fake_transport import FakeTransport
from g4_support import SCHEMAS, Clock, copy_release, good_responder

ACCEPTED = ("CALL_001", "CALL_002")


@pytest.fixture()
def partial_g3(tmp_path, monkeypatch):
    """A pilot copy whose G3 has accepted exactly CALL_001 and CALL_002."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = copy_release(tmp_path / "release", keep_g3=False)
    holder = {}
    fake = FakeTransport(lambda body: holder["respond"](body))
    runner = ConfigRunner(root, audit_root=tmp_path / "audit_g3", schema_dir=SCHEMAS, transport=fake, clock=g3_support.Clock(),
                          inputs=g3_support.shared_inputs())
    holder["respond"] = g3_support.good_responder(runner)
    for call_id in ACCEPTED:
        assert runner.run_sync(call_id)["state"] == "ACCEPTED"
    return root


def g4_runner(root, tmp_path, selected=None):
    holder = {}
    fake = FakeTransport(lambda body: holder["respond"](body))
    runner = CheckPlanRunner(root, audit_root=tmp_path / "audit_g4", schema_dir=SCHEMAS, transport=fake, clock=Clock(),
                             selected_call_ids=selected)
    holder["respond"] = good_responder(runner)
    return runner


def test_a_selected_call_with_an_accepted_current_config_loads_while_unrelated_calls_have_none(partial_g3):
    inputs = load_inputs(partial_g3, SCHEMAS, ["CALL_001"])
    assert inputs.ready_call_ids == list(ACCEPTED) == list(inputs.configs) and len(inputs.call_ids) == 50
    assert "CALL_007" not in inputs.configs and "CALL_007" not in inputs.config_hashes


def test_a_selected_call_without_an_accepted_config_is_refused(partial_g3):
    with pytest.raises(G3StateError, match="no accepted G3 call config for the selected call"):
        load_inputs(partial_g3, SCHEMAS, ["CALL_001", "CALL_007"])
    with pytest.raises(G3StateError, match="unknown call id"):
        load_inputs(partial_g3, SCHEMAS, ["CALL_999"])


def test_without_a_selection_the_whole_release_still_needs_a_complete_g3(partial_g3):
    with pytest.raises(G3StateError, match="G3 call configs are not complete"):
        load_inputs(partial_g3, SCHEMAS)
    with pytest.raises(G3StateError, match="G3 call configs are not complete"):
        load_inputs(partial_g3, SCHEMAS, None)


def test_corrupt_unrelated_g3_state_still_blocks_a_selected_run(partial_g3):
    path = partial_g3 / "authoring/call_configs/CALL_002.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"Alex Tan", b"Alex Tay", 1) if b"Alex Tan" in original else original + b" ")
    with pytest.raises(G3StateError, match="CALL_002"):
        load_inputs(partial_g3, SCHEMAS, ["CALL_001"])
    path.write_bytes(original)
    assert load_inputs(partial_g3, SCHEMAS, ["CALL_001"]).ready_call_ids == list(ACCEPTED)


def test_the_runner_is_eligible_only_for_selected_calls_and_validate_still_demands_every_plan_when_asked(partial_g3, tmp_path):
    runner = g4_runner(partial_g3, tmp_path, selected=["CALL_001"])
    assert runner.eligibility("CALL_001")[0] and runner.eligibility("CALL_002")[0]  # both have an accepted, current, valid config
    ok, reason = runner.eligibility("CALL_007")
    assert not ok and "no accepted G3 call config" in reason
    with pytest.raises(RunError, match="CALL_007"):
        runner._select(["CALL_007"], retry_rejected=False, reopen_call_ids=set())
    assert runner.run_sync("CALL_001")["state"] == "ACCEPTED"
    assert runner.validate() == []
    assert any("no accepted check plan" in e for e in runner.validate(require_complete=True))
    assert not any(runner.check().values())


def test_a_second_selection_and_the_unselected_gate_behave_independently(partial_g3, tmp_path):
    first = g4_runner(partial_g3, tmp_path, selected=["CALL_001"])
    assert first.run_sync("CALL_001")["state"] == "ACCEPTED"
    second = g4_runner(partial_g3, tmp_path, selected=["CALL_002"])
    assert second.inputs.ready_call_ids == list(ACCEPTED) and second.run_sync("CALL_002")["state"] == "ACCEPTED"
    assert second.validate() == []  # the earlier CALL_001 plan is still validated against its own G3 config
    whole = CheckPlanRunner(partial_g3, audit_root=tmp_path / "audit_whole", schema_dir=SCHEMAS)
    with pytest.raises(G3StateError):
        whole.inputs
