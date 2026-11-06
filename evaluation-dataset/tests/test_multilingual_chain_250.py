"""Phase 3A-4 gate: the offline G3 -> G4 -> G5 chain over all 250 calls of the multilingual release (fake transports only),
with the staleness cascade (including the register guide), selected calls, historical reconstruction, the paid-run guard
and exposure persistence. One module-scoped chain is built once; the checks read it and never mutate it."""

import json
import shutil

import pytest

import multilingual_support as M
import g3_support
from fyp_evaluation import register_guide
from fyp_evaluation.call_configs.feasibility import synthesize_response as synthesize_config
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.call_configs.slots import CallContext
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body
from fyp_evaluation.check_plans.inputs import load_inputs as load_g4_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.llm import exposure
from fyp_evaluation.plan_verifications.inputs import load_inputs as load_g5_inputs
from fyp_evaluation.plan_verifications.run import PlanVerificationRunner

ALL = list(M.upstream_inputs().call_ids)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chain250")
    chain = M.chain(tmp, ALL)
    assert set(chain.run_g3().values()) == {"ACCEPTED"}
    assert set(chain.run_g4().values()) == {"ACCEPTED"}
    assert set(chain.run_g5().values()) == {"ACCEPTED"}
    chain.tmp = tmp
    return chain


def singlish_calls():
    specs = M.upstream_inputs().specs
    return [c for c in ALL if register_guide.applies(specs[c])]


def test_all_250_calls_are_accepted_at_every_stage_with_their_provenance(chain):
    for stage in ("call_configs", "check_plans", "plan_verifications"):
        assert len(list((chain.root / "authoring" / stage).glob("CALL_*.json"))) == 250
    full = load_g5_inputs(chain.root, M.SCHEMAS)  # complete mode: every call has an accepted config and plan
    assert len(full.plans) == 250 and len(full.call_ids) == 250
    assert {json.loads(p.read_text(encoding="utf-8"))["schema_version"] for p in (chain.root / "authoring/plan_verifications").glob("CALL_*.json")} == {"2.0"}


def test_exactly_the_singlish_artifacts_record_the_guide_hash(chain):
    singlish = set(singlish_calls())
    assert len(singlish) == 50
    for stage, directory in (("G4", "check_plans"), ("G5", "plan_verifications")):
        for call_id in ALL:
            source = json.loads((chain.root / "authoring" / directory / f"{call_id}.json").read_text(encoding="utf-8"))["source"]
            assert ("register_guide_sha256" in source) is (call_id in singlish), (stage, call_id)
            assert source["language_profiles_sha256"]


def test_the_chain_is_current_and_the_offline_attempts_are_never_exposure(chain):
    assert all(s["status"] == "current" for s in chain.g4.dependency_status(ALL).values())
    assert all(s["status"] == "current" for s in chain.g5.dependency_status(ALL).values())
    attempt = chain.g5.store.read_attempt(ALL[0], "A001")
    assert attempt["exposure"]["run_kind"] == "offline" and attempt["exposure"]["code_commit"] is None
    ledger = exposure.build(chain.root, chain.audit_root, chain.g3.inputs.dataset_id)
    assert ledger["live_attempts"] == [] and ledger["summary"]["exposed_calls"] == 0


def test_a_guide_change_stales_only_singlish_plans_and_reports_and_the_old_version_stays_resolvable(chain, tmp_path, monkeypatch):
    directory = tmp_path / "guides"
    directory.mkdir()
    for path in register_guide.GUIDE_DIR.glob("register_singlish.*.md"):
        shutil.copyfile(path, directory / path.name)
    recorded = register_guide.guide_sha256()
    monkeypatch.setattr(register_guide, "GUIDE_DIR", directory)
    (directory / "register_singlish.v1.md").write_text(register_guide.guide_text() + "\nA wording change.\n", encoding="utf-8")
    singlish = set(singlish_calls())
    for runner in (chain.g4, chain.g5):
        statuses = runner.dependency_status(ALL)
        stale = {c for c, s in statuses.items() if s["status"] == "stale"}
        assert stale == singlish and all(statuses[c]["changed"] == ["register_guide_sha256"] for c in stale)
    # the recorded hash no longer matches the edited v1, so the exact earlier request cannot be rebuilt from it: a change is a new
    # version file, and the old one stays resolvable
    assert register_guide.version_of(recorded) is None
    (directory / "register_singlish.v1.md").write_text((M.RELEASE.parents[1] / "prompts/shared/register_singlish.v1.md").read_text(encoding="utf-8"),
                                                       encoding="utf-8")
    assert register_guide.version_of(recorded) == "v1"
    assert all(s["status"] == "current" for s in chain.g4.dependency_status(ALL).values())


def test_a_g3_reopen_cascades_to_g4_and_g5_for_that_call_only(chain, tmp_path):
    call_id = ALL[10]
    root = M.scratch_release(tmp_path / "copy")
    for name in ("call_configs", "call_config_provenance", "check_plans", "check_plan_provenance", "plan_verifications", "plan_verification_provenance"):
        shutil.copytree(chain.root / "authoring" / name, root / "authoring" / name, dirs_exist_ok=True)
    for name in ("call_config_record.json", "check_plan_record.json", "plan_verification_record.json"):
        shutil.copyfile(chain.root / "authoring" / name, root / "authoring" / name)

    def reworded(body):
        response = synthesize_config(CallContext(M.upstream_inputs(root), call_id))
        response["setting"]["conversation_style"] += " Slightly more formal."
        return response_body(json.dumps(response), response_id="resp_reopen")

    holder = ConfigRunner(root, audit_root=tmp_path / "audit", schema_dir=M.SCHEMAS, clock=g3_support.Clock(), inputs=M.upstream_inputs(root),
                          transport=FakeTransport(reworded))
    assert holder.run_sync(call_id, reopen_reason="cascade test")["state"] == "ACCEPTED"
    g4 = CheckPlanRunner(root, audit_root=tmp_path / "audit4", schema_dir=M.SCHEMAS, inputs=load_g4_inputs(root, M.SCHEMAS))
    g5 = PlanVerificationRunner(root, audit_root=tmp_path / "audit5", schema_dir=M.SCHEMAS, inputs=load_g5_inputs(root, M.SCHEMAS))
    for runner in (g4, g5):
        statuses = runner.dependency_status(ALL)
        assert {c for c, s in statuses.items() if s["status"] == "stale"} == {call_id}


def test_selected_call_mode_runs_one_call_and_leaves_the_others_untouched(tmp_path):
    calls = [ALL[0], ALL[60]]
    chain = M.chain(tmp_path, calls)
    assert set(chain.run_g3().values()) == {"ACCEPTED"}
    assert not (chain.root / "authoring/call_configs" / f"{ALL[1]}.json").exists()
    with pytest.raises(Exception, match="no accepted G3 call config"):
        load_g4_inputs(chain.root, M.SCHEMAS, [ALL[1]])
    assert set(chain.run_g4().values()) == {"ACCEPTED"}
    only = M.chain(tmp_path / "second", calls)
    only.run_g3(), only.run_g4()
    inputs = load_g5_inputs(only.root, M.SCHEMAS, [calls[0]])
    assert set(inputs.plans) == set(calls), "every accepted plan stays loaded, only the selected call is runnable"
    runner = PlanVerificationRunner(only.root, audit_root=only.audit_root, schema_dir=M.SCHEMAS, selected_call_ids=[calls[0]], inputs=inputs)
    assert runner._runnable(calls[0])
