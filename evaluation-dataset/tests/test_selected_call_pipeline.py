"""Selected-call readiness through G7: the frozen 250-call G0-G2B release with accepted downstream outputs for only four calls.

G6 and G7 used to call G5's loader without the selection, which reached G4's complete-G3 requirement, so a four-call pilot could not
pass G5. Offline, fake transports only. Release-wide integrity checks and the historical (complete-mode) behaviour are unchanged.
"""

import json

import pytest

import multilingual_support as M
from fyp_evaluation.check_plans.inputs import G3StateError
from fyp_evaluation.dialogue_verifications.cli import main as g7_main
from fyp_evaluation.dialogue_verifications.inputs import load_inputs as load_g7
from fyp_evaluation.dialogues.cli import main as g6_main
from fyp_evaluation.dialogues.inputs import load_inputs as load_g6

PILOT = ["CALL_112", "CALL_030", "CALL_014", "CALL_002"]


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("selected")
    chain = M.chain(tmp, PILOT)
    assert chain.run_g3() == dict.fromkeys(PILOT, "ACCEPTED")
    assert chain.run_g4() == dict.fromkeys(PILOT, "ACCEPTED")
    assert chain.run_g5() == dict.fromkeys(PILOT, "ACCEPTED")
    chain.tmp = tmp
    return chain


def test_the_four_pilot_calls_cover_every_language_profile():
    specs = M.upstream_inputs().specs
    assert sorted(specs[c]["metadata"]["language_profile"] for c in PILOT) == sorted(M.PROFILES)


def test_four_selected_calls_progress_through_g6_and_g7_while_246_have_no_g3_output(pilot):
    assert set(pilot.run_g6().values()) == {"ACCEPTED"}
    assert set(pilot.run_g7().values()) == {"ACCEPTED"}
    authoring = pilot.root / "authoring"
    for stage in ("call_configs", "check_plans", "plan_verifications", "dialogues", "dialogue_verifications"):
        assert sorted(p.stem for p in (authoring / stage).glob("CALL_*.json")) == sorted(PILOT), stage
    for call_id in PILOT:
        report = json.loads((authoring / "dialogue_verifications" / f"{call_id}.json").read_text(encoding="utf-8"))
        assert report["verdict"] == "VALID" and report["schema_version"] == "2.0"
    assert len(M.upstream_inputs(pilot.root).call_ids) == 250
    # the runtime validation of both stages is clean, and only the selected calls are reported as eligible
    assert pilot.g6.validate() == [] and pilot.g7.validate() == []
    assert {c for c, row in pilot.g7.g8_eligibility().items() if row["eligible"]} == set(PILOT)


def test_unselected_calls_are_ineligible_not_an_error(pilot):
    other = next(c for c in M.upstream_inputs().call_ids if c not in PILOT)
    assert pilot.g6.eligibility(other)[0] is False and pilot.g7.eligibility(other)[0] is False
    assert pilot.g6.dependency_status([other])[other]["status"] == "none"


def test_complete_mode_still_requires_every_g3_config(pilot):
    for loader in (load_g6, load_g7):
        with pytest.raises(G3StateError, match="not complete"):
            loader(pilot.root, M.SCHEMAS)


def test_a_selected_call_without_an_accepted_g3_config_is_refused(pilot):
    other = next(c for c in M.upstream_inputs().call_ids if c not in PILOT)
    for loader in (load_g6, load_g7):
        with pytest.raises(G3StateError, match="no accepted G3 call config"):
            loader(pilot.root, M.SCHEMAS, [other])


def test_release_wide_g3_integrity_is_still_enforced_in_selected_mode(pilot, tmp_path):
    copy = M.copy_chain(pilot.root, tmp_path / "dataset")
    path = copy.root / "authoring/call_configs" / f"{PILOT[0]}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["scenario"]["setting"]["call_reason"] += " tampered"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for loader in (load_g6, load_g7):
        with pytest.raises(Exception):  # noqa: B017 - G3's own integrity error, whatever its subclass
            loader(copy.root, M.SCHEMAS, [PILOT[1]])


def test_the_cli_selects_calls_for_g6_and_g7_dry_runs(pilot, capsys):
    base = ["--dataset-root", str(pilot.root), "--audit-dir", str(pilot.tmp / "cli_audit"), "--schema-dir", str(M.SCHEMAS)]
    assert g6_main([*base, "--status", "--call-id", PILOT[0]]) == 0
    assert g7_main([*base, "--status", "--call-id", PILOT[0]]) == 0
    capsys.readouterr()
    assert g6_main([*base, "--dry-run", "--call-id", PILOT[0]]) == 0
    assert g7_main([*base, "--dry-run", "--call-id", PILOT[0]]) == 0
    other = next(c for c in M.upstream_inputs().call_ids if c not in PILOT)
    assert g6_main([*base, "--dry-run", "--call-id", other]) == 2  # a selected call with no accepted G3 config is a precondition error
