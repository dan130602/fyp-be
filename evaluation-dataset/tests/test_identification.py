"""ARTIFICIAL in-memory contracts only; never writes call artifacts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fixtures import documents
from fyp_evaluation.authoring.coverage import parse_k01_cell
from fyp_evaluation.validation import load_json, validate, validate_answers, validate_checklist
from fyp_evaluation.validation.identification import is_invalid_test_identifier

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
NAMES = ["call_config", "check_plan", "ground_truth"]


def identification_fixture(name, value):
    data = documents()[name]
    key = "intended_answers" if name == "call_config" else "checks"
    check = data[key][0]
    check.update(question_id="K01", expected={
        "kind": "extraction", "value_type": "text", "expected_value": value,
    })
    if name != "call_config":
        if value is None:
            check["evidence_plan"] = {
                "mode": "absence", "scope": "entire_dialogue", "required_speaker": "ANY",
                "absent_event": "ARTIFICIAL fixture: no identification number stated",
                "content_constraints": ["ARTIFICIAL fixture: no identifier anywhere"],
                "forbidden_patterns": [],
            }
            if name == "ground_truth":
                check.update(gold_evidence_turns=[], gold_evidence_text="")
        else:
            check["evidence_plan"]["planned_evidence"][0]["required_speaker"] = "CLIENT"
            if name == "ground_truth":
                data["dialogue"][0]["speaker"] = "CLIENT"
    return data, check


@pytest.mark.parametrize("value", ["S0000000X", "T0000000X", "F0000000A", "G0000000A", "M0000000A"])
def test_invalid_synthetic_checksum_is_required(value):
    assert is_invalid_test_identifier(value)


@pytest.mark.parametrize("value", [
    # Checksum-correct mathematical controls only, not sourced identities.
    "S0000000J", "T0000000G", "F0000000X", "G0000000R", "M0000000T",
    "", "null", "PASS", "FAIL", "N/A", "UNCLEAR", "TEST-NRIC-001",
    "s0000000x", "S000000X", "S00000000X", "S0000000X\n", " S0000000X",
    "S０００００００X", 123456789, None, True,
])
def test_cell_decoder_rejects_invalid_inputs_and_checksum_correct_controls(value):
    assert not is_invalid_test_identifier(value)
    with pytest.raises(ValueError):
        parse_k01_cell(value)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("cell", ["S0000000X", "NOT_STATED"])
def test_k01_value_or_absence_survives_all_json_contracts(name, cell):
    value = parse_k01_cell(cell)
    data, check = identification_fixture(name, value)
    before = deepcopy(data)
    checklist = [{"id": "K01", "type": "CLIENT_KYC", "field": "client_identification_number",
                  "question": "ARTIFICIAL identification fixture?"}]
    normalization = load_json(ROOT / "configs/normalization.json")
    assert validate_answers(data, name, checklist, normalization, SCHEMAS) == []
    assert json.loads(json.dumps(check))["expected"]["expected_value"] == value
    if cell == "NOT_STATED":
        assert value is None
        assert '"expected_value": null' in json.dumps(check)
    assert data == before


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", ["NOT_STATED", "null", "", "PASS", "N/A", 123456789, "S0000000J"])
def test_k01_rejects_sentinels_and_invalid_gold_representations(name, value):
    data, _ = identification_fixture(name, value)
    assert validate(data, name, SCHEMAS)


@pytest.mark.parametrize("name", NAMES)
def test_null_exception_does_not_extend_to_other_question_ids(name):
    data, check = identification_fixture(name, None)
    check["question_id"] = "FIX_OTHER"
    assert validate(data, name, SCHEMAS)


@pytest.mark.parametrize("name", ["check_plan", "ground_truth"])
@pytest.mark.parametrize("case", ["null_presence", "null_client_only", "populated_absence", "populated_employee"])
def test_identification_evidence_requirements(name, case):
    value = None if case.startswith("null") else "S0000000X"
    data, check = identification_fixture(name, value)
    if case == "null_presence":
        check["evidence_plan"] = documents()["check_plan"]["checks"][0]["evidence_plan"]
    elif case == "null_client_only":
        check["evidence_plan"]["required_speaker"] = "CLIENT"
    elif case == "populated_absence":
        check["evidence_plan"] = identification_fixture(name, None)[1]["evidence_plan"]
    else:
        check["evidence_plan"]["planned_evidence"][0]["required_speaker"] = "EMPLOYEE"
    assert validate(data, name, SCHEMAS)


@pytest.mark.parametrize("field,value", [("gold_evidence_turns", [0]), ("gold_evidence_text", "invented quote")])
def test_null_ground_truth_requires_empty_evidence(field, value):
    data, check = identification_fixture("ground_truth", None)
    check[field] = value
    assert validate(data, "ground_truth", SCHEMAS)


@pytest.mark.parametrize("case", ["wrong_field", "wrong_category", "wrong_rule"])
def test_k01_exception_requires_identification_metadata(case):
    checklist = [{"id": "K01", "type": "CLIENT_KYC", "field": "client_identification_number",
                  "question": "ARTIFICIAL fixture?"}]
    normalization = load_json(ROOT / "configs/normalization.json")
    if case == "wrong_field":
        checklist[0]["field"] = "transaction_instrument"
    elif case == "wrong_category":
        checklist[0]["type"] = "TRANSACTION"
    else:
        normalization["field_rules"]["client_identification_number"] = "exact_integer"
    assert validate_checklist(checklist, normalization, SCHEMAS)


@pytest.mark.parametrize("name", NAMES)
def test_pre_nullable_contract_versions_need_explicit_migration(name):
    data, _ = identification_fixture(name, None)
    data["schema_version"] = "0.3.0"
    assert validate(data, name, SCHEMAS)


def test_pilot_k01_question_wording():
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    assert next(item for item in checklist if item["id"] == "K01") == {
        "id": "K01", "type": "CLIENT_KYC", "field": "client_identification_number",
        "question": "What is the client's NRIC or FIN number?",
    }
