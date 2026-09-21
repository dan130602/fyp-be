"""Artificial contract fixtures and pilot checklist checks; no approval of gold."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fixtures import documents, mixed_documents
from fyp_evaluation.validation import (
    check_schemas, load_json, to_langgraph_checklist, validate, validate_answers,
    validate_checklist,
)
from fyp_evaluation.validation.__main__ import main
from fyp_evaluation.validation.schemas import SCHEMA_NAMES
from fyp_evaluation.validation.dates import ENGLISH_MONTHS, is_canonical_dob

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


def test_schema_definitions():
    check_schemas(SCHEMAS)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_artificial_valid_documents(name):
    data = documents()[name]
    before = deepcopy(data)
    assert validate(data, name, SCHEMAS) == []
    assert data == before


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_required_structure(name):
    assert validate([] if name == "checklist" else {}, name, SCHEMAS)


@pytest.mark.parametrize("name", ["call_config", "check_plan", "ground_truth"])
@pytest.mark.parametrize("value_type,value", [
    ("text", "artificial value"), ("integer", 7), ("number", 1.25),
    ("boolean", False), ("dob_dd_month_yyyy", "01 January 2000"),
])
def test_extraction_preserves_typed_value(name, value_type, value):
    data = documents()[name]
    items = data["intended_answers" if name == "call_config" else "checks"]
    items[0]["expected"] = {"kind": "extraction", "value_type": value_type, "expected_value": value}
    assert validate(data, name, SCHEMAS) == []


@pytest.mark.parametrize("expected", [
    {"kind": "compliance", "expected_verdict": "N/A"},
    {"kind": "compliance", "expected_verdict": "UNCLEAR"},
    {"kind": "extraction", "value_type": "integer", "expected_value": "7"},
    {"kind": "extraction", "value_type": "boolean", "expected_value": 1},
    {"kind": "extraction", "value_type": "dob_dd_month_yyyy", "expected_value": 1012000},
    {"kind": "extraction", "value_type": "dob_dd_month_yyyy", "expected_value": "31 February 2000"},
    {"kind": "extraction", "value_type": "text", "expected_value": None},
    {"kind": "extraction", "value_type": "text", "expected_value": "fixture", "expected_verdict": "PASS"},
])
def test_invalid_gold_answers(expected):
    data = documents()["check_plan"]
    data["checks"][0]["expected"] = expected
    assert validate(data, "check_plan", SCHEMAS)


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
def test_absence_does_not_require_positive_quote(verdict):
    data = documents()["ground_truth"]
    check = data["checks"][0]
    check["expected"]["expected_verdict"] = verdict
    check["evidence_plan"] = {
        "mode": "absence", "scope": "entire_dialogue", "required_speaker": "ANY",
        "absent_event": "ARTIFICIAL missing marker", "content_constraints": [],
    }
    check["gold_evidence_turns"] = []
    check["gold_evidence_text"] = ""
    assert validate(data, "ground_truth", SCHEMAS) == []
    check["gold_evidence_turns"] = [0]
    assert validate(data, "ground_truth", SCHEMAS)


@pytest.mark.parametrize("turn_id", ["T001", "0", -1, 1, True, 0.0])
def test_invalid_turn_ids(turn_id):
    data = documents()["ground_truth"]
    data["dialogue"][0]["turn_id"] = turn_id
    assert validate(data, "ground_truth", SCHEMAS)


def test_pipeline_boundary_and_unresolved_authoring():
    data = documents()
    data["checklist"][0]["answer_type"] = "compliance"
    data["dialogue"]["turns"][0]["turn_id"] = 0
    data["check_plan"]["checks"][0]["gold_evidence_turns"] = [0]
    data["ground_truth"]["dialogue"][0]["voice_id"] = "fixture"
    for name in ["checklist", "dialogue", "check_plan", "ground_truth"]:
        assert validate(data[name], name, SCHEMAS)


def test_duplicate_question_ids():
    data = documents()["checklist"]
    data.append({**data[0], "text": "Another artificial question"})
    assert any("Duplicate id" in e for e in validate(data, "checklist", SCHEMAS))


def test_missing_and_ambiguous_evidence():
    data = documents()["ground_truth"]
    data["dialogue"].append({**data["dialogue"][0], "turn_id": 1})
    assert any("exactly one" in e for e in validate(data, "ground_truth", SCHEMAS))
    data["dialogue"] = [{"turn_id": 0, "speaker": "CLIENT", "text": "artificial marker", "critical_tokens": []}]
    assert any("exactly one" in e for e in validate(data, "ground_truth", SCHEMAS))


def test_resolved_evidence_must_agree():
    data = documents()["ground_truth"]
    data["checks"][0]["gold_evidence_turns"] = [99]
    data["checks"][0]["gold_evidence_text"] = "invented quote"
    errors = validate(data, "ground_truth", SCHEMAS)
    assert any("missing turn" in e for e in errors)
    assert any("text disagrees" in e for e in errors)


def test_evidence_ids_require_integer_json_numbers():
    data = documents()["ground_truth"]
    data["checks"][0]["gold_evidence_turns"] = [0.0]
    assert validate(data, "ground_truth", SCHEMAS)


def test_multiple_planned_phrases_in_one_turn():
    data = documents()["ground_truth"]
    check = data["checks"][0]
    check["evidence_plan"]["planned_evidence"].append({
        "evidence_id": "FIX_SECOND", "required_speaker": "EMPLOYEE", "text": "marker",
    })
    check["gold_evidence_text"] = "artificial marker\nmarker"
    assert validate(data, "ground_truth", SCHEMAS) == []


def test_critical_tokens_must_be_spoken():
    data = documents()["dialogue"]
    data["turns"][0]["critical_tokens"] = ["not spoken"]
    assert validate(data, "dialogue", SCHEMAS)


@pytest.mark.parametrize("case", ["reverse", "beyond_audio", "word_outside", "overlap", "id_gap"])
def test_invalid_timing_relationships(case):
    data = documents()["timings"]
    turn = data["turns"][0]
    if case == "reverse":
        turn["start_seconds"] = 1.5
    elif case == "beyond_audio":
        turn["end_seconds"] = 3
    elif case == "word_outside":
        turn["words"][0]["end_seconds"] = 1.5
    elif case == "overlap":
        data["turns"].append({"turn_id": 1, "start_seconds": 0.5, "end_seconds": 2})
    else:
        turn["turn_id"] = 2
    assert validate(data, "timings", SCHEMAS)


@pytest.mark.parametrize("case", ["valid", "missing", "unknown_rule", "wrong_type", "wrong_dob", "bad_compliance"])
def test_checklist_normalization_links(case):
    """Replaces metadata duplication checks with reusable field-rule checks."""
    data = mixed_documents()
    normalization = data["normalization"]
    if case == "missing":
        del normalization["field_rules"]["dob"]
    elif case == "unknown_rule":
        normalization["field_rules"]["dob"] = "missing_rule"
    elif case == "wrong_type":
        normalization["field_rules"]["fixture_quantity"] = "fixture_exact"
    elif case == "wrong_dob":
        normalization["field_rules"]["dob"] = "fixture_text"
    elif case == "bad_compliance":
        normalization["compliance_rule"] = "fixture_integer"
    errors = validate_checklist(data["checklist"], normalization, SCHEMAS)
    assert bool(errors) == (case != "valid")


@pytest.mark.parametrize("raw", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_strict_json_loading(tmp_path, raw):
    path = tmp_path / "artificial.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        load_json(path)


def test_templates_cannot_pass_as_approved_data():
    # Keep the empty-input guard independent of the now-populated checklist.
    assert validate([], "checklist", SCHEMAS)
    assert validate(load_json(ROOT / "configs/normalization.json"), "normalization", SCHEMAS) == []


def test_cli_schema_check(monkeypatch):
    monkeypatch.setattr("sys.argv", ["fyp-validate", "--schema-dir", str(SCHEMAS), "schemas"])
    assert main() == 0


def test_cli_rejects_incomplete_dataset(monkeypatch, tmp_path):
    empty_checklist = tmp_path / "artificial_empty_checklist.json"
    empty_checklist.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "fyp-validate", "--schema-dir", str(SCHEMAS), "checklist",
        str(empty_checklist),
        str(ROOT / "configs/normalization.json"),
    ])
    assert main() == 1


def test_pilot_checklist_ids_categories_and_schema():
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    assert [item["id"] for item in checklist] == [
        "C01", "C02", "C03", "C04", "C05", "C06", "K01", "K02", "T01", "T02",
    ]
    assert [item["type"] for item in checklist] == ["COMPLIANCE"] * 6 + ["CLIENT_KYC"] * 2 + ["TRANSACTION"] * 2
    assert {item["requirement_type"] for item in checklist[:6]} == {"OBLIGATION", "PROHIBITION"}
    assert validate(checklist, "checklist", SCHEMAS) == []


def test_pilot_extraction_fields_use_existing_exact_rules():
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    normalization = load_json(ROOT / "configs/normalization.json")
    assert {item["id"]: item["field"] for item in checklist[6:]} == {
        "K01": "client_identification_number", "K02": "dob",
        "T01": "transaction_instrument", "T02": "transaction_currency",
    }
    assert {item["field"]: normalization["field_rules"][item["field"]] for item in checklist[6:]} == {
        "client_identification_number": "exact_text", "dob": "exact_dob_dd_month_yyyy",
        "transaction_instrument": "exact_text", "transaction_currency": "exact_text",
    }
    assert validate_checklist(checklist, normalization, SCHEMAS) == []


@pytest.mark.parametrize("month", ENGLISH_MONTHS)
def test_canonical_dob_uses_full_english_month_names(month):
    assert is_canonical_dob(f"04 {month} 1991")


@pytest.mark.parametrize("name", ["call_config", "check_plan", "ground_truth"])
@pytest.mark.parametrize("value", ["04 August 1991", "29 February 2000"])
def test_new_dob_format_in_all_gold_contracts(name, value):
    data = documents()[name]
    key = "intended_answers" if name == "call_config" else "checks"
    data[key][0]["expected"] = {
        "kind": "extraction", "value_type": "dob_dd_month_yyyy", "expected_value": value,
    }
    assert validate(data, name, SCHEMAS) == []


@pytest.mark.parametrize("value", [
    "13052001", "13/05/2001", 13052001, None, True,
    "4 August 1991", "04 Aug 1991", "04 august 1991", "04 Août 1991",
    "04  August 1991", "04 August 91", "04 August 1991\n", " 04 August 1991",
    "31 April 1991", "29 February 1900", "29 February 2001", "00 May 2001",
    "01 January 0000", "13 May 10000", "０４ August 1991",
])
def test_dob_rejects_old_formats_and_invalid_dates(value):
    assert not is_canonical_dob(value)
    data = documents()["check_plan"]
    data["checks"][0]["expected"] = {
        "kind": "extraction", "value_type": "dob_dd_month_yyyy", "expected_value": value,
    }
    assert validate(data, "check_plan", SCHEMAS)


def test_legacy_dob_tag_is_rejected():
    data = documents()["check_plan"]
    data["checks"][0]["expected"] = {
        "kind": "extraction", "value_type": "dob_ddmmyyyy", "expected_value": "13052001",
    }
    assert validate(data, "check_plan", SCHEMAS)


@pytest.mark.parametrize("name", ["call_config", "check_plan", "ground_truth", "normalization"])
def test_old_contract_versions_require_explicit_migration(name):
    data = documents()[name]
    data["schema_version"] = "0.2.0" if name == "normalization" else "0.1.0"
    assert validate(data, name, SCHEMAS)


def test_pilot_langgraph_projection_is_lossless_and_read_only():
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    before = deepcopy(checklist)
    projected = to_langgraph_checklist(checklist, SCHEMAS)
    assert projected == [{"id": item["id"], "question": item["text"]} for item in checklist]
    assert checklist == before


@pytest.mark.parametrize("requirement_type", ["OBLIGATION", "PROHIBITION"])
def test_compliance_requirement_types(requirement_type):
    data = documents()["checklist"]
    data[0]["requirement_type"] = requirement_type
    assert validate(data, "checklist", SCHEMAS) == []


@pytest.mark.parametrize("item", [
    {"id": "FIX_OLD", "question": "Artificial legacy question?"},
    {"id": "FIX_BAD", "type": "COMPLIANCE", "text": "Artificial missing requirement type?"},
    {"id": "FIX_BAD", "type": "CLIENT_KYC", "text": "Artificial missing field?"},
    {"id": "FIX_BAD", "type": "TRANSACTION", "text": "Artificial missing field?"},
    {"id": "FIX_BAD", "type": "COMPLIANCE", "requirement_type": "OPTIONAL", "text": "Artificial?"},
    {"id": "FIX_BAD", "type": "OTHER", "field": "fixture", "text": "Artificial?"},
    {"id": "FIX_BAD", "type": "TRANSACTION", "field": " ", "text": "Artificial?"},
    {"id": "FIX_BAD", "type": "CLIENT_KYC", "field": "dob", "text": " "},
])
def test_invalid_checklist_shapes(item):
    assert validate([item], "checklist", SCHEMAS)


@pytest.mark.parametrize("category", ["COMPLIANCE", "CLIENT_KYC", "TRANSACTION"])
@pytest.mark.parametrize("extra", [
    {"answer_type": "text"}, {"question": "Old format"}, {"expected_value": "fixture"},
    {"gold_evidence_turns": [0]}, {"source_passage": "fixture"}, {"call_id": "CALL_999"},
    {"depends_on": []}, {"turn_id": 0},
])
def test_checklist_rejects_out_of_scope_fields(category, extra):
    item = {"id": "FIX_BAD", "type": category, "text": "Artificial fixture?"}
    item.update({"requirement_type": "OBLIGATION"} if category == "COMPLIANCE" else {"field": "fixture"})
    item.update(extra)
    assert validate([item], "checklist", SCHEMAS)


@pytest.mark.parametrize("category", ["COMPLIANCE", "CLIENT_KYC", "TRANSACTION"])
def test_reject_incompatible_field_combinations(category):
    item = {"id": "FIX_BAD", "type": category, "text": "Artificial fixture?",
            "field": "fixture", "requirement_type": "OBLIGATION"}
    assert validate([item], "checklist", SCHEMAS)


def test_langgraph_projection_preserves_ids_order_and_text():
    checklist = mixed_documents()["checklist"]
    before = deepcopy(checklist)
    projected = to_langgraph_checklist(checklist, SCHEMAS)
    assert projected == [{"id": item["id"], "question": item["text"]} for item in checklist]
    assert all(set(item) == {"id", "question"} for item in projected)
    assert checklist == before
    projected[0]["question"] = "changed projection"
    assert checklist == before


def test_langgraph_projection_rejects_legacy_and_duplicate_input():
    with pytest.raises(ValueError):
        to_langgraph_checklist([{"id": "FIX_OLD", "question": "Artificial?"}], SCHEMAS)
    checklist = documents()["checklist"]
    with pytest.raises(ValueError, match="Duplicate id"):
        to_langgraph_checklist(checklist * 2, SCHEMAS)


@pytest.mark.parametrize("name", ["call_config", "check_plan", "ground_truth"])
def test_mixed_category_answers_retain_values(name):
    data = mixed_documents()
    before = deepcopy(data)
    assert validate_answers(data[name], name, data["checklist"], data["normalization"], SCHEMAS) == []
    key = "intended_answers" if name == "call_config" else "checks"
    answers = [item["expected"] for item in data[name][key]]
    assert answers == [item["expected"] for item in data["call_config"]["intended_answers"]]
    assert answers[1]["expected_value"] == "01 January 2000"
    assert answers[2]["expected_value"] == 7
    assert answers[3]["expected_value"] == "artificial label"
    assert data == before


@pytest.mark.parametrize("name", ["call_config", "check_plan", "ground_truth"])
@pytest.mark.parametrize("case", ["missing", "extra", "extraction_as_verdict", "compliance_as_value", "wrong_value_type", "duplicate"])
def test_answers_must_match_checklist(name, case):
    data = mixed_documents()
    items = data[name]["intended_answers" if name == "call_config" else "checks"]
    if case == "missing":
        items.pop()
    elif case == "extra":
        items[-1]["question_id"] = "FIX_UNKNOWN"
    elif case == "extraction_as_verdict":
        items[1]["expected"] = {"kind": "compliance", "expected_verdict": "PASS"}
    elif case == "compliance_as_value":
        items[0]["expected"] = {"kind": "extraction", "value_type": "text", "expected_value": "fixture"}
    elif case == "wrong_value_type":
        items[1]["expected"] = {"kind": "extraction", "value_type": "text", "expected_value": "01 January 2000"}
    else:
        items.append(deepcopy(items[0]))
    assert validate_answers(data[name], name, data["checklist"], data["normalization"], SCHEMAS)


def test_normalization_reuse_is_by_field_not_question_id():
    data = mixed_documents()
    data["checklist"].append({**data["checklist"][1], "id": "FIX_ANOTHER_DOB"})
    assert validate_checklist(data["checklist"], data["normalization"], SCHEMAS) == []


def test_legacy_normalization_classification_is_rejected():
    normalization = documents()["normalization"]
    rule = normalization["rules"]["fixture_exact"]
    rule["answer_type"] = rule.pop("value_type")
    assert validate(normalization, "normalization", SCHEMAS)


@pytest.mark.parametrize("command", ["checklist", "answers"])
def test_cli_accepts_artificial_mixed_checklist(tmp_path, monkeypatch, command):
    data = mixed_documents()
    paths = {}
    for name in ["checklist", "normalization", "check_plan"]:
        paths[name] = tmp_path / f"artificial_{name}.json"
        paths[name].write_text(json.dumps(data[name]), encoding="utf-8")
    arguments = ["fyp-validate", "--schema-dir", str(SCHEMAS), command]
    if command == "answers":
        arguments += ["check_plan", str(paths["check_plan"])]
    arguments += [str(paths["checklist"]), str(paths["normalization"])]
    monkeypatch.setattr("sys.argv", arguments)
    assert main() == 0
