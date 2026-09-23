"""Stage 3A: artificial authoring fixtures and read-only checks of pilot inputs."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from fixtures import mixed_documents
from fyp_evaluation.authoring.call_configs import (
    SCENARIO_INTENT_PLACEHOLDER, build_call_config, write_call_configs,
)
from fyp_evaluation.authoring.coverage import compile_expected, load_matrix
from fyp_evaluation.validation import load_json
from fyp_evaluation.validation.__main__ import main
from fyp_evaluation.validation.call_config import validate_call_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


@pytest.fixture
def artificial():
    data = mixed_documents()
    checklist, normalization = data["checklist"], data["normalization"]
    matrix = {"CALL_999": {
        "FIX_MARKER": "FAIL", "FIX_DOB": "01 January 2000",
        "FIX_QUANTITY": "7", "FIX_LABEL": "artificial label",
    }}
    return checklist, normalization, matrix


def write_matrix(path, checklist, matrix):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question_id", *matrix])
        for item in checklist:
            writer.writerow([item["id"], *(cells[item["id"]] for cells in matrix.values())])


def test_loader_preserves_quoted_text_dates_and_row_identity(tmp_path, artificial):
    checklist, _, matrix = artificial
    matrix["CALL_999"]["FIX_LABEL"] = '  artificial, "label"\nsecond line  '
    path = tmp_path / "matrix.csv"
    write_matrix(path, list(reversed(checklist)), matrix)
    loaded = load_matrix(path, checklist=checklist, call_ids=["CALL_999"])
    assert loaded == matrix


@pytest.mark.parametrize("raw", [
    "", "question_id\nFIX_MARKER\n", "id,CALL_999\nFIX_MARKER,PASS\n",
    "question_id,CALL_999,CALL_999\nFIX_MARKER,PASS,PASS\n",
    "question_id,\nFIX_MARKER,PASS\n",
    "question_id,CALL_998\nFIX_MARKER,PASS\n",
    "question_id,CALL_999,CALL_998\nFIX_MARKER,PASS,PASS\n",
    "question_id,CALL_999\nFIX_MARKER\n",
    "question_id,CALL_999\nFIX_MARKER,PASS,extra\n",
    "question_id,CALL_999\nFIX_MARKER,\n",
    "question_id,CALL_999\nFIX_MARKER,   \n",
    "question_id,CALL_999\nFIX_MARKER,PASS\nFIX_MARKER,FAIL\n",
    "question_id,CALL_999\nUNKNOWN,PASS\n",
    "question_id,CALL_999\n,PASS\n",
    "question_id,CALL_999\n",
    'question_id,CALL_999\nFIX_MARKER,"unterminated\n',
    'question_id,CALL_999\nFIX_MARKER,"PASS"junk\n',
    "question_id, CALL_999\nFIX_MARKER,PASS\n",
    "question_id,CALL_999\n\nFIX_MARKER,PASS\n",
    "question_id,CALL_999\nFIX_MARKER,PA\x00SS\n",
])
def test_malformed_matrix_is_rejected(tmp_path, artificial, raw):
    checklist, _, _ = artificial
    path = tmp_path / "bad.csv"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        load_matrix(path, checklist=checklist[:1], call_ids=["CALL_999"])


def test_loader_rejects_reordered_approved_call_headers(tmp_path, artificial):
    checklist, _, matrix = artificial
    matrix["CALL_998"] = dict(matrix["CALL_999"])
    path = tmp_path / "matrix.csv"
    write_matrix(path, checklist, matrix)
    with pytest.raises(ValueError, match="call columns"):
        load_matrix(path, checklist=checklist, call_ids=["CALL_998", "CALL_999"])


@pytest.mark.parametrize("value_type,cell,value", [
    ("integer", "0", 0), ("integer", "-7", -7), ("number", "1.25", 1.25),
    ("number", "1e3", 1000.0), ("number", "1", 1),
    ("boolean", "false", False), ("boolean", "true", True),
    ("text", "  artificial label  ", "  artificial label  "),
    ("text", "PASS", "PASS"), ("dob_dd_month_yyyy", "04 August 1991", "04 August 1991"),
])
def test_compilation_preserves_scalar_value_and_type(artificial, value_type, cell, value):
    _, normalization, _ = artificial
    item = {"id": "FIX_VALUE", "type": "TRANSACTION", "field": "fixture", "question": "Artificial?"}
    normalization["field_rules"]["fixture"] = "rule"
    normalization["rules"]["rule"] = {"operation": "exact", "value_type": value_type}
    expected = compile_expected(item, normalization, cell)
    assert expected == {"kind": "extraction", "value_type": value_type, "expected_value": value}
    assert type(expected["expected_value"]) is type(value)


@pytest.mark.parametrize("value_type,cell", [
    ("integer", "01"), ("integer", "1.0"), ("integer", "true"), ("integer", " 1"),
    ("integer", "null"), ("number", "NaN"), ("number", "Infinity"),
    ("number", "1e999"), ("number", "true"), ("number", "[]"),
    ("boolean", "False"), ("boolean", "0"), ("boolean", "null"),
    ("dob_dd_month_yyyy", "29 February 2001"), ("dob_dd_month_yyyy", "04081991"),
    ("text", ""), ("text", "   "), ("integer", 7),
])
def test_compilation_never_repairs_invalid_cells(artificial, value_type, cell):
    _, normalization, _ = artificial
    normalization["field_rules"]["fixture"] = "rule"
    normalization["rules"]["rule"] = {"operation": "exact", "value_type": value_type}
    item = {"id": "FIX_VALUE", "type": "TRANSACTION", "field": "fixture"}
    with pytest.raises(ValueError):
        compile_expected(item, normalization, cell)


@pytest.mark.parametrize("cell", ["pass", " PASS", "FAIL\n", "N/A", "UNCLEAR", "NOT_STATED"])
def test_compliance_cells_are_exact_verdicts(artificial, cell):
    checklist, normalization, _ = artificial
    with pytest.raises(ValueError):
        compile_expected(checklist[0], normalization, cell)


def test_k01_decodes_only_its_approved_missing_value():
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    normalization = load_json(ROOT / "configs/normalization.json")
    item = next(item for item in checklist if item["id"] == "K01")
    expected = compile_expected(item, normalization, "NOT_STATED")
    assert expected == {"kind": "extraction", "value_type": "text", "expected_value": None}
    assert json.loads(json.dumps(expected))["expected_value"] is None
    for cell in ["null", "", "NOT_STATED ", "PASS"]:
        with pytest.raises(ValueError):
            compile_expected(item, normalization, cell)


def test_build_is_deterministic_and_does_not_mutate_inputs(artificial):
    checklist, normalization, matrix = artificial
    before = deepcopy(artificial)
    first = build_call_config("CALL_999", checklist, normalization, matrix)
    second = build_call_config("CALL_999", checklist, normalization, matrix)
    assert first == second
    assert first["metadata"] == {"language": "en-SG", "scenario": "private_banking_investment_advice_and_order"}
    assert first["target_duration_seconds"] == {"min": 120, "max": 180}
    assert first["scenario_facts"] == []
    assert [item["question_id"] for item in first["intended_answers"]] == [item["id"] for item in checklist]
    first["intended_answers"][0]["expected"]["expected_verdict"] = "PASS"
    assert second["intended_answers"][0]["expected"]["expected_verdict"] == "FAIL"
    assert artificial == before


def test_compiler_rejects_unknown_call_and_missing_matrix_question(artificial):
    checklist, normalization, matrix = artificial
    with pytest.raises(ValueError, match="no column"):
        build_call_config("CALL_998", checklist, normalization, matrix)
    del matrix["CALL_999"]["FIX_DOB"]
    with pytest.raises(ValueError, match="checklist IDs"):
        build_call_config("CALL_999", checklist, normalization, matrix)


def test_two_completeness_modes(artificial):
    checklist, normalization, matrix = artificial
    data = build_call_config("CALL_999", *artificial)
    args = (checklist, normalization, matrix, SCHEMAS)
    assert validate_call_config(data, *args, complete=False) == []
    errors = validate_call_config(data, *args, complete=True)
    assert len(errors) == 1 + len(checklist)
    assert all(error.startswith("Incomplete:") for error in errors)
    data["scenario_facts"] = [{"name": "fixture_context", "value": "ARTIFICIAL ONLY"}]
    for answer in data["intended_answers"]:
        answer["scenario_intent"] = "ARTIFICIAL authored fixture intent"
    assert validate_call_config(data, *args, complete=True) == []


@pytest.mark.parametrize("marker", ["TODO_AUTHOR_SCENARIO_INTENT", " todo ", "TBD", "FIXME: later", "placeholder", "PENDING", "N/A", "..."])
def test_complete_mode_rejects_reserved_placeholder_markers(artificial, marker):
    data = build_call_config("CALL_999", *artificial)
    data["scenario_facts"] = [{"name": "fixture_context", "value": True}]
    for answer in data["intended_answers"]:
        answer["scenario_intent"] = "ARTIFICIAL fixture intent"
    data["intended_answers"][0]["scenario_intent"] = marker
    errors = validate_call_config(data, *artificial, SCHEMAS, complete=True)
    assert len(errors) == 1 and "placeholder" in errors[0]


@pytest.mark.parametrize("case", [
    "verdict", "text", "date", "integer_float", "wrong_value_type", "duplicate", "missing",
    "unknown", "reordered", "call_id", "duration", "duplicate_fact", "extraction_fact",
])
def test_cross_file_validation_rejects_tampering(artificial, case):
    data = build_call_config("CALL_999", *artificial)
    answers = data["intended_answers"]
    if case == "verdict":
        answers[0]["expected"]["expected_verdict"] = "PASS"
    elif case == "text":
        answers[3]["expected"]["expected_value"] = "altered label"
    elif case == "date":
        answers[1]["expected"]["expected_value"] = "02 January 2000"
    elif case == "integer_float":
        answers[2]["expected"]["expected_value"] = 7.0
    elif case == "wrong_value_type":
        answers[2]["expected"].update(value_type="text", expected_value="7")
    elif case == "duplicate":
        answers.append(deepcopy(answers[0]))
    elif case == "missing":
        answers.pop()
    elif case == "unknown":
        answers[0]["question_id"] = "FIX_UNKNOWN"
    elif case == "reordered":
        answers.reverse()
    elif case == "call_id":
        data["call_id"] = "CALL_998"
    elif case == "duration":
        data["target_duration_seconds"] = {"min": 180, "max": 120}
    elif case == "duplicate_fact":
        data["scenario_facts"] = [{"name": "fixture", "value": 1}] * 2
    else:
        data["scenario_facts"] = [{"name": "dob", "value": "01 January 2000"}]
    assert validate_call_config(data, *artificial, SCHEMAS)


def test_matrix_changes_cannot_be_hidden_by_existing_config(artificial):
    data = build_call_config("CALL_999", *artificial)
    artificial[2]["CALL_999"]["FIX_QUANTITY"] = "8"
    assert any("differs from approved matrix" in error for error in validate_call_config(data, *artificial, SCHEMAS))


def test_writer_validates_batch_and_refuses_overwrite_before_any_write(tmp_path, artificial):
    checklist, normalization, matrix = artificial
    matrix["CALL_998"] = dict(matrix["CALL_999"])
    occupied = tmp_path / "CALL_998.json"
    occupied.write_text("existing artifact", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_call_configs(tmp_path, checklist, normalization, matrix, SCHEMAS)
    assert not (tmp_path / "CALL_999.json").exists()
    assert occupied.read_text() == "existing artifact"
    paths = write_call_configs(tmp_path, checklist, normalization, matrix, SCHEMAS, overwrite=True)
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        write_call_configs(tmp_path, checklist, normalization, matrix, SCHEMAS)
    assert {path: path.read_bytes() for path in paths} == before
    write_call_configs(tmp_path, checklist, normalization, matrix, SCHEMAS, overwrite=True)
    assert {path: path.read_bytes() for path in paths} == before


def test_invalid_later_call_does_not_write_earlier_calls(tmp_path, artificial):
    checklist, normalization, matrix = artificial
    matrix["CALL_998"] = {**matrix["CALL_999"], "FIX_QUANTITY": "not a number"}
    output = tmp_path / "uncreated"
    with pytest.raises(ValueError):
        write_call_configs(output, checklist, normalization, matrix, SCHEMAS)
    assert not output.exists()


def test_five_pilot_scaffolds_and_all_fifty_answers_in_memory():
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    normalization = load_json(ROOT / "configs/normalization.json")
    matrix = load_matrix(ROOT / "authoring/coverage_matrix.csv")
    assert len(matrix) == 5 and len(checklist) == 10
    for call_id, cells in matrix.items():
        data = build_call_config(call_id, checklist, normalization, matrix)
        assert validate_call_config(data, checklist, normalization, matrix, SCHEMAS) == []
        errors = validate_call_config(data, checklist, normalization, matrix, SCHEMAS, complete=True)
        assert len(errors) == 11 and all(error.startswith("Incomplete:") for error in errors)
        assert "NOT_STATED" not in json.dumps(data)
        for item, answer in zip(checklist, data["intended_answers"]):
            assert answer["scenario_intent"] == SCENARIO_INTENT_PLACEHOLDER
            expected = answer["expected"]
            if item["type"] == "COMPLIANCE":
                assert expected == {"kind": "compliance", "expected_verdict": cells[item["id"]]}
            else:
                value = expected["expected_value"]
                if item["id"] == "K01" and call_id == "CALL_005":
                    assert cells["K01"] == "NOT_STATED" and value is None
                else:
                    assert type(value) is str and value == cells[item["id"]]


@pytest.mark.parametrize("call_id,question_id,value", [
    ("CALL_001", "K01", None), ("CALL_005", "K01", "S0000000X"),
    ("CALL_005", "K01", "NOT_STATED"), ("CALL_005", "K01", "null"),
    ("CALL_005", "K02", None), ("CALL_005", "T01", None), ("CALL_005", "T02", None),
])
def test_pilot_absence_is_locked_to_matrix_cell(call_id, question_id, value):
    checklist = load_json(ROOT / "evaluation_dataset/checklist.json")
    normalization = load_json(ROOT / "configs/normalization.json")
    matrix = load_matrix(ROOT / "authoring/coverage_matrix.csv")
    data = build_call_config(call_id, checklist, normalization, matrix)
    next(answer for answer in data["intended_answers"] if answer["question_id"] == question_id)["expected"]["expected_value"] = value
    assert validate_call_config(data, checklist, normalization, matrix, SCHEMAS)


def test_cli_matrix_scaffold_validation_and_overwrite(tmp_path, artificial, monkeypatch, capsys):
    checklist, normalization, matrix = artificial
    csv_path = tmp_path / "matrix.csv"
    write_matrix(csv_path, checklist, matrix)
    checklist_path, rules_path = tmp_path / "checklist.json", tmp_path / "normalization.json"
    checklist_path.write_text(json.dumps(checklist), encoding="utf-8")
    rules_path.write_text(json.dumps(normalization), encoding="utf-8")
    shared = [str(checklist_path), str(rules_path), "--call-ids", "CALL_999"]
    prefix = ["fyp-validate", "--schema-dir", str(SCHEMAS)]

    def run(command):
        monkeypatch.setattr("sys.argv", prefix + command)
        return main()

    assert run(["matrix", str(csv_path), *shared]) == 0
    output = tmp_path / "configs"
    generate = ["scaffold-call-configs", str(csv_path), str(output), *shared]
    assert run(generate) == 0
    assert run(generate) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    config = output / "CALL_999.json"
    check = ["call-config", str(config), str(csv_path), *shared]
    assert run(check) == 0
    assert run([*check, "--complete"]) == 1
    assert run([*generate, "--overwrite"]) == 0
    data = load_json(config)
    data["intended_answers"][0]["expected"]["expected_verdict"] = "PASS"
    config.write_text(json.dumps(data), encoding="utf-8")
    assert run(check) == 1
