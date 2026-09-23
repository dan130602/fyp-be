"""ARTIFICIAL fixtures only. Writer tests use pytest temporary directories.

Never generates a check plan from the real pilot matrix or writes authoring/.
"""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from fixtures import documents
from fyp_evaluation.authoring.call_configs import build_call_config
from fyp_evaluation.authoring.check_plans import (
    build_check_plan, derive_evidence_mode, write_check_plans,
)
from fyp_evaluation.validation import load_json, validate
from fyp_evaluation.validation.check_plan import validate_check_plan, validate_forbidden_patterns
from fyp_evaluation.validation.__main__ import main

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


@pytest.fixture
def inputs():
    checklist = [
        {"id": qid, "type": "COMPLIANCE", "requirement_type": requirement, "text": "ARTIFICIAL check"}
        for qid, requirement in [("FIX_OP", "OBLIGATION"), ("FIX_OF", "OBLIGATION"),
                                  ("FIX_PP", "PROHIBITION"), ("FIX_PF", "PROHIBITION")]
    ] + [
        {"id": "K01", "type": "CLIENT_KYC", "field": "client_identification_number", "text": "ARTIFICIAL identifier"},
        {"id": "FIX_NUMBER", "type": "TRANSACTION", "field": "fixture_number", "text": "ARTIFICIAL number"},
    ]
    normalization = documents()["normalization"]
    normalization["rules"].update(fixture_text={"operation": "exact", "value_type": "text"},
                                  fixture_integer={"operation": "exact", "value_type": "integer"})
    normalization["field_rules"].update(client_identification_number="fixture_text", fixture_number="fixture_integer")
    matrix = {"CALL_999": {"FIX_OP": "PASS", "FIX_OF": "FAIL", "FIX_PP": "PASS", "FIX_PF": "FAIL",
                            "K01": "NOT_STATED", "FIX_NUMBER": "7"}}
    matrix["CALL_998"] = {**matrix["CALL_999"], "K01": "S0000000X"}
    configs = {call_id: build_call_config(call_id, checklist, normalization, matrix) for call_id in matrix}
    return checklist, normalization, matrix, configs


def validate_plan(plan, inputs, *, complete=False):
    checklist, normalization, matrix, configs = inputs
    return validate_check_plan(plan, configs["CALL_999"], checklist, normalization, matrix, SCHEMAS, complete=complete)


def plan_for(inputs):
    return build_check_plan("CALL_999", *inputs[:3])


def presence(qid):
    return {"mode": "presence", "planned_evidence": [{"evidence_id": f"{qid}_E1",
            "required_speaker": "EMPLOYEE", "text": "ARTIFICIAL marker"}],
            "content_constraints": [], "forbidden_patterns": []}


def authored(plan):
    plan = deepcopy(plan)
    for check in plan["checks"]:
        evidence = check["evidence_plan"]
        if evidence["mode"] == "presence":
            for phrase in evidence["planned_evidence"]:
                phrase["text"] = "ARTIFICIAL authored marker"
        else:
            evidence["absent_event"] = "ARTIFICIAL event must be absent"
    return plan


@pytest.mark.parametrize("category,requirement,verdict,value,mode", [
    ("COMPLIANCE", "OBLIGATION", "PASS", None, "presence"),
    ("COMPLIANCE", "OBLIGATION", "FAIL", None, "absence"),
    ("COMPLIANCE", "PROHIBITION", "PASS", None, "absence"),
    ("COMPLIANCE", "PROHIBITION", "FAIL", None, "presence"),
    ("CLIENT_KYC", None, None, "ARTIFICIAL", "presence"),
    ("CLIENT_KYC", None, None, None, "absence"),
    ("TRANSACTION", None, None, "ARTIFICIAL", "presence"),
    ("TRANSACTION", None, None, None, None),
])
def test_all_eight_derivation_combinations(category, requirement, verdict, value, mode):
    item = {"id": "K01" if category == "CLIENT_KYC" and value is None else "FIX_TEST", "type": category}
    if category == "COMPLIANCE":
        item["requirement_type"] = requirement
        expected = {"kind": "compliance", "expected_verdict": verdict}
    else:
        item["field"] = "client_identification_number" if item["id"] == "K01" else "fixture"
        expected = {"kind": "extraction", "value_type": "text", "expected_value": value}
    before = deepcopy((item, expected))
    if mode is None:
        with pytest.raises(ValueError):
            derive_evidence_mode(item, expected)
    else:
        assert derive_evidence_mode(item, expected) == mode
    assert (item, expected) == before


@pytest.mark.parametrize("expected", [
    {"kind": "compliance", "expected_verdict": "N/A"},
    {"kind": "compliance", "expected_verdict": "UNCLEAR"},
    {"kind": "extraction", "value_type": "text", "expected_value": "PASS"}, {}, None,
])
def test_unsupported_compliance_answers_raise(expected):
    with pytest.raises(ValueError):
        derive_evidence_mode({"id": "FIX", "type": "COMPLIANCE", "requirement_type": "OBLIGATION"}, expected)


@pytest.mark.parametrize("item,expected", [
    ({"type": "OTHER"}, {}),
    ({"type": "COMPLIANCE", "requirement_type": "OPTIONAL"}, {"kind": "compliance", "expected_verdict": "PASS"}),
    ({"id": "FIX", "type": "CLIENT_KYC"}, {"kind": "extraction", "value_type": "text", "expected_value": None}),
    ({"id": "K01", "type": "TRANSACTION", "field": "client_identification_number"}, {"kind": "extraction", "value_type": "text", "expected_value": None}),
    ({"id": "FIX", "type": "TRANSACTION"}, {"kind": "extraction", "value_type": "integer", "expected_value": True}),
    ({"id": "FIX", "type": "TRANSACTION"}, {"kind": "extraction", "value_type": "text"}),
])
def test_other_unsupported_combinations_raise(item, expected):
    with pytest.raises(ValueError):
        derive_evidence_mode(item, expected)


@pytest.mark.parametrize("value_type,value", [("integer", 0), ("number", 0.0), ("boolean", False)])
def test_falsey_values_still_have_presence(value_type, value):
    assert derive_evidence_mode({"id": "FIX", "type": "TRANSACTION"},
                                {"kind": "extraction", "value_type": value_type, "expected_value": value}) == "presence"


def test_scaffold_constants_modes_and_completeness(inputs):
    before = deepcopy(inputs)
    plan = plan_for(inputs)
    assert plan == plan_for(inputs)
    assert inputs == before
    assert plan["schema_version"] == "0.6.0"
    assert plan["ordering_constraints"] == []
    assert [c["evidence_plan"]["mode"] for c in plan["checks"]] == ["presence", "absence", "absence", "presence", "absence", "presence"]
    for check in plan["checks"]:
        evidence = check["evidence_plan"]
        assert evidence["content_constraints"] == evidence["forbidden_patterns"] == []
        if evidence["mode"] == "absence":
            assert evidence["scope"] == "entire_dialogue"
            assert evidence["required_speaker"] == ("ANY" if check["question_id"] == "K01" else "EMPLOYEE")
            assert evidence["absent_event"] == "TODO_AUTHOR_ABSENT_EVENT"
        else:
            assert "required_speaker" not in evidence
            assert len(evidence["planned_evidence"]) == 1
            phrase = evidence["planned_evidence"][0]
            assert phrase["evidence_id"] == check["question_id"] + "_E1"
            assert phrase["text"] == "TODO_AUTHOR_PLANNED_EVIDENCE"
            assert phrase["required_speaker"] == ("CLIENT" if check["question_id"] == "FIX_NUMBER" else "EMPLOYEE")
    assert validate_plan(plan, inputs) == []
    assert validate_plan(plan, inputs, complete=True)
    assert validate_plan(authored(plan), inputs, complete=True) == []


def test_obligation_fail_with_presence_rejected_but_prohibition_fail_accepted(inputs):
    plan = plan_for(inputs)
    assert plan["checks"][3]["evidence_plan"]["mode"] == "presence"
    assert validate_plan(plan, inputs) == []
    plan["checks"][1]["evidence_plan"] = presence("FIX_OF")
    assert validate(plan, "check_plan", SCHEMAS) == []  # representation alone cannot derive mode
    assert any("derived as absence" in error for error in validate_plan(plan, inputs))


@pytest.mark.parametrize("case", ["plan_answer", "config_answer", "both_answers", "number_type", "reorder", "wrong_call", "unknown_call", "missing", "duplicate"])
def test_cross_file_disagreements_rejected(inputs, case):
    plan = plan_for(inputs)
    if case in {"plan_answer", "both_answers"}:
        plan["checks"][0]["expected"]["expected_verdict"] = "FAIL"
    if case in {"config_answer", "both_answers"}:
        inputs[3]["CALL_999"]["intended_answers"][0]["expected"]["expected_verdict"] = "FAIL"
    if case == "number_type":
        plan["checks"][-1]["expected"]["expected_value"] = 7.0
    elif case == "reorder":
        plan["checks"].reverse()
    elif case == "wrong_call":
        plan["call_id"] = "CALL_998"
    elif case == "unknown_call":
        plan["call_id"] = "CALL_997"
    elif case == "missing":
        plan["checks"].pop()
    elif case == "duplicate":
        plan["checks"].append(deepcopy(plan["checks"][0]))
    assert validate_plan(plan, inputs)


@pytest.mark.parametrize("complete", [False, True])
def test_call_identity_fails_immediately_and_names_both_ids(inputs, complete):
    plan = plan_for(inputs)
    config = inputs[3]["CALL_998"]
    assert validate(plan, "check_plan", SCHEMAS) == []
    assert validate(config, "call_config", SCHEMAS) == []
    before = deepcopy((plan, config))
    # A missing schema directory proves the ID gate precedes schema work.
    errors = validate_check_plan(plan, config, *inputs[:3], "NONEXISTENT_SCHEMA_DIRECTORY", complete=complete)
    assert errors == ["check_plan.call_id 'CALL_999' must equal call_config.call_id 'CALL_998'"]
    assert (plan, config) == before


def test_two_item_prompt_response_evidence_accepted(inputs):
    plan = plan_for(inputs)
    plan["checks"][0]["evidence_plan"]["planned_evidence"].append({
        "evidence_id": "FIX_OP_E2", "required_speaker": "CLIENT", "text": "ARTIFICIAL client response",
    })
    assert validate_plan(plan, inputs) == []


@pytest.mark.parametrize("bad_id", ["FIX_OP_E1", "FIX_OP_E3", "FIX_OP_E0", "FIX_OP_E02", "FIX_PF_E1", "OTHER_E2"])
def test_noncontiguous_or_duplicate_ids_rejected(inputs, bad_id):
    plan = plan_for(inputs)
    plan["checks"][0]["evidence_plan"]["planned_evidence"].append({
        "evidence_id": bad_id, "required_speaker": "CLIENT", "text": "ARTIFICIAL second marker",
    })
    assert validate_plan(plan, inputs)


@pytest.mark.parametrize("pattern", ["[", "(?", "(?<=a+)b", r"(?P<bad-name>x)"])
def test_invalid_python_regex_rejected(inputs, pattern):
    plan = plan_for(inputs)
    plan["checks"][0]["evidence_plan"]["forbidden_patterns"] = [pattern]
    assert any("invalid Python regex" in error for error in validate_plan(plan, inputs))


def test_empty_and_python_specific_patterns_accepted(inputs):
    plan = plan_for(inputs)
    assert validate_plan(plan, inputs) == []
    plan["checks"][1]["evidence_plan"]["forbidden_patterns"] = [r"(?P<marker>ARTIFICIAL)\s+token"]
    assert validate_plan(plan, inputs) == []


@pytest.mark.parametrize("edges", [
    [("FIX_OP", "FIX_OP")], [("FIX_OP", "FIX_PF"), ("FIX_PF", "FIX_OP")],
    [("FIX_OP", "FIX_PF"), ("FIX_PF", "FIX_NUMBER"), ("FIX_NUMBER", "FIX_OP")],
    [("FIX_OP", "FIX_OF")], [("K01", "FIX_OP")], [("UNKNOWN", "FIX_OP")],
])
def test_bad_ordering_constraints_rejected(inputs, edges):
    plan = plan_for(inputs)
    plan["ordering_constraints"] = [{"before": a, "after": b} for a, b in edges]
    assert validate_plan(plan, inputs)


def test_presence_ordering_dag_accepted(inputs):
    plan = plan_for(inputs)
    plan["ordering_constraints"] = [{"before": "FIX_OP", "after": "FIX_PF"},
                                     {"before": "FIX_PF", "after": "FIX_NUMBER"}]
    assert validate_plan(plan, inputs) == []


@pytest.mark.parametrize("field", ["planned_evidence", "absent_event", "content_constraints", "forbidden_patterns"])
@pytest.mark.parametrize("marker", ["TODO", "tbd", "FIXME", "PLACEHOLDER"])
def test_complete_checks_all_authored_text(inputs, field, marker):
    plan = authored(plan_for(inputs))
    evidence = plan["checks"][1 if field == "absent_event" else 0]["evidence_plan"]
    if field == "planned_evidence":
        evidence[field][0]["text"] = f"ARTIFICIAL {marker} text"
    elif field == "absent_event":
        evidence[field] = f"ARTIFICIAL {marker} event"
    else:
        evidence[field] = [f"ARTIFICIAL {marker} constraint"]
    assert validate_plan(plan, inputs) == []
    assert any("placeholder" in error for error in validate_plan(plan, inputs, complete=True))


def test_required_fields_roles_and_critical_tokens(inputs):
    plan = plan_for(inputs)
    for field in ["forbidden_patterns", "content_constraints"]:
        bad = deepcopy(plan)
        del bad["checks"][0]["evidence_plan"][field]
        assert validate_plan(bad, inputs)
    for field in ["critical_tokens", "turn_id", "channel_id"]:
        bad = deepcopy(plan)
        bad["checks"][0]["evidence_plan"]["planned_evidence"][0][field] = 0
        assert validate_plan(bad, inputs)
    for index, role in [(1, "CLIENT"), (4, "EMPLOYEE"), (5, "EMPLOYEE")]:
        bad = deepcopy(plan)
        evidence = bad["checks"][index]["evidence_plan"]
        if evidence["mode"] == "absence":
            evidence["required_speaker"] = role
        else:
            evidence["planned_evidence"][0]["required_speaker"] = role
        assert validate_plan(bad, inputs)
    nonnull = build_check_plan("CALL_998", *inputs[:3])
    assert nonnull["checks"][4]["evidence_plan"]["planned_evidence"][0]["required_speaker"] == "CLIENT"


def test_ground_truth_inherits_patterns_without_other_shape_changes():
    data = documents()["ground_truth"]
    assert data["schema_version"] == "0.6.0"
    assert validate(data, "ground_truth", SCHEMAS) == []
    assert validate_forbidden_patterns(data) == []
    data["checks"][0]["evidence_plan"]["forbidden_patterns"] = ["["]
    assert validate_forbidden_patterns(data)
    del data["checks"][0]["evidence_plan"]["forbidden_patterns"]
    assert validate(data, "ground_truth", SCHEMAS)


def test_writer_only_preflight_and_overwrite(inputs, tmp_path):
    selected = write_check_plans(tmp_path, *inputs, SCHEMAS, only=["CALL_998"])
    assert [p.name for p in selected] == ["CALL_998.json"]
    assert not (tmp_path / "CALL_999.json").exists()
    before = selected[0].read_bytes()
    assert b"\r\n" not in before
    with pytest.raises(FileExistsError):
        write_check_plans(tmp_path, *inputs, SCHEMAS)
    assert not (tmp_path / "CALL_999.json").exists()
    assert selected[0].read_bytes() == before
    write_check_plans(tmp_path, *inputs, SCHEMAS, only=["CALL_998"], overwrite=True)
    assert selected[0].read_bytes() == before


@pytest.mark.parametrize("selection", [[], ["CALL_997"], ["CALL_999", "CALL_999"]])
def test_only_rejects_invalid_selection(inputs, tmp_path, selection):
    with pytest.raises(ValueError):
        write_check_plans(tmp_path, *inputs, SCHEMAS, only=selection)
    assert not list(tmp_path.iterdir())


def test_only_does_not_skip_validation_of_other_columns(inputs, tmp_path):
    inputs[2]["CALL_999"]["FIX_NUMBER"] = "invalid scalar"
    with pytest.raises(ValueError):
        write_check_plans(tmp_path, *inputs, SCHEMAS, only=["CALL_998"])
    assert not list(tmp_path.iterdir())


def test_cli_only_keeps_full_matrix_and_checks_config(inputs, tmp_path, monkeypatch):
    checklist, normalization, matrix, configs = inputs
    authoring = tmp_path / "authoring"
    config_dir = authoring / "call_configs"
    config_dir.mkdir(parents=True)
    for call_id, config in configs.items():
        (config_dir / f"{call_id}.json").write_text(json.dumps(config), encoding="utf-8")
    matrix_path = authoring / "matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question_id", *matrix])
        for item in checklist:
            writer.writerow([item["id"], *(cells[item["id"]] for cells in matrix.values())])
    checklist_path, normalization_path = tmp_path / "checklist.json", tmp_path / "normalization.json"
    checklist_path.write_text(json.dumps(checklist), encoding="utf-8")
    normalization_path.write_text(json.dumps(normalization), encoding="utf-8")
    output = tmp_path / "test_plans"
    trailing = [str(checklist_path), str(normalization_path), "--call-ids", *matrix]

    def run(args):
        monkeypatch.setattr("sys.argv", ["fyp-validate", "--schema-dir", str(SCHEMAS), *args])
        return main()

    command = ["scaffold-check-plans", str(matrix_path), str(output), *trailing, "--only", "CALL_999"]
    assert run(command) == 0
    assert sorted(p.name for p in output.iterdir()) == ["CALL_999.json"]
    assert run(command) == 2
    path = output / "CALL_999.json"
    check = ["check-plan", str(path), str(config_dir / "CALL_999.json"), str(matrix_path), *trailing]
    assert run(check) == 0
    assert run([*check, "--complete"]) == 1
    path.write_text(json.dumps(authored(load_json(path))), encoding="utf-8")
    assert run([*check, "--complete"]) == 0
    assert run([*command, "--overwrite"]) == 0
    # --only cannot hide a missing approved CSV column.
    reduced = matrix_path.read_text().replace(",CALL_998", "")
    matrix_path.write_text(reduced, encoding="utf-8")
    assert run([*command, "--overwrite"]) == 2
