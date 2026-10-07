"""G1 deterministic planning: reproducibility, approved constraints and loud failures.

Uses the approved advisory_execution_pilot_v1 G0 inputs. No LLM, network or AWS.
"""

from copy import deepcopy
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from fyp_evaluation.planning import adapter, allocate, outputs
from fyp_evaluation.planning.allocate import PlanningError, build_plan
from fyp_evaluation.planning.cli import main
from fyp_evaluation.planning.outputs import (
    ASSIGNMENTS, MANIFEST, MATRIX, RECORD, check_outputs, generate, parse_assignments, parse_manifest,
    parse_matrix, serialize_plan, validate_written,
)
from fyp_evaluation.planning.validate import validate_plan
from fyp_evaluation.validation.schemas import load_json, validate_checklist
from fyp_evaluation.validation.stage0 import load_stage0, validate_stage0

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"


@pytest.fixture(scope="module")
def documents():
    return load_stage0(RELEASE)


@pytest.fixture(scope="module")
def generated():
    return generate(RELEASE, SCHEMAS)


@pytest.fixture()
def parsed(generated):
    texts, _ = generated
    return (parse_manifest(texts[MANIFEST]), parse_matrix(texts[MATRIX]), parse_assignments(texts[ASSIGNMENTS]))


def test_generation_is_byte_reproducible(generated):
    again, _ = generate(RELEASE, SCHEMAS)
    assert again == generated[0]


def test_seed_changes_the_plan(documents):
    other = deepcopy(documents)
    other["generation_plan"]["random_seed"] = 43
    assert serialize_plan(other, build_plan(other)) != serialize_plan(documents, build_plan(documents))


def test_written_outputs_reproduce_and_validate(generated):
    assert check_outputs(RELEASE, generated[0]) == []
    assert validate_written(RELEASE, SCHEMAS) == []
    assert main(["--dataset-root", str(RELEASE), "--check"]) == 0
    assert main(["--dataset-root", str(RELEASE), "--validate"]) == 0


def test_approved_counts(generated):
    summary = json.loads(generated[0][RECORD])["summary"]
    assert summary["calls"] == 50
    assert summary["duration_profiles"] == {"SHORT": 45, "LONG": 5}
    assert summary["call_composition"] == {
        "all_pass_without_edge_cases": 5, "all_pass_with_edge_cases": 5,
        "some_fail_without_edge_cases": 5, "some_fail_with_edge_cases": 35,
    }
    assert summary["fails_per_call"] == {"0": 10, "1": 10, "2": 15, "3": 10, "4": 5}
    assert summary["total_compliance_fail_pairs"] == 90
    assert all(v["FAIL"] >= 6 for v in summary["compliance_verdicts"].values())
    assert sum(v["FAIL"] for v in summary["compliance_verdicts"].values()) == 90
    assert summary["transaction_direction"] == {"BUY": 25, "SELL": 25}
    assert summary["order_type"] == {"MARKET": 25, "LIMIT": 25}
    assert summary["transaction_currency"] == {c: 10 for c in ("SGD", "USD", "EUR", "CNY", "JPY")}
    assert summary["edge_case_families"] == {
        "CORRECTION_SUPERSESSION": 18, "DISTRACTOR_COMPETING_VALUES": 16, "REFERENCE_RESOLUTION": 10,
        "SPOKEN_NUMERIC_NORMALISATION": 16, "DIALOGUE_STRUCTURE": 14, "COMPLIANCE_SEMANTICS": 16,
    }
    assert summary["edge_cases_per_call"] == {"0": 10, "1": 10, "2": 15, "3": 10, "4": 5}
    assert summary["total_edge_case_assignments"] == 90


def test_c03_fails_at_least_as_often_as_c02(generated):
    verdicts = json.loads(generated[0][RECORD])["summary"]["compliance_verdicts"]
    assert verdicts["C03"]["FAIL"] >= verdicts["C02"]["FAIL"] + 4  # RISK_EXPLAINED forces C03 FAIL with C02 PASS


def test_every_assignment_declares_exactly_its_catalog_parameters(documents, parsed):
    cases = {c["id"]: c for c in documents["edge_case_catalog"]["edge_cases"]}
    for a in parsed[2]:
        assert set(a["parameters"]) == set(cases[a["edge_case_id"]]["required_parameters"]), a["assignment_id"]
    assert not hasattr(__import__("fyp_evaluation.planning.edge_params", fromlist=["x"]), "EXTRA_PARAMETERS")


def test_multiple_corrections_never_target_binary_fields(parsed):
    targets = {q for a in parsed[2] if a["edge_case_id"] == "EC_MULTIPLE_CORRECTIONS" for q in a["target_question_ids"]}
    assert targets and not targets & {"T05", "T06"}


def test_no_fee_distractor_where_c04_fails(parsed):
    _, matrix, assignments = parsed
    for a in assignments:
        if matrix[a["call_id"]]["C04"] != "FAIL":
            continue
        if a["edge_case_id"] == "EC_MULTIPLE_CURRENCIES":
            assert a["parameters"]["distractor_role"] != "FEE_CURRENCY"
        if a["edge_case_id"] == "EC_MULTIPLE_MONETARY_VALUES":
            assert all("FEE" not in d["role"] for d in a["parameters"]["distractor_values"])


def test_distribution_deviations_are_rejected(documents, parsed):
    manifest, matrix, assignments = parsed
    call = next(c for c, cells in matrix.items() if cells["C01"] == "PASS" and cells["C09"] == "PASS" and cells["C10"] == "PASS")
    matrix[call]["C10"] = "FAIL"
    errors = validate_plan(documents, manifest, matrix, assignments, SCHEMAS)
    assert any("FAILs per call counts" in e for e in errors), errors


def test_call_intents_are_canonical(generated):
    rows = parse_manifest(generated[0][MANIFEST])
    assert {",".join(r["call_intents"]) for r in rows} == {"ADVISORY,EXECUTION"}
    assert "ADVISORY|EXECUTION" in generated[0][MANIFEST]


def test_adapter_reuses_legacy_tooling_without_legacy_config(documents):
    shaped = adapter.legacy_normalization(documents["checklist"], documents["check_semantics"], documents["normalization"])
    assert validate_checklist(documents["checklist"], shaped, SCHEMAS) == []
    legacy = load_json(ROOT / "configs" / "normalization.json")
    assert "transaction_amount" not in legacy["field_rules"]


def _first(assignments, case_id):
    return next(a for a in assignments if a["edge_case_id"] == case_id)


def _expect(documents, parsed_outputs, message):
    errors = validate_plan(documents, *parsed_outputs, SCHEMAS)
    assert any(message in e for e in errors), errors


def test_final_value_must_equal_matrix_gold(documents, parsed):
    manifest, matrix, assignments = parsed
    a = _first(assignments, "EC_CORRECTION_CROSS_TURN")
    a["parameters"]["final_value"] = a["parameters"]["initial_value"]
    _expect(documents, parsed, "differs from coverage-matrix gold")


def test_superseded_value_must_differ(documents, parsed):
    a = _first(parsed[2], "EC_CORRECTION_SAME_TURN")
    a["parameters"]["initial_value"] = a["parameters"]["final_value"]
    _expect(documents, parsed, "must be distinct, differ from gold and stay realistic")


def test_initial_order_must_be_plausible(documents, parsed):
    a = _first(parsed[2], "EC_COMPLETE_ORDER_REPLACEMENT")
    initial = a["parameters"]["initial_order"]
    initial["T04"] = 1
    _expect(documents, parsed, "initial_order is not a plausible configured order")


def test_one_edge_case_per_question(documents, parsed):
    assignments = parsed[2]
    per_call = Counter(a["call_id"] for a in assignments if a["edge_case_id"] != "EC_LONG_DISTANCE_DEPENDENCY")
    call = next(c for c, n in per_call.items() if n >= 2)
    first, second = [a for a in assignments if a["call_id"] == call and a["edge_case_id"] != "EC_LONG_DISTANCE_DEPENDENCY"][:2]
    second["target_question_ids"] = list(first["target_question_ids"])
    _expect(documents, parsed, "more than one edge case targets")


def test_fee_distractor_forbidden_when_c04_fails(documents, parsed):
    _, matrix, assignments = parsed
    a = _first(assignments, "EC_MULTIPLE_CURRENCIES")
    a["parameters"]["distractor_role"] = "FEE_CURRENCY"
    matrix[a["call_id"]]["C04"] = "FAIL"
    _expect(documents, parsed, "fee distractor is not allowed when C04 is FAIL")


def test_c03_pass_requires_c02_pass(documents, parsed):
    _, matrix, _ = parsed
    call = next(c for c, cells in matrix.items() if cells["C02"] == "PASS" and cells["C03"] == "PASS")
    matrix[call]["C02"] = "FAIL"
    _expect(documents, parsed, "violates C03_PASS_REQUIRES_C02_PASS")


def test_product_currency_compatibility(documents, parsed):
    _, matrix, _ = parsed
    call = next(c for c, cells in matrix.items() if cells["T01"] == "Harbor SGD Bond Fund")
    matrix[call]["T02"] = "JPY"
    _expect(documents, parsed, "not a plausible configured order")


def test_unique_k01(documents, parsed):
    _, matrix, _ = parsed
    matrix["CALL_002"]["K01"] = matrix["CALL_001"]["K01"]
    _expect(documents, parsed, "K01 values must be unique")


def test_intent_order_is_canonical(documents, parsed):
    parsed[0][0]["call_intents"] = ["EXECUTION", "ADVISORY"]
    _expect(documents, parsed, "vocabulary order")


def test_mutually_exclusive_readbacks(documents, parsed):
    a = _first(parsed[2], "EC_EMPLOYEE_READBACK_ERROR")
    b = deepcopy(_first(parsed[2], "EC_ONE_FIELD_READBACK_ERROR"))
    b["call_id"] = a["call_id"]
    parsed[2].append(b)
    _expect(documents, parsed, "mutually exclusive")


@pytest.mark.parametrize("mutate,message", [
    (lambda t: t.replace('""final_value"":', '""final_value"": ', 1), "canonical JSON"),
    (lambda t: t.replace("assignment_id", "assignment", 1), "header must be exactly"),
])
def test_strict_assignment_loader(generated, mutate, message):
    with pytest.raises(ValueError, match=message):
        parse_assignments(mutate(generated[0][ASSIGNMENTS]))


def test_strict_manifest_loader(generated):
    with pytest.raises(ValueError, match="canonical positive integer"):
        parse_manifest(generated[0][MANIFEST].replace(",180,300", ",0180,300", 1))


def test_infeasible_plan_fails_loudly(documents):
    """G0-valid but unplaceable: 50 currency/monetary distractors, yet only 40 calls hold edge cases."""
    impossible = deepcopy(documents)
    distribution = impossible["generation_plan"]["edge_case_distribution"]
    distribution["edge_cases_per_call_counts"] = {"0": 10, "4": 40}
    distribution["family_quotas"] = {
        "CORRECTION_SUPERSESSION": 12, "DISTRACTOR_COMPETING_VALUES": 100, "REFERENCE_RESOLUTION": 12,
        "SPOKEN_NUMERIC_NORMALISATION": 12, "DIALOGUE_STRUCTURE": 12, "COMPLIANCE_SEMANTICS": 12,
    }
    assert validate_stage0(impossible, SCHEMAS) == []
    with pytest.raises(PlanningError, match="No feasible plan"):
        build_plan(impossible, max_attempts=3)


def test_inconsistent_quotas_are_refused_by_g0_not_relaxed(documents):
    inconsistent = deepcopy(documents)
    inconsistent["generation_plan"]["edge_case_distribution"]["family_quotas"]["DIALOGUE_STRUCTURE"] = 20
    with pytest.raises(PlanningError, match="must equal per-call edge-case total"):
        build_plan(inconsistent)


def test_missing_parameter_builder_fails_loudly(documents):
    extended = deepcopy(documents)
    clone = deepcopy(next(c for c in extended["edge_case_catalog"]["edge_cases"] if c["id"] == "EC_FALSE_START"))
    clone["id"] = "EC_UNIMPLEMENTED_CASE"
    extended["edge_case_catalog"]["edge_cases"].append(clone)
    with pytest.raises(PlanningError, match="EC_UNIMPLEMENTED_CASE"):
        build_plan(extended)


def test_invalid_g0_is_refused(documents):
    broken = deepcopy(documents)
    broken["generation_plan"]["duration_profiles"][1]["count"] = 6
    with pytest.raises(PlanningError, match="G0 contracts are invalid"):
        build_plan(broken)


def test_planning_uses_no_llm_or_network_clients():
    package = Path(allocate.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in ("anthropic", "openai", "boto3", "requests", "httpx", "urllib.request", "socket"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source


def test_cli_refuses_to_overwrite_without_flag(tmp_path):
    release = tmp_path / "advisory_execution_pilot_v1"
    shutil.copytree(RELEASE, release)
    before = {name: (release / name).read_bytes() for name in outputs.OUTPUTS}
    assert main(["--dataset-root", str(release)]) == 2
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    after = {name: (release / name).read_bytes().replace(b"\r\n", b"\n") for name in outputs.OUTPUTS}
    assert after == {name: data.replace(b"\r\n", b"\n") for name, data in before.items()}
