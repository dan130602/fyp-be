"""G0 contracts for advisory_execution_pilot_v1: approved decisions and cross-file checks.

These tests validate declarative Stage 0 inputs only. They plan no calls,
choose no gold answers and implement no spoken-form normalization.
"""

from copy import deepcopy
import json
import shutil
from pathlib import Path

import pytest

from fyp_evaluation.validation.stage0 import (
    STAGE0_FILES, check_stage0_schemas, load_stage0, validate_stage0, validate_stage0_release,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"


@pytest.fixture(scope="module")
def release_documents():
    return load_stage0(RELEASE)


def _case(documents, case_id):
    return next(c for c in documents["edge_case_catalog"]["edge_cases"] if c["id"] == case_id)


def _semantics(documents, question_id):
    return next(c for c in documents["check_semantics"]["checks"] if c["question_id"] == question_id)


def test_stage0_schema_definitions():
    check_stage0_schemas(SCHEMAS)


def test_release_stage0_is_valid(release_documents):
    before = deepcopy(release_documents)
    assert validate_stage0(release_documents, SCHEMAS) == []
    assert release_documents == before
    assert validate_stage0_release(RELEASE, SCHEMAS) == []


def test_stage0_layout():
    assert set(STAGE0_FILES.values()) == {
        "checklist.json", "check_semantics.json", "normalization.json",
        "edge_case_catalog.json", "authoring/generation_plan.json",
    }
    for relative in STAGE0_FILES.values():
        assert (RELEASE / relative).is_file()


def test_approved_checklist_wording(release_documents):
    items = {item["id"]: item for item in release_documents["checklist"]}
    assert len(items) == 20
    assert items["C02"]["question"] == "The employee should explain a concrete risk of the recommended product."
    assert items["C07"]["question"] == "The employee should explain the basis for recommending the product to the client."
    assert items["T06"]["field"] == "order_type"
    profile = release_documents["normalization"]["profiles"][_semantics(release_documents, "T06")["normalization_profile"]]
    assert profile["canonical"]["allowed_values"] == ["MARKET", "LIMIT"]


def test_release_local_normalization_owns_extraction_forms(release_documents):
    semantics = release_documents["check_semantics"]
    assert "configs/normalization.json" not in json.dumps(semantics)
    profiles = release_documents["normalization"]["profiles"]
    expected = {
        "K01": "synthetic_nric_fin", "K02": "dob_dd_month_yyyy", "T01": "synthetic_product_name",
        "T02": "currency_iso_4217_alpha3", "T03": "monetary_amount", "T04": "integer_quantity",
        "T05": "transaction_direction", "T06": "order_type",
    }
    for question_id, profile_id in expected.items():
        assert _semantics(release_documents, question_id)["normalization_profile"] == profile_id
    assert profiles["dob_dd_month_yyyy"]["value_type"] == "dob_dd_month_yyyy"
    assert profiles["integer_quantity"]["value_type"] == "integer"
    assert profiles["transaction_direction"]["canonical"]["allowed_values"] == ["BUY", "SELL"]
    currency = profiles["currency_iso_4217_alpha3"]
    assert currency["canonical"]["allowed_values"] == ["SGD", "USD", "EUR", "CNY", "JPY"]
    assert {"Singapore dollars", "US dollars", "Chinese yuan", "renminbi", "Japanese yen"} <= {
        form for forms in currency["aliases"].values() for form in forms
    }
    gated = {
        form["id"]: form["enabled_by_edge_case"]
        for profile in profiles.values() for form in profile["spoken_forms"]
        if form["usage"] == "EDGE_CASE_ONLY"
    }
    assert gated == {
        "OH_FOR_ZERO": "EC_OH_VS_ZERO",
        "DOUBLE_OR_TRIPLE_DIGIT": "EC_DOUBLE_TRIPLE_DIGITS",
        "FINANCIAL_SHORTHAND": "EC_FINANCIAL_SHORTHAND",
        "MAGNITUDE_EXPRESSION": "EC_LARGE_NUMBER_EXPRESSION",
    }


def test_only_hard_compliance_dependency_is_c03_requires_c02(release_documents):
    rules = release_documents["check_semantics"]["cross_check_rules"]
    assert [(r["if"], r["then"]) for r in rules] == [
        ({"question_id": "C03", "verdict": "PASS"}, [{"question_id": "C02", "verdict": "PASS"}]),
    ]


def test_approved_generation_plan(release_documents):
    plan = release_documents["generation_plan"]
    assert plan["num_calls"] == 50
    assert [(d["profile"], d["count"], d["min_seconds"], d["max_seconds"]) for d in plan["duration_profiles"]] == [
        ("SHORT", 45, 180, 300), ("LONG", 5, 1801, 3600),
    ]
    assert plan["call_intent_set_counts"] == [{"call_intents": ["ADVISORY", "EXECUTION"], "count": 50}]
    assert plan["planning_rules"]["call_intents_csv_encoding"] == "PIPE_DELIMITED"
    assert plan["planning_rules"]["call_intents_are_descriptive_only"] is True
    assert plan["question_scope"] == {"strategy": "ALL_CHECKLIST_QUESTIONS"}
    assert plan["call_composition_counts"] == {
        "all_pass_without_edge_cases": 5, "all_pass_with_edge_cases": 5,
        "some_fail_without_edge_cases": 5, "some_fail_with_edge_cases": 35,
    }
    assert plan["compliance_distribution"] == {
        "failures_per_non_all_pass_call": {"min": 1, "max": 3}, "min_fail_cases_per_check": 5,
    }
    assert plan["edge_case_distribution"]["family_quotas"] == {
        "CORRECTION_SUPERSESSION": 12, "DISTRACTOR_COMPETING_VALUES": 10, "REFERENCE_RESOLUTION": 6,
        "SPOKEN_NUMERIC_NORMALISATION": 10, "DIALOGUE_STRUCTURE": 8, "COMPLIANCE_SEMANTICS": 10,
    }
    nric = plan["extraction_distribution"]["synthetic_kyc"]["nric_fin"]
    assert nric["require_checksum_invalid"] and nric["unique_within_dataset"]
    assert plan["extraction_distribution"]["synthetic_kyc"]["dob"]["unique_within_dataset"]
    assert "limit_price" not in json.dumps(plan)
    assert "spoken_form_preferences" not in plan
    keys = []

    def collect(value):
        if isinstance(value, dict):
            keys.extend(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(plan)
    assert not [key for key in keys if "fraction" in key]


def test_edge_case_catalog_structure_preserved(release_documents):
    catalog = release_documents["edge_case_catalog"]
    assert all("parameter_builder" not in case for case in catalog["edge_cases"])
    assert _case(release_documents, "EC_EMPLOYEE_READBACK_ERROR")["mutually_exclusive_with"] == ["EC_ONE_FIELD_READBACK_ERROR"]
    assert _case(release_documents, "EC_ONE_FIELD_READBACK_ERROR")["mutually_exclusive_with"] == ["EC_EMPLOYEE_READBACK_ERROR"]
    rules = release_documents["generation_plan"]["planning_rules"]
    assert rules["complete_order_replacement_excludes_other_transaction_correction_edge_cases_in_same_call"] is True


def test_approved_g1_rules(release_documents):
    rules = release_documents["generation_plan"]["planning_rules"]
    assert rules["max_edge_cases_per_target_question_per_call"] == 1
    assert rules["per_question_limit_exempt_edge_cases"] == ["EC_LONG_DISTANCE_DEPENDENCY"]
    assert rules["superseded_and_distractor_values_must_satisfy_domain_and_realism_constraints"] is True
    assert rules["complete_order_replacement_initial_order_must_be_plausible"] is True
    assert rules["no_fee_distractor_when_c04_fail"] is True
    assert rules["unspecified_choices_strategy"] == "SEEDED_BALANCED_STABLE_TIE_BREAK"


def _set(path, value):
    def mutate(documents):
        *parents, last = path
        target = documents
        for key in parents:
            target = target[key]
        target[last] = value
    return mutate


def _edit_case(case_id, function):
    return lambda documents: function(_case(documents, case_id))


def _edit_semantics(question_id, function):
    return lambda documents: function(_semantics(documents, question_id))


def _profile(profile_id, function):
    return lambda documents: function(documents["normalization"]["profiles"][profile_id])


def _swap_first_checks(documents):
    checks = documents["check_semantics"]["checks"]
    checks[0], checks[1] = checks[1], checks[0]


EXTRACTION = ("generation_plan", "extraction_distribution")

INVALID = {
    "missing profile reference": (
        _edit_semantics("T02", lambda entry: entry.pop("normalization_profile")), "check_semantics: $.checks"),
    "unknown profile": (
        _edit_semantics("T02", lambda entry: entry.update(normalization_profile="missing")), "unknown normalization profile"),
    "check order": (_swap_first_checks, "checklist order"),
    "N/A note": (
        _edit_semantics("C01", lambda entry: entry["notes"].append("N/A")), "N/A-style labels"),
    "NOT_APPLICABLE note": (
        _edit_semantics("C01", lambda entry: entry["notes"].append("NOT_APPLICABLE")), "N/A-style labels"),
    "applicability key": (
        _profile("currency_iso_4217_alpha3", lambda p: p["aliases"].update(applicability=["x"])), "applicability keys"),
    "duration total": (_set(("generation_plan", "duration_profiles", 1, "count"), 6), "duration counts sum"),
    "fractional count": (_set(("generation_plan", "duration_profiles", 0, "count"), 0.9), "generation_plan: $.duration_profiles"),
    "composition total": (_set(("generation_plan", "call_composition_counts", "some_fail_with_edge_cases"), 34), "call composition counts"),
    "intent order": (
        _set(("generation_plan", "call_intent_set_counts", 0, "call_intents"), ["EXECUTION", "ADVISORY"]), "vocabulary order"),
    "unknown intent": (
        _set(("generation_plan", "call_intent_set_counts", 0, "call_intents"), ["ADVISORY", "TRADING"]), "not in call_intent_vocabulary"),
    "applicability scope": (
        _set(("generation_plan", "question_scope", "strategy"), "BY_CALL_INTENT"), "generation_plan: $.question_scope"),
    "asymmetric exclusion": (
        _edit_case("EC_ONE_FIELD_READBACK_ERROR", lambda c: c.pop("mutually_exclusive_with")), "must be symmetric"),
    "contradicts C03 rule": (
        _edit_case("EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED", lambda c: (
            c.update(verdict_requirements={"C02": "FAIL", "C03": "PASS"}),
            c["plan_template"].update(required_related_gold={"C02": "FAIL", "C03": "PASS"}))),
        "contradict C03_PASS_REQUIRES_C02_PASS"),
    "related gold mismatch": (
        _edit_case("EC_WEAK_CONFIRMATION", lambda c: c["plan_template"].update(required_related_gold={"C08": "PASS"})),
        "must equal verdict_requirements"),
    "unplanned product currency": (
        _set((*EXTRACTION, "product_pool", 2, "allowed_currencies"), ["GBP"]), "unplanned currencies"),
    "incoherent amount range": (
        _set((*EXTRACTION, "amount_quantity_coherence", "ranges", "JPY"), {"min": 1e9, "max": 2e9}), "JPY has no coherent"),
    "unsatisfiable prerequisite": (
        _edit_case("EC_LARGE_NUMBER_EXPRESSION", lambda c: c["gold_prerequisites"].update(minimum=10 ** 12)),
        "satisfies EC_LARGE_NUMBER_EXPRESSION"),
    "limit price planned": (
        _set((*EXTRACTION, "amount_quantity_coherence", "limit_price"), 1), "generation_plan: $.extraction_distribution"),
    "spoken forms in plan": (_set(("generation_plan", "spoken_form_preferences"), {}), "generation_plan: $"),
    "duplicate alias": (
        _profile("currency_iso_4217_alpha3", lambda p: p["aliases"]["JPY"].append("yuan")), "more than one canonical value"),
    "unknown gating edge case": (
        _profile("synthetic_nric_fin", lambda p: p["spoken_forms"][1].update(enabled_by_edge_case="EC_UNKNOWN")),
        "unknown edge case EC_UNKNOWN"),
    "ungated spoken edge case": (
        _profile("synthetic_nric_fin", lambda p: p["spoken_forms"].pop(1)), "no spoken form is enabled by EC_OH_VS_ZERO"),
    "reference default mismatch": (
        _profile("monetary_amount", lambda p: p["reference_text"].update(default_form="FINANCIAL_SHORTHAND")),
        "must name the DEFAULT spoken form"),
    "cross rule on extraction": (
        _set(("check_semantics", "cross_check_rules", 0, "then", 0, "question_id"), "T01"), "not a compliance question"),
    "fail capacity": (_set(("generation_plan", "compliance_distribution", "min_fail_cases_per_check"), 11), "exceed capacity"),
    "missing family quota": (
        lambda d: d["generation_plan"]["edge_case_distribution"]["family_quotas"].pop("REFERENCE_RESOLUTION"),
        "family_quotas keys"),
    "edge quota overflow": (
        _set(("generation_plan", "edge_case_distribution", "family_quotas", "CORRECTION_SUPERSESSION"), 100), "cannot fill"),
    "direction values": (
        _set((*EXTRACTION, "transaction_direction_counts"), {"BUY": 25, "HOLD": 25}), "normalization allowed_values"),
    "nric prefix": (_set((*EXTRACTION, "synthetic_kyc", "nric_fin", "prefixes"), ["S", "X"]), "prefix X"),
    "unknown exempt edge case": (
        _set(("generation_plan", "planning_rules", "per_question_limit_exempt_edge_cases"), ["EC_UNKNOWN"]),
        "unknown edge cases ['EC_UNKNOWN']"),
    "per-question limit relaxed": (
        _set(("generation_plan", "planning_rules", "max_edge_cases_per_target_question_per_call"), 2),
        "generation_plan: $.planning_rules"),
    "fee distractor rule disabled": (
        _set(("generation_plan", "planning_rules", "no_fee_distractor_when_c04_fail"), False),
        "generation_plan: $.planning_rules"),
    "parameter builder": (
        _edit_case("EC_FALSE_START", lambda c: c.update(parameter_builder="x")), "edge_case_catalog: $.edge_cases"),
    "compliance target on extraction family": (
        _edit_case("EC_FALSE_START", lambda c: c["allowed_target_question_ids"].append("C01")), "must target only extraction"),
}


@pytest.mark.parametrize("label", INVALID)
def test_stage0_contradictions_fail_loudly(release_documents, label):
    mutate, message = INVALID[label]
    documents = deepcopy(release_documents)
    mutate(documents)
    errors = validate_stage0(documents, SCHEMAS)
    assert any(message in error for error in errors), errors


def test_release_dataset_id_must_match_manifest(tmp_path):
    release = tmp_path / "advisory_execution_pilot_v1"
    shutil.copytree(RELEASE, release)
    manifest = json.loads((release / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "other_release"
    (release / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "normalization: dataset_id must equal dataset_manifest.json dataset_id" in validate_stage0_release(release, SCHEMAS)
