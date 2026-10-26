"""G0 contracts for advisory_execution_pilot_v1: approved decisions and cross-file checks.

These tests validate declarative Stage 0 inputs only. They plan no calls,
choose no gold answers and implement no spoken-form normalization.
"""

from copy import deepcopy
import json
import re
from pathlib import Path

import pytest

from fyp_evaluation.validation.stage0 import (
    STAGE0_FILES, check_stage0_schemas, load_stage0, validate_stage0, validate_stage0_release,
)
from release_copy import copy_tree

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
        "fail_count_per_call_counts": {"0": 10, "1": 10, "2": 15, "3": 10, "4": 5},
        "min_fail_cases_per_check": 6,
    }
    assert plan["edge_case_distribution"]["edge_cases_per_call_counts"] == {"0": 10, "1": 10, "2": 15, "3": 10, "4": 5}
    assert plan["edge_case_distribution"]["family_quotas"] == {
        "CORRECTION_SUPERSESSION": 18, "DISTRACTOR_COMPETING_VALUES": 16, "REFERENCE_RESOLUTION": 10,
        "SPOKEN_NUMERIC_NORMALISATION": 16, "DIALOGUE_STRUCTURE": 14, "COMPLIANCE_SEMANTICS": 16,
    }
    assert sum(int(n) * c for n, c in plan["compliance_distribution"]["fail_count_per_call_counts"].items()) == 90
    assert sum(plan["edge_case_distribution"]["family_quotas"].values()) == 90
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


def test_catalog_declares_every_parameter_g1_produces(release_documents):
    assert _case(release_documents, "EC_MULTIPLE_CURRENCIES")["required_parameters"] == [
        "distractor_currency", "distractor_role", "final_currency",
    ]
    multiple = _case(release_documents, "EC_MULTIPLE_CORRECTIONS")
    assert multiple["allowed_target_question_ids"] == ["K01", "K02", "T01", "T02", "T03", "T04"]


def test_k01_suffix_pool_excludes_letter_o(release_documents):
    plan = release_documents["generation_plan"]["extraction_distribution"]["synthetic_kyc"]["nric_fin"]
    assert "O" not in plan["suffix_letters"] and "O" not in plan["prefixes"]
    assert len(plan["suffix_letters"]) == 25
    canonical = release_documents["normalization"]["profiles"]["synthetic_nric_fin"]["canonical"]
    assert canonical["excluded_characters"] == ["O"]
    assert not re.fullmatch(canonical["pattern"], "S1234567O") and re.fullmatch(canonical["pattern"], "S1234567P")


def test_presence_patterns_and_ordering_rules(release_documents):
    semantics = release_documents["check_semantics"]
    patterns = {c["question_id"]: c["presence_pattern"] for c in semantics["checks"] if "presence_pattern" in c}
    assert patterns == {
        **{q: "EMPLOYEE_ONLY" for q in ("C01", "C02", "C04", "C05", "C07", "C11", "C12")},
        **{q: "EXCHANGE" for q in ("C03", "C06", "C08")},
        **{q: "ANY_SPEAKER" for q in ("C09", "C10")},
    }
    assert [(r["id"], r["question_id"], r["after"]) for r in semantics["ordering_rules"]] == [
        ("C03_AFTER_C02_DISCLOSURE", "C03", {"type": "CHECK_EVIDENCE", "question_id": "C02"}),
        ("C08_AFTER_FINAL_TRANSACTION_STATE", "C08",
         {"type": "FINAL_STATE", "question_ids": ["T01", "T02", "T03", "T04", "T05", "T06"]}),
    ]
    assert "turn" not in json.dumps(semantics["ordering_rules"]).replace("transaction", "")


def test_c05_fail_is_the_complement_of_pass(release_documents):
    c05 = _semantics(release_documents, "C05")
    assert "does not explicitly explain" in c05["fail_condition"]
    assert "Merely implying loss" in c05["fail_condition"] and "loss tolerance" in c05["fail_condition"]
    assert "communicate" not in c05["fail_condition"]


def test_relative_structure_only_catalog_additions(release_documents):
    assert _case(release_documents, "EC_MULTIPLE_CORRECTIONS")["plan_template"]["superseded_states_before_final"] is True
    false_start = _case(release_documents, "EC_FALSE_START")["plan_template"]
    assert false_start["same_turn"] is True and false_start["fragment_must_be_incomplete"] is True
    readback = _case(release_documents, "EC_ONE_FIELD_READBACK_ERROR")["plan_template"]
    assert readback["min_readback_fields"] == 2 and readback["readback_wrong_fields"] == "TARGET_ONLY"
    for case_id in ("EC_MULTIPLE_CURRENCIES", "EC_MULTIPLE_MONETARY_VALUES"):
        assert _case(release_documents, case_id)["allowed_distractor_speakers"] == ["CLIENT", "EMPLOYEE"]
    assert _case(release_documents, "EC_PARAPHRASED_COMPLIANCE_STATEMENT")["plan_template"]["wording_constraint"] == "NOT_VERBATIM_CHECKLIST"
    large = _case(release_documents, "EC_LARGE_NUMBER_EXPRESSION")["gold_prerequisites"]
    assert large == {"kind": "NUMERIC_MINIMUM_NOT_MULTIPLE", "minimum": 1000000, "not_multiple_of": 1000000}
    # No exact turn numbers or arbitrary gaps beyond the two pre-existing, intrinsic minimums.
    gaps = {c["id"]: c["plan_template"].get("min_intervening_turns", c["plan_template"].get("default_min_turn_gap"))
            for c in release_documents["edge_case_catalog"]["edge_cases"]
            if {"min_intervening_turns", "default_min_turn_gap"} & set(c["plan_template"])}
    assert gaps == {"EC_CORRECTION_SAME_TURN": 0, "EC_CORRECTION_CROSS_TURN": 2, "EC_LONG_DISTANCE_DEPENDENCY": 6}


def test_spoken_forms_are_deterministic(release_documents):
    forms = {f["id"]: f for p in release_documents["normalization"]["profiles"].values() for f in p["spoken_forms"]}
    assert forms["OH_FOR_ZERO"]["applies_to"] == "ALL_ZERO_DIGITS"
    assert forms["DOUBLE_OR_TRIPLE_DIGIT"]["applies_to"] == "ALL_RUNS_OF_ASSIGNED_VARIANT_LENGTH"
    for form_id in ("OH_FOR_ZERO", "DOUBLE_OR_TRIPLE_DIGIT"):
        assert forms[form_id]["remaining_characters_form"] == "CHARACTER_BY_CHARACTER"
        assert forms[form_id]["usage"] == "EDGE_CASE_ONLY"
    assert forms["FINANCIAL_SHORTHAND"]["applies_to"] == forms["MAGNITUDE_EXPRESSION"]["applies_to"] == "WHOLE_VALUE"
    assert all("applies_to" not in f for f in forms.values() if f["usage"] != "EDGE_CASE_ONLY")


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
    "fail capacity": (
        _set(("generation_plan", "compliance_distribution", "min_fail_cases_per_check"), 8), "exceed the 90 planned FAIL cells"),
    "fail counts total": (
        _set(("generation_plan", "compliance_distribution", "fail_count_per_call_counts", "4"), 4), "fail_count_per_call counts sum"),
    "zero-FAIL calls differ from all-PASS calls": (
        lambda d: d["generation_plan"]["compliance_distribution"]["fail_count_per_call_counts"].update({"0": 9, "1": 11}),
        "must equal all-PASS calls"),
    "too many FAILs for one call": (
        lambda d: d["generation_plan"]["compliance_distribution"]["fail_count_per_call_counts"].update({"4": 4, "13": 1}),
        "more FAILs than the 12 compliance questions"),
    "edge counts total": (
        _set(("generation_plan", "edge_case_distribution", "edge_cases_per_call_counts", "3"), 9), "edge_cases_per_call counts sum"),
    "zero-edge calls differ from clean calls": (
        lambda d: d["generation_plan"]["edge_case_distribution"]["edge_cases_per_call_counts"].update({"0": 9, "1": 11}),
        "must equal calls without edge cases"),
    "quota total differs from per-call total": (
        _set(("generation_plan", "edge_case_distribution", "family_quotas", "DIALOGUE_STRUCTURE"), 13),
        "must equal per-call edge-case total"),
    "binary field cannot take two superseded values": (
        _edit_case("EC_MULTIPLE_CORRECTIONS", lambda c: c["allowed_target_question_ids"].append("T05")),
        "two superseded values plus the final value need at least three"),
    "fractional per-call count": (
        _set(("generation_plan", "edge_case_distribution", "edge_cases_per_call_counts", "2"), 15.5), "generation_plan: $.edge_case_distribution"),
    "missing family quota": (
        lambda d: d["generation_plan"]["edge_case_distribution"]["family_quotas"].pop("REFERENCE_RESOLUTION"),
        "family_quotas keys"),
    "edge quota overflow": (
        _set(("generation_plan", "edge_case_distribution", "family_quotas", "CORRECTION_SUPERSESSION"), 100), "must equal per-call edge-case total"),
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
    "O reintroduced into suffix pool": (
        lambda d: d["generation_plan"]["extraction_distribution"]["synthetic_kyc"]["nric_fin"].update(
            suffix_letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "does not fit the canonical pattern"),
    "excluded characters without reason": (
        _profile("synthetic_nric_fin", lambda p: p["canonical"].pop("excluded_characters_reason")),
        "must appear together"),
    "canonical pattern accepts excluded character": (
        _profile("synthetic_nric_fin", lambda p: p["canonical"].update(pattern="^[STFGM][0-9]{7}[A-Z]$")),
        "canonical pattern still accepts excluded character O"),
    "edge-case form without applies_to": (
        _profile("synthetic_nric_fin", lambda p: p["spoken_forms"][1].pop("applies_to")),
        "normalization: $.profiles"),
    "partial coverage without remaining form": (
        _profile("synthetic_nric_fin", lambda p: p["spoken_forms"][1].pop("remaining_characters_form")),
        "partial coverage requires remaining_characters_form"),
    "remaining form is not the default": (
        _profile("synthetic_nric_fin", lambda p: p["spoken_forms"][1].update(remaining_characters_form="DOUBLE_OR_TRIPLE_DIGIT")),
        "remaining_characters_form must name the DEFAULT spoken form"),
    "default form carries applies_to": (
        _profile("synthetic_nric_fin", lambda p: p["spoken_forms"][0].update(applies_to="ALL_ZERO_DIGITS")),
        "normalization: $.profiles"),
    "missing presence_pattern": (
        _edit_semantics("C03", lambda entry: entry.pop("presence_pattern")), "check_semantics: $.checks"),
    "EMPLOYEE_ONLY with both speakers": (
        _edit_semantics("C01", lambda entry: entry.update(speaker_scope=["EMPLOYEE", "CLIENT"])),
        "EMPLOYEE_ONLY requires speaker_scope [EMPLOYEE]"),
    "EXCHANGE with one speaker": (
        _edit_semantics("C03", lambda entry: entry.update(speaker_scope=["EMPLOYEE"])), "EXCHANGE requires both speakers"),
    "ordering rule on extraction question": (
        _set(("check_semantics", "ordering_rules", 0, "question_id"), "T01"), "is not a compliance question"),
    "ordering after itself": (
        _set(("check_semantics", "ordering_rules", 0, "after", "question_id"), "C03"), "cannot order a question after itself"),
    "FINAL_STATE names compliance question": (
        _set(("check_semantics", "ordering_rules", 1, "after", "question_ids"), ["C02"]), "FINAL_STATE must name transaction questions"),
    "ordering cycle": (
        lambda d: d["check_semantics"]["ordering_rules"].append({
            "id": "C02_AFTER_C03", "question_id": "C02", "after": {"type": "CHECK_EVIDENCE", "question_id": "C03"},
            "reason": "artificial cycle"}),
        "form a cycle"),
    "distractor speakers removed": (
        _edit_case("EC_MULTIPLE_CURRENCIES", lambda c: c.pop("allowed_distractor_speakers")),
        "allowed_distractor_speakers must appear exactly on DISTRACTOR_PLUS_TARGET cases"),
    "distractor speakers on wrong case": (
        _edit_case("EC_FALSE_START", lambda c: c.update(allowed_distractor_speakers=["CLIENT"])),
        "allowed_distractor_speakers must appear exactly on DISTRACTOR_PLUS_TARGET cases"),
    "incomplete fragment outside same turn": (
        _edit_case("EC_FALSE_START", lambda c: c["plan_template"].update(same_turn=False)),
        "incomplete false-start fragment requires same_turn true"),
    "readback minimum without wrong-field rule": (
        _edit_case("EC_ONE_FIELD_READBACK_ERROR", lambda c: c["plan_template"].pop("readback_wrong_fields")),
        "must appear together"),
    "readback minimum exceeds fields": (
        _edit_case("EC_ONE_FIELD_READBACK_ERROR", lambda c: c["plan_template"].update(min_readback_fields=7)),
        "needs at least min_readback_fields"),
    "wording constraint on extraction case": (
        _edit_case("EC_FALSE_START", lambda c: c["plan_template"].update(wording_constraint="NOT_VERBATIM_CHECKLIST")),
        "wording_constraint applies only to compliance cases"),
    "large-number prerequisite unsatisfiable": (
        _edit_case("EC_LARGE_NUMBER_EXPRESSION", lambda c: c["gold_prerequisites"].update(not_multiple_of=1)),
        "satisfies EC_LARGE_NUMBER_EXPRESSION"),
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
    copy_tree(RELEASE, release)  # no rendered audio (release_copy)
    manifest = json.loads((release / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "other_release"
    (release / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "normalization: dataset_id must equal dataset_manifest.json dataset_id" in validate_stage0_release(release, SCHEMAS)
