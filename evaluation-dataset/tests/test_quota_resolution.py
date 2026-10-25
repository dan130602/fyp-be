"""Quota resolution for the resized multilingual release (250 calls): joint rounding, stored-plan reproducibility and coverage.

The resolver is pure Python over the committed design and catalog; nothing here calls an LLM, a network or AWS service.
"""

import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from fyp_evaluation.planning.quotas import QuotaError, largest_remainder, resolve_histogram, resolve_quotas, round_matrix, round_table
from fyp_evaluation.planning.quotas_cli import DESIGN, PLAN, resolved_plan
from fyp_evaluation.planning.reports import coverage_report, coverage_groups, proportional_split, small_group_flag, split_flag
from fyp_evaluation.validation.stage0 import load_stage0, validate_stage0

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_multilingual_v1"
PROFILES = ["ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH"]
CALLS = {"ENGLISH": 75, "MANDARIN": 75, "EN_ZH_MIXED": 50, "SINGLISH": 50}


@pytest.fixture(scope="module")
def docs():
    return load_stage0(RELEASE)


@pytest.fixture(scope="module")
def design():
    return json.loads((RELEASE / DESIGN).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plans(docs):
    return {p["language_profile"]: p for p in docs["generation_plan"]["language_profile_plans"]}


def total(plans, key, sub=None):
    return sum((p[key] if sub is None else p[key][sub]) if sub is None or isinstance(p[key], dict) else 0 for p in plans.values())


# ---------------------------------------------------------------------------------------- rounding primitives
def test_largest_remainder_hits_the_total_and_stays_close():
    shares = {"a": Fraction(5, 2), "b": Fraction(5, 2), "c": Fraction(2)}
    result = largest_remainder(shares, 7, seed=1, label="t")
    assert sum(result.values()) == 7 and result["c"] == 2 and {result["a"], result["b"]} == {2, 3}
    assert result == largest_remainder(shares, 7, seed=1, label="t")
    with pytest.raises(QuotaError):
        largest_remainder(shares, 12, seed=1, label="t")


def test_round_matrix_keeps_every_marginal_exact_and_cells_within_floor_and_ceil():
    shares = {("r1", "x"): Fraction(3, 2), ("r1", "y"): Fraction(3, 2), ("r2", "x"): Fraction(3, 2), ("r2", "y"): Fraction(3, 2)}
    table = round_matrix(shares, ["r1", "r2"], ["x", "y"], {"r1": 3, "r2": 3}, {"x": 3, "y": 3}, 1, "m")
    assert sum(table["r1", c] for c in "xy") == 3 and sum(table[r, "x"] for r in ("r1", "r2")) == 3
    assert all(v in (1, 2) for v in table.values())


def test_round_matrix_and_table_refuse_an_impossible_request():
    shares = {("r1", "x"): Fraction(1, 2), ("r1", "y"): Fraction(1, 2)}
    with pytest.raises(QuotaError):
        round_matrix(shares, ["r1"], ["x", "y"], {"r1": 2}, {"x": 2, "y": 0}, 1, "m")
    with pytest.raises(QuotaError):
        round_table(shares, ["r1"], ["x", "y"], {"r1": 5}, 1, "m")


def test_positive_shares_keep_at_least_the_minimum():
    shares = {("r1", "x"): Fraction(1, 10), ("r1", "y"): Fraction(19, 10)}
    table = round_matrix(shares, ["r1"], ["x", "y"], {"r1": 2}, {"x": 1, "y": 1}, 1, "m", minimum_positive=1)
    assert table["r1", "x"] == 1


def test_histogram_resolution_is_exact_in_every_total():
    ref = {"A": {"0": 3, "1": 4, "2": 3}, "B": {"0": 3, "1": 4, "2": 3}}
    scale = {"A": Fraction(1, 2), "B": Fraction(1, 2)}
    result = resolve_histogram(["A", "B"], {"A": 5, "B": 5}, ref, scale, {"A": 5, "B": 5}, {0: 3, 1: 4, 2: 3}, 1, "h")
    assert [sum(result[p].values()) for p in "AB"] == [5, 5]
    assert [sum(int(k) * n for k, n in result[p].items()) for p in "AB"] == [5, 5]
    assert sum(result[p]["0"] for p in "AB") == 3
    with pytest.raises(QuotaError):
        resolve_histogram(["A", "B"], {"A": 5, "B": 5}, ref, scale, {"A": 9, "B": 1}, {0: 3, 1: 4, 2: 3}, 1, "h")


# ---------------------------------------------------------------------------------------- the resolved release
def test_stored_quotas_equal_the_recomputed_quotas(docs):
    resolved, text = resolved_plan(RELEASE)
    assert (RELEASE / PLAN).read_bytes().decode("utf-8").replace("\r\n", "\n") == text
    assert validate_stage0(resolved, SCHEMAS) == []


def test_resolution_is_deterministic_and_records_its_design(docs, design):
    first, second = resolved_plan(RELEASE)[1], resolved_plan(RELEASE)[1]
    assert first == second
    resolution = docs["generation_plan"]["quota_resolution"]
    assert resolution["design_file"] == DESIGN
    import hashlib
    assert resolution["design_sha256"] == hashlib.sha256((RELEASE / DESIGN).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_design_reference_is_the_approved_500_call_plan(design):
    reference = design["reference"]["language_profile_plans"]
    assert sum(p["num_calls"] for p in reference) == 500
    assert sum(sum(p["edge_case_counts"].values()) for p in reference) == 1120
    assert sum(sum(p["fail_counts_per_check"].values()) for p in reference) == 1120


def test_profile_and_global_totals_are_consistent(docs, plans):
    assert docs["generation_plan"]["num_calls"] == 250 and {n: p["num_calls"] for n, p in plans.items()} == CALLS
    assert sum(sum(p["edge_case_counts"].values()) for p in plans.values()) == 560
    assert sum(sum(p["fail_counts_per_check"].values()) for p in plans.values()) == 560
    assert [sum(p["fail_count_per_call_counts"][k] for p in plans.values()) for k in "01234"] == [25, 40, 75, 70, 40]
    assert [sum(p["edge_cases_per_call_counts"][k] for p in plans.values()) for k in "01234"] == [25, 40, 75, 70, 40]
    composition = [p["call_composition_counts"] for p in plans.values()]
    assert sum(c["all_pass_without_edge_cases"] for c in composition) == 10
    assert sum(c["all_pass_without_edge_cases"] + c["all_pass_with_edge_cases"] for c in composition) == 25
    assert sum(c["all_pass_without_edge_cases"] + c["some_fail_without_edge_cases"] for c in composition) == 25
    assert sum(p["duration_counts"]["SHORT"] for p in plans.values()) == 225
    assert sum(p["duration_counts"]["LONG"] for p in plans.values()) == 25
    for p in plans.values():
        assert sum(p["fail_count_per_call_counts"].values()) == p["num_calls"] == sum(p["transaction_direction_counts"].values())
        assert sum(sum(c.values()) for c in [p["transaction_currency_counts"], p["order_type_counts"]]) == 2 * p["num_calls"]
    for key in ("transaction_direction_counts", "order_type_counts"):
        for name in plans["ENGLISH"][key]:
            assert sum(p[key][name] for p in plans.values()) == 125


def test_per_question_fail_targets_follow_the_dependency_and_reach_the_global_targets(plans):
    for p in plans.values():
        fails = p["fail_counts_per_check"]
        assert fails["C03"] == fails["C02"] + p["edge_case_counts"]["EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED"]
        assert fails["C08"] >= p["edge_case_counts"]["EC_WEAK_CONFIRMATION"]
    expected = {"C02": 40, "C03": 65, "C08": 50}
    for q in plans["ENGLISH"]["fail_counts_per_check"]:
        assert sum(p["fail_counts_per_check"][q] for p in plans.values()) == expected.get(q, 45)


def test_exposure_caps_round_down(plans):
    assert {n: p["target_exposure_caps"]["T03"] for n, p in plans.items()} == {"ENGLISH": 26, "MANDARIN": 26, "EN_ZH_MIXED": 17, "SINGLISH": 17}
    for p in plans.values():
        assert p["target_exposure_caps"]["T03"] * 100 <= 35 * p["num_calls"] < (p["target_exposure_caps"]["T03"] + 1) * 100


def test_mandarin_option_b_counts_are_preserved(plans):
    zh = plans["MANDARIN"]
    assert zh["language_variant_counts"] == {"EC_MULTIPLE_MONETARY_VALUES": {"ZH_MAGNITUDE_CONTRAST": 6}}
    assert (zh["edge_case_counts"]["EC_MULTIPLE_CURRENCIES"], zh["edge_case_counts"]["EC_MULTIPLE_MONETARY_VALUES"]) == (10, 17)
    assert zh["edge_case_counts"]["EC_MULTIPLE_MONETARY_VALUES"] - 6 == zh["edge_case_counts"]["EC_MULTIPLE_CURRENCIES"] + 1
    for case_id in ("EC_DOUBLE_TRIPLE_DIGITS", "EC_OH_VS_ZERO", "EC_FINANCIAL_SHORTHAND", "EC_LARGE_NUMBER_EXPRESSION"):
        assert zh["edge_case_counts"][case_id] == 0


def test_mixed_cross_language_variants_total_half_the_approved_68_and_cover_both_directions(plans, design):
    mixed = plans["EN_ZH_MIXED"]["language_variant_counts"]
    cross = {c: by for c, by in mixed.items() if any(v.startswith("CROSS_LANGUAGE_") for v in by)}
    assert len(cross) == 12 and sum(sum(by.values()) for by in cross.values()) == 34
    for case_id, by in cross.items():
        assert set(by) == {"CROSS_LANGUAGE_EN_TO_ZH", "CROSS_LANGUAGE_ZH_TO_EN"} and min(by.values()) >= 1, case_id
        assert sum(by.values()) <= plans["EN_ZH_MIXED"]["edge_case_counts"][case_id]
    assert sum(by["CROSS_LANGUAGE_EN_TO_ZH"] for by in cross.values()) == 17
    reference = {p["language_profile"]: p for p in design["reference"]["language_profile_plans"]}["EN_ZH_MIXED"]
    assert sum(n for by in reference["language_variant_counts"].values() for v, n in by.items() if v.startswith("CROSS")) == 68


def test_every_eligible_mechanism_is_covered_in_every_applicable_profile(docs, plans):
    for case in docs["edge_case_catalog"]["edge_cases"]:
        for name, p in plans.items():
            if name in case["language_profiles"] and not (case["family"] == "SPOKEN_NUMERIC_NORMALISATION" and name == "MANDARIN"):
                assert p["edge_case_counts"][case["id"]] >= 1, (name, case["id"])
            if name not in case["language_profiles"]:
                assert p["edge_case_counts"][case["id"]] == 0, (name, case["id"])


def test_mixing_styles_and_stratified_distributions(plans):
    assert plans["EN_ZH_MIXED"]["mixing_style_counts"] == {"EN_LED": 17, "ZH_LED": 18, "BALANCED": 15}
    for p in plans.values():
        counts = p["transaction_currency_counts"]
        assert set(counts.values()) == {p["num_calls"] // 5}


# ---------------------------------------------------------------------------------------- refusals
def test_a_resize_without_an_exact_integer_solution_is_refused(design, docs):
    broken = deepcopy(design)
    broken["target"]["profile_calls"]["ENGLISH"] = 76
    broken["target"]["num_calls"] = 251
    with pytest.raises(QuotaError):
        resolve_quotas(broken, docs["edge_case_catalog"], docs["generation_plan"]["planning_rules"])


def test_another_seed_changes_only_tie_breaks_and_stays_exact(design, docs):
    other = deepcopy(design)
    other["seed"] = design["seed"] + 1
    plans, calls = resolve_quotas(other, docs["edge_case_catalog"], docs["generation_plan"]["planning_rules"])
    assert calls == 250 and sum(sum(p["edge_case_counts"].values()) for p in plans) == 560
    assert sum(sum(p["fail_counts_per_check"].values()) for p in plans) == 560


# ---------------------------------------------------------------------------------------- coverage and split assessment
def test_split_assessment_helpers():
    assert small_group_flag(1) == "SINGLETON" and small_group_flag(3) == "VERY SMALL" and small_group_flag(4) == ""
    assert proportional_split(1, 30) == (1, 0) and proportional_split(2, 30) == (1, 1) and proportional_split(4, 30) == (3, 1)
    assert proportional_split(10, 30) == (7, 3) and proportional_split(5, 30) == (3, 2)
    assert split_flag(1, 1, 0) == "CANNOT APPEAR IN BOTH" and split_flag(4, 3, 1) == "ONE EXAMPLE IN A SUBSET" and split_flag(10, 7, 3) == ""


def test_coverage_report_flags_what_a_split_would_leave_thin(docs):
    groups = coverage_groups(docs)
    assert ("EN_ZH_MIXED", "EC_MULTIPLE_CORRECTIONS / CROSS_LANGUAGE_EN_TO_ZH", 1) in groups
    assert not [g for g in groups if g[0] == "ALL PROFILES" and "CROSS_LANGUAGE" in g[1]], "single-profile mechanisms are not repeated"
    report = coverage_report(RELEASE)
    assert "Eligible mechanisms with no examples: none." in report
    assert "Cross-language cases missing a switch direction: none." in report
    assert "EN_ZH_MIXED | EC_MULTIPLE_CORRECTIONS / CROSS_LANGUAGE_EN_TO_ZH | 1 | 1 | 0 | CANNOT APPEAR IN BOTH" in report


# ---------------------------------------------------------------------------------------- generated artifacts
def test_only_250_calls_remain_in_the_active_release():
    manifest = json.loads((RELEASE / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["call_ids"] == [f"CALL_{n:03d}" for n in range(1, 251)]
    for directory in ("call_specs", "check_plan_skeletons"):
        names = sorted(path.name for path in (RELEASE / "authoring" / directory).iterdir())
        assert names == [f"CALL_{n:03d}.json" for n in range(1, 251)]
    assert not (RELEASE / "calls").exists()
