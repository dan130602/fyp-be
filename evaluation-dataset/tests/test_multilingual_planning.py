"""Multilingual release (advisory_execution_multilingual_v1): G0 contracts, G1 planning, G2A specs and G2B skeletons.

The approved 500-call configuration is planned in memory and compared with the committed artifacts; negative cases
mutate in-memory copies of the documents or outputs. No LLM, network or AWS service is used.
"""

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from fyp_evaluation.call_configs.inputs import G2BStateError, load_inputs as load_g3_inputs
from fyp_evaluation.call_specs import compile as g2a
from fyp_evaluation.call_specs.outputs import read_disk as read_g2a_disk
from fyp_evaluation.call_specs.validate import validate_spec_set
from fyp_evaluation.check_plan_skeletons import inputs as g2b_inputs
from fyp_evaluation.check_plan_skeletons.compile import skeleton_path
from fyp_evaluation.check_plan_skeletons.outputs import read_disk as read_g2b_disk
from fyp_evaluation.check_plan_skeletons.validate import validate_skeleton_set
from fyp_evaluation.planning import outputs
from fyp_evaluation.planning.allocate import build_plan
from fyp_evaluation.planning.domain import Domain
from fyp_evaluation.planning.reports import compatibility_report, composition_report
from fyp_evaluation.planning.validate import validate_plan
from fyp_evaluation.validation.stage0 import (
    STAGE0_FILES, check_stage0_schemas, is_multilingual, load_stage0, stage0_files, validate_stage0, validate_stage0_release,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_multilingual_v1"
PILOT = ROOT / "datasets" / "advisory_execution_pilot_v1"
PROFILES = ["ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH"]
COMPLIANCE = [f"C{i:02d}" for i in range(1, 13)]
PROHIBITION = ["C09", "C10", "C11", "C12"]


@pytest.fixture(scope="module")
def docs():
    return load_stage0(RELEASE)


@pytest.fixture(scope="module")
def generated():
    return outputs.generate(RELEASE, SCHEMAS)


@pytest.fixture(scope="module")
def parsed(generated):
    texts, _ = generated
    manifest = outputs.parse_manifest(texts[outputs.MANIFEST])
    matrix = outputs.parse_matrix(texts[outputs.MATRIX])
    assignments = outputs.parse_assignments(texts[outputs.ASSIGNMENTS])
    return manifest, matrix, assignments


@pytest.fixture(scope="module")
def plan_view(docs, parsed):
    """Typed gold, per-call assignments and profile membership of the generated plan."""
    from fyp_evaluation.planning.validate import typed_gold
    manifest, matrix, assignments = parsed
    gold, errors = typed_gold(docs, matrix, [r["call_id"] for r in manifest], SCHEMAS)
    assert not errors
    by_call = defaultdict(list)
    for a in assignments:
        by_call[a["call_id"]].append(a)
    profile = {r["call_id"]: r["language_profile"] for r in manifest}
    return gold, by_call, profile, {r["call_id"]: r for r in manifest}


def plan_of(docs, name):
    return next(p for p in docs["generation_plan"]["language_profile_plans"] if p["language_profile"] == name)


# ---------------------------------------------------------------------------------------------- G0
def test_both_releases_validate_and_schemas_are_well_formed():
    check_stage0_schemas(SCHEMAS)
    assert validate_stage0_release(PILOT, SCHEMAS) == []
    assert validate_stage0_release(RELEASE, SCHEMAS) == []


def test_release_file_sets(docs):
    pilot = load_stage0(PILOT)
    assert not is_multilingual(pilot) and stage0_files(pilot) == STAGE0_FILES
    assert is_multilingual(docs) and set(stage0_files(docs)) == set(STAGE0_FILES) | {"language_profiles"}


def test_contract_generations_cannot_be_mixed(docs):
    mixed = deepcopy(docs)
    mixed["edge_case_catalog"]["schema_version"] = "1.0"
    assert any("schema_version must be 2.0" in e for e in validate_stage0(mixed, SCHEMAS))
    missing = deepcopy(docs)
    del missing["language_profiles"]
    assert any("language_profiles.json is required" in e for e in validate_stage0(missing, SCHEMAS))


def test_large_number_currency_hole_is_rejected_at_g0(docs):
    """Regression for the EUR/CNY gap: every planned currency needs a coherent large-number amount."""
    broken = deepcopy(docs)
    amounts = broken["generation_plan"]["extraction_distribution"]["transaction_amount_values_by_currency"]
    amounts["EUR"].remove(1200000)
    amounts["CNY"].remove(2500000)
    errors = validate_stage0(broken, SCHEMAS)
    for currency in ("EUR", "CNY"):
        assert any(f"no coherent {currency} value satisfies EC_LARGE_NUMBER_EXPRESSION" in e for e in errors)


def test_dependent_fail_counts_and_variant_eligibility_are_checked(docs):
    off = deepcopy(docs)
    plan_of(off, "ENGLISH")["fail_counts_per_check"]["C03"] -= 1
    plan_of(off, "ENGLISH")["fail_counts_per_check"]["C04"] += 1
    assert any("C03 FAILs (38) must equal C02 FAILs (24) plus" in e for e in validate_stage0(off, SCHEMAS))
    ineligible = deepcopy(docs)
    counts = plan_of(ineligible, "MANDARIN")["edge_case_counts"]
    counts["EC_OH_VS_ZERO"] += 1
    counts["EC_FALSE_START"] -= 1
    assert any("EC_OH_VS_ZERO is not eligible" in e for e in validate_stage0(ineligible, SCHEMAS))
    excluded = deepcopy(docs)
    plan_of(excluded, "MANDARIN")["language_variant_counts"]["EC_LARGE_NUMBER_EXPRESSION"] = {"ZH_BAIWAN_DECIMAL": 1}
    assert any("ZH_BAIWAN_DECIMAL is not a READY_V1 variant" in e for e in validate_stage0(excluded, SCHEMAS))


def test_chinese_product_names_are_approved_unique_and_pure_han(docs):
    names = docs["normalization"]["profiles"]["synthetic_product_name"]["language_names"]
    pool = [p["name"] for p in docs["generation_plan"]["extraction_distribution"]["product_pool"]]
    assert list(names) == pool
    surfaces = [entry["zh"]["surface"] for entry in names.values()]
    assert len(set(surfaces)) == len(surfaces) and all(entry["zh"]["status"] == "APPROVED" for entry in names.values())
    pending = deepcopy(docs)
    pending["normalization"]["profiles"]["synthetic_product_name"]["language_names"][pool[0]]["zh"]["status"] = "PENDING"
    assert any("needs an APPROVED zh name" in e for e in validate_stage0(pending, SCHEMAS))
    latin = deepcopy(docs)
    latin["normalization"]["profiles"]["synthetic_product_name"]["language_names"][pool[0]]["zh"]["surface"] = "猎户座ETF"
    assert any("must be Han script" in e for e in validate_stage0(latin, SCHEMAS))


def test_currency_generation_surfaces_exclude_bare_yuan_and_yen(docs):
    currency = docs["normalization"]["profiles"]["currency_iso_4217_alpha3"]
    generated = {text for forms in currency["generation_surfaces"].values() for texts in forms.values() for text in texts}
    assert {"yuan", "yen", "EUR", "JPY", "CNY"}.isdisjoint(generated)
    assert {"Chinese yuan", "Japanese yen", "RMB", "American dollars", "新币", "美金"} <= generated
    assert {"yuan", "yen"} <= set(currency["aliases"]["CNY"]) | set(currency["aliases"]["JPY"])


def test_terminology_labels_are_approved(docs):
    zh = docs["language_profiles"]["terminology"]["zh"]
    assert zh["NRIC"]["formal"] == "新加坡国民身份证号码" and zh["NRIC"]["ordinary"] == ["身份证号码"]
    assert zh["FIN"]["formal"] == "外籍人士身份证号码"
    assert all(entry["status"] == "APPROVED" for entry in zh.values())


# ---------------------------------------------------------------------------------------------- G1
def test_plan_reproduces_the_committed_artifacts(generated):
    texts, _ = generated
    assert outputs.check_outputs(RELEASE, texts) == []
    assert outputs.validate_written(RELEASE, SCHEMAS) == []


def test_seed_changes_the_plan(docs):
    changed = deepcopy(docs)
    changed["generation_plan"]["random_seed"] = 43
    first, second = build_plan(docs), build_plan(changed)
    assert first["gold"] != second["gold"]


def test_profile_counts_are_exact(docs, generated):
    _, summary = generated
    assert summary["language_profiles"] == {"ENGLISH": 150, "MANDARIN": 150, "EN_ZH_MIXED": 100, "SINGLISH": 100}
    assert summary["total_compliance_fail_pairs"] == 1120 and summary["total_edge_case_assignments"] == 1120
    for p in docs["generation_plan"]["language_profile_plans"]:
        block = summary["by_language_profile"][p["language_profile"]]
        assert block["duration_profiles"] == p["duration_counts"]
        assert block["call_composition"] == p["call_composition_counts"]
        assert block["fails_per_call"] == p["fail_count_per_call_counts"]
        assert block["edge_cases_per_call"] == p["edge_cases_per_call_counts"]
        assert {q: v["FAIL"] for q, v in block["compliance_verdicts"].items()} == p["fail_counts_per_check"]
        assert block["edge_case_ids"] == p["edge_case_counts"]
        assert block["language_variants"] == {k: dict(sorted(v.items())) for k, v in p["language_variant_counts"].items()}
        assert block["transaction_currency"] == p["transaction_currency_counts"]
        assert block["mixing_styles"] == dict(sorted((p.get("mixing_style_counts") or {"NONE": p["num_calls"]}).items()))


def test_planning_rules_hold_for_every_call(docs, plan_view):
    gold, by_call, profile, rows = plan_view
    domain = Domain(docs)
    for call, row in gold.items():
        cases = [a["edge_case_id"] for a in by_call[call]]
        fails = sum(row[q] == "FAIL" for q in COMPLIANCE)
        assert (row["C03"] == "FAIL" and row["C02"] == "PASS") == ("EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED" in cases)
        assert not (row["C02"] == "FAIL" and row["C03"] == "PASS")
        assert sum(row[q] == "FAIL" for q in PROHIBITION) <= 1
        if rows[call]["duration_profile"] == "SHORT":
            assert fails + len(cases) <= 6
        if "EC_COMPLETE_ORDER_REPLACEMENT" in cases:
            assert "EC_LONG_DISTANCE_DEPENDENCY" not in cases
            for a in by_call[call]:
                if a["family"] in {"CORRECTION_SUPERSESSION", "REFERENCE_RESOLUTION"} and a["edge_case_id"] != "EC_COMPLETE_ORDER_REPLACEMENT":
                    assert not set(a["target_question_ids"]) & set(domain.order_ids)
    for name in PROFILES:
        exposed = {c for c, rows_ in by_call.items() if profile[c] == name for a in rows_ if "T03" in a["target_question_ids"]}
        assert len(exposed) <= plan_of(docs, name)["target_exposure_caps"]["T03"]


def test_mandarin_magnitude_contrasts_are_exact_and_labelled(docs, plan_view):
    gold, by_call, profile, _ = plan_view
    domain = Domain(docs)
    labelled = 0
    for call, rows in by_call.items():
        for a in rows:
            if a["edge_case_id"] != "EC_MULTIPLE_MONETARY_VALUES" or profile[call] != "MANDARIN":
                continue
            final, currency = gold[call]["T03"], gold[call]["T02"]
            partners = set(domain.magnitude_partners(currency, final, 10))
            hits = [d for d in a["parameters"]["distractor_values"] if d["amount"] in partners]
            if a["language_variant"] == "ZH_MAGNITUDE_CONTRAST":
                labelled += 1
                assert len(hits) == 1
                assert hits[0]["amount"] > final or hits[0]["role"] in {"PREVIOUS_INVESTMENT_AMOUNT", "ILLUSTRATIVE_AMOUNT"}
            else:
                assert hits == []
    assert labelled == 12


def test_spoken_numeric_cases_never_reach_mandarin_and_carry_en_surface_in_mixed(plan_view):
    _, by_call, profile, _ = plan_view
    numeric = {"EC_DOUBLE_TRIPLE_DIGITS", "EC_OH_VS_ZERO", "EC_FINANCIAL_SHORTHAND", "EC_LARGE_NUMBER_EXPRESSION"}
    for call, rows in by_call.items():
        for a in rows:
            if a["edge_case_id"] in numeric:
                assert profile[call] != "MANDARIN"
                assert a["language_variant"] == ("EN_SURFACE" if profile[call] == "EN_ZH_MIXED" else "NONE")
            if a["language_variant"].startswith("CROSS_LANGUAGE"):
                assert profile[call] == "EN_ZH_MIXED"


def test_k01_and_k02_are_unique_release_wide(plan_view):
    gold = plan_view[0]
    for q in ("K01", "K02"):
        assert len({row[q] for row in gold.values()}) == len(gold)


@pytest.mark.parametrize("mutation", ["swap_profile", "variant", "d3", "prohibition"])
def test_validator_rejects_tampered_outputs(docs, parsed, mutation):
    manifest, matrix, assignments = deepcopy(parsed)
    if mutation == "swap_profile":
        a, b = next(r for r in manifest if r["language_profile"] == "ENGLISH"), next(r for r in manifest if r["language_profile"] == "MANDARIN")
        a["language_profile"], b["language_profile"] = b["language_profile"], a["language_profile"]
    elif mutation == "variant":
        row = next(a for a in assignments if a["language_variant"] == "NONE" and a["edge_case_id"] == "EC_CORRECTION_SAME_TURN")
        row["language_variant"] = "CROSS_LANGUAGE_EN_TO_ZH"
    elif mutation == "d3":
        call = next(c for c, cells in matrix.items() if cells["C02"] == "PASS" and cells["C03"] == "PASS")
        matrix[call]["C03"] = "FAIL"
    else:
        call = next(c for c, cells in matrix.items() if cells["C09"] == "FAIL" and cells["C10"] == "PASS")
        matrix[call]["C10"] = "FAIL"
    assert validate_plan(docs, manifest, matrix, assignments, SCHEMAS)


# ---------------------------------------------------------------------------------------------- G2A / G2B
@pytest.fixture(scope="module")
def g2a_sources():
    return g2a.load_g1_sources(RELEASE, SCHEMAS)


def test_call_specs_validate_and_freeze_language_requirements(docs, g2a_sources):
    disk = read_g2a_disk(RELEASE)
    assert validate_spec_set(g2a_sources, disk, SCHEMAS) == []
    profiles = docs["language_profiles"]["profiles"]
    for call_id in g2a_sources.call_ids:
        spec = json.loads(disk[g2a.spec_path(call_id)])
        name = spec["metadata"]["language_profile"]
        assert spec["schema_version"] == "2.0"
        assert spec["language_requirements"]["speakers"] == profiles[name]["speakers"]
        assert spec["source"]["language_profiles_sha256"] == g2a_sources.hashes["language_profiles_sha256"]
    tampered = dict(disk)
    spec = json.loads(disk[g2a.spec_path("CALL_001")])
    spec["language_requirements"]["value_surface_languages"] = ["en"]
    tampered[g2a.spec_path("CALL_001")] = g2a.serialize_json(spec)
    assert any("language_requirements differ" in e for e in validate_spec_set(g2a_sources, tampered, SCHEMAS))


@pytest.fixture(scope="module")
def skeleton_inputs():
    return g2b_inputs.load_inputs(RELEASE, SCHEMAS)


def test_skeletons_validate_with_language_filtered_forms_and_exact_pins(skeleton_inputs):
    disk = read_g2b_disk(RELEASE)
    assert validate_skeleton_set(skeleton_inputs, disk, SCHEMAS) == []
    expected_forms = {"MANDARIN": {"ZH_NATURAL_NAME"}, "ENGLISH": {"NATURAL_NAME", "SPELLED_CODE"},
                      "SINGLISH": {"NATURAL_NAME", "SPELLED_CODE"}, "EN_ZH_MIXED": {"NATURAL_NAME", "SPELLED_CODE", "ZH_NATURAL_NAME"}}
    for call_id in skeleton_inputs.call_ids:
        spec = skeleton_inputs.specs[call_id]
        sk = json.loads(disk[skeleton_path(call_id)])
        t02 = next(c for c in sk["checks"] if c["question_id"] == "T02")
        assert set(t02["non_gold_spoken_form"]["forms"]) == expected_forms[spec["metadata"]["language_profile"]]
        variants = [a for a in spec["edge_cases"] if a["language_variant"] != "NONE"]
        pinned = [e for e in sk["planned_events"] if "required_language" in e]
        assert bool(pinned) == any(a["language_variant"] != "ZH_MAGNITUDE_CONTRAST" for a in variants)


def test_skeleton_validator_rejects_a_moved_pin(skeleton_inputs):
    disk = dict(read_g2b_disk(RELEASE))
    call_id = next(c for c in skeleton_inputs.call_ids
                   if any(a["language_variant"].startswith("CROSS_LANGUAGE") for a in skeleton_inputs.specs[c]["edge_cases"]))
    sk = json.loads(disk[skeleton_path(call_id)])
    event = next(e for e in sk["planned_events"] if "required_language" in e)
    event["required_language"] = "zh" if event["required_language"] == "en" else "en"
    disk[skeleton_path(call_id)] = g2a.serialize_json(sk)
    assert any("required_language pins" in e for e in validate_skeleton_set(skeleton_inputs, disk, SCHEMAS))


def test_g3_refuses_a_multilingual_release_until_language_aware_briefs_exist():
    with pytest.raises(G2BStateError, match="multilingual release"):
        load_g3_inputs(RELEASE, SCHEMAS)


# ---------------------------------------------------------------------------------------------- reports
def test_reports_are_derived_from_the_maintained_contracts(docs):
    compatibility = compatibility_report(RELEASE)
    for case in docs["edge_case_catalog"]["edge_cases"]:
        assert f"| {case['id']} |" in compatibility
    assert "ZH_BAIWAN_DECIMAL (MANDARIN; EXCLUDED_V1)" in compatibility
    composition = composition_report(RELEASE)
    assert "| C03 | 39 | 39 | 26 | 26 | 130 |" in composition
    assert "| EC_MULTIPLE_MONETARY_VALUES | ZH_MAGNITUDE_CONTRAST | 0 | 12 | 0 | 0 | 12 |" in composition
