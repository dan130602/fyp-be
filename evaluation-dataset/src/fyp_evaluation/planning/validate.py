"""Independent G1 output validation.

Re-derives every approved G0/G1 constraint from the three tabular outputs
(manifest rows, raw coverage-matrix cells and edge-case assignment rows). It
does not reuse allocator state, so it also validates files read back from disk.
"""

from collections import Counter, defaultdict

from fyp_evaluation.authoring.coverage import check_matrix, compile_expected
from fyp_evaluation.validation.schemas import validate_checklist
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR, is_multilingual, numeric_prerequisite_met

from .adapter import legacy_normalization
from .domain import (
    BALANCE_ROLES, CURRENCY_DISTRACTOR_ROLES, FEE_ROLES, MONETARY_DISTRACTOR_ROLES, Domain,
    financial_shorthand, magnitude_expression, maximal_digit_runs, parse_magnitude,
)
from .edge_params import (
    FEE_DISCLOSURE_QUESTION, GOLD_PARAMETER, K01_VARIANT_RUNS, SUPERSEDED_PARAMETER, magnitude_contrast,
)
from .multilingual import (
    NO_MIXING_STYLE, NO_VARIANT, contrast_factors_avoided, separates_order_finals, supplies_final, variant_pins,
)


def typed_gold(documents, matrix, call_ids, schema_dir=DEFAULT_SCHEMA_DIR):
    """Validate raw matrix cells through the existing tooling; return (gold, errors)."""
    checklist = documents["checklist"]
    normalization = legacy_normalization(checklist, documents["check_semantics"], documents["normalization"])
    errors = [f"adapter: {e}" for e in validate_checklist(checklist, normalization, schema_dir)]
    if errors:
        return None, errors
    try:
        check_matrix(matrix, checklist, call_ids=call_ids)
    except ValueError as error:
        return None, [f"coverage_matrix: {error}"]
    gold = {}
    for call_id in call_ids:
        gold[call_id] = {}
        for item in checklist:
            try:
                expected = compile_expected(item, normalization, matrix[call_id][item["id"]])
            except ValueError as error:
                errors.append(f"coverage_matrix: {call_id}/{item['id']}: {error}")
                continue
            gold[call_id][item["id"]] = expected.get("expected_verdict", expected.get("expected_value"))
    return gold, errors


def validate_plan(documents, manifest, matrix, assignments, schema_dir=DEFAULT_SCHEMA_DIR):
    """Return every constraint violation found in the G1 outputs ([] when valid)."""
    if is_multilingual(documents):
        return _validate_multilingual(documents, manifest, matrix, assignments, schema_dir)
    plan = documents["generation_plan"]
    domain = Domain(documents)
    cases = {case["id"]: case for case in documents["edge_case_catalog"]["edge_cases"]}
    case_index = {case_id: index for index, case_id in enumerate(cases)}
    question_index = {item["id"]: index for index, item in enumerate(documents["checklist"])}
    rules = plan["planning_rules"]
    exempt = set(rules["per_question_limit_exempt_edge_cases"])
    errors = []

    # Manifest ----------------------------------------------------------------
    call_ids = [row["call_id"] for row in manifest]
    if call_ids != [f"CALL_{n:03d}" for n in range(1, plan["num_calls"] + 1)]:
        errors.append(f"call_manifest: call IDs must be CALL_001..CALL_{plan['num_calls']:03d} in order")
        return errors
    vocabulary = plan["call_intent_vocabulary"]
    durations = {d["profile"]: d for d in plan["duration_profiles"]}
    for row in manifest:
        intents = row["call_intents"]
        if any(i not in vocabulary for i in intents) or intents != sorted(set(intents), key=vocabulary.index):
            errors.append(f"call_manifest: {row['call_id']}: call_intents must be unique vocabulary terms in vocabulary order")
        profile = durations.get(row["duration_profile"])
        if profile is None or (row["target_duration_min_seconds"], row["target_duration_max_seconds"]) != (
                profile["min_seconds"], profile["max_seconds"]):
            errors.append(f"call_manifest: {row['call_id']}: duration bounds must match the configured profile")
    expected_counts = [
        ("language", Counter(r["language"] for r in manifest), Counter(plan["language_counts"])),
        ("duration", Counter(r["duration_profile"] for r in manifest),
         Counter({d["profile"]: d["count"] for d in plan["duration_profiles"]})),
        ("call intent set", Counter(tuple(r["call_intents"]) for r in manifest),
         Counter({tuple(s["call_intents"]): s["count"] for s in plan["call_intent_set_counts"]})),
    ]

    # Coverage matrix through the existing loader/compiler --------------------
    gold, matrix_errors = typed_gold(documents, matrix, call_ids, schema_dir)
    errors += matrix_errors
    if gold is None or matrix_errors:
        return errors
    extraction = plan["extraction_distribution"]
    expected_counts += [
        ("transaction direction", Counter(gold[c][domain.t05] for c in call_ids),
         Counter(extraction["transaction_direction_counts"])),
        ("order type", Counter(gold[c][domain.t06] for c in call_ids), Counter(extraction["order_type_counts"])),
        ("currency", Counter(gold[c][domain.t02] for c in call_ids), Counter(extraction["transaction_currency_counts"])),
    ]
    for c in call_ids:
        row = gold[c]
        if not domain.valid_k01(row[domain.k01]):
            errors.append(f"coverage_matrix: {c}/{domain.k01}: not a configured checksum-invalid identifier")
        if not domain.valid_k02(row[domain.k02]):
            errors.append(f"coverage_matrix: {c}/{domain.k02}: DOB outside the configured range")
        if not domain.plausible_order({q: row[q] for q in domain.order_ids}):
            errors.append(f"coverage_matrix: {c}: product/currency/amount/quantity is not a plausible configured order")
    for q in (domain.k01, domain.k02):
        if len({gold[c][q] for c in call_ids}) != len(call_ids):
            errors.append(f"coverage_matrix: {q} values must be unique across calls")

    # Compliance ----------------------------------------------------------------
    fail_count = {c: sum(gold[c][q] == "FAIL" for q in domain.compliance_ids) for c in call_ids}
    expected_counts.append((
        "FAILs per call", Counter(fail_count.values()),
        Counter({int(n): count for n, count in plan["compliance_distribution"]["fail_count_per_call_counts"].items()}),
    ))
    for c in call_ids:
        for rule in documents["check_semantics"]["cross_check_rules"]:
            if gold[c][rule["if"]["question_id"]] == rule["if"]["verdict"]:
                for implied in rule["then"]:
                    if gold[c][implied["question_id"]] != implied["verdict"]:
                        errors.append(f"coverage_matrix: {c}: violates {rule['id']}")
    for q in domain.compliance_ids:
        fails = sum(gold[c][q] == "FAIL" for c in call_ids)
        if fails < plan["compliance_distribution"]["min_fail_cases_per_check"]:
            errors.append(f"coverage_matrix: {q} has only {fails} FAIL cases")

    # Edge-case assignments ---------------------------------------------------------
    by_call = defaultdict(list)
    ids = [a["assignment_id"] for a in assignments]
    if ids != [f"EA_{n:03d}" for n in range(1, len(assignments) + 1)]:
        errors.append("edge_case_assignments: assignment IDs must be EA_001.. in file order")
    order_key = [(a["call_id"], case_index.get(a["edge_case_id"], -1),
                  [question_index.get(q, -1) for q in a["target_question_ids"]]) for a in assignments]
    if order_key != sorted(order_key):
        errors.append("edge_case_assignments: rows must be sorted by call, catalog order and target")
    for a in assignments:
        if a["call_id"] not in gold:
            errors.append(f"edge_case_assignments: {a['assignment_id']}: unknown call {a['call_id']}")
            continue
        by_call[a["call_id"]].append(a)
        errors += [f"edge_case_assignments: {a['assignment_id']} ({a['edge_case_id']}): {e}"
                   for e in _assignment_errors(a, cases, gold, domain, exempt, question_index)]
    family_counts = Counter(a["family"] for a in assignments if a["edge_case_id"] in cases)
    if family_counts != Counter({f: q for f, q in plan["edge_case_distribution"]["family_quotas"].items() if q}):
        errors.append(f"edge_case_assignments: family counts {dict(family_counts)} differ from family_quotas")
    case_counts = Counter(a["edge_case_id"] for a in assignments)
    for family, quota in plan["edge_case_distribution"]["family_quotas"].items():
        members = [cid for cid, case in cases.items() if case["family"] == family]
        base, remainder = divmod(quota, len(members))
        for position, case_id in enumerate(members):
            if case_counts[case_id] != base + (position < remainder):
                errors.append(f"edge_case_assignments: {case_id} count {case_counts[case_id]} is not the balanced quota share")
    expected_counts.append((
        "edge cases per call", Counter(len(by_call[c]) for c in call_ids),
        Counter({int(n): count for n, count in plan["edge_case_distribution"]["edge_cases_per_call_counts"].items()}),
    ))
    for c in call_ids:
        rows = by_call[c]
        targeted = Counter(q for a in rows if a["edge_case_id"] not in exempt for q in a["target_question_ids"])
        if over := sorted(q for q, n in targeted.items() if n > rules["max_edge_cases_per_target_question_per_call"]):
            errors.append(f"edge_case_assignments: {c}: more than one edge case targets {over}")
        present = {a["edge_case_id"] for a in rows}
        for case_id in present:
            if set(cases.get(case_id, {}).get("mutually_exclusive_with", [])) & present:
                errors.append(f"edge_case_assignments: {c}: {case_id} is mutually exclusive with another assigned case")
        if any(cases[a["edge_case_id"]].get("target_scope") == "TRANSACTION_BUNDLE" for a in rows if a["edge_case_id"] in cases):
            for a in rows:
                case = cases.get(a["edge_case_id"], {})
                if (case.get("family") == "CORRECTION_SUPERSESSION" and case.get("target_scope") != "TRANSACTION_BUNDLE"
                        and set(a["target_question_ids"]) & set(domain.order_ids)):
                    errors.append(f"edge_case_assignments: {c}: correction on a transaction field alongside order replacement")

    composition = Counter(
        ("all_pass" if not fail_count[c] else "some_fail") + ("_with_edge_cases" if by_call[c] else "_without_edge_cases")
        for c in call_ids
    )
    expected_counts.append(("call composition", composition, Counter(plan["call_composition_counts"])))
    for label, actual, expected in expected_counts:
        if +actual != +expected:
            errors.append(f"{label} counts {dict(actual)} differ from the plan {dict(expected)}")
    return errors


def _assignment_errors(a, cases, gold, domain, exempt, question_index):
    case = cases.get(a["edge_case_id"])
    if case is None:
        return ["unknown edge case"]
    errors = []
    row = gold[a["call_id"]]
    targets, params = a["target_question_ids"], a["parameters"]
    if a["family"] != case["family"]:
        errors.append("family differs from the catalog")
    if not targets or len(set(targets)) != len(targets) or set(targets) - set(case["allowed_target_question_ids"]):
        errors.append(f"illegal targets {targets}")
        return errors
    if targets != sorted(targets, key=question_index.get):
        errors.append("targets must follow checklist order")
    scope = case.get("target_scope")
    if scope is None and len(targets) != 1:
        errors.append("single-question edge case must have exactly one target")
    if scope in {"TRANSACTION_BUNDLE", "MULTI_QUESTION"} and len(targets) < 2:
        errors.append("multi-question edge case needs at least two targets")
    expected_keys = set(case["required_parameters"])
    if set(params) != expected_keys:
        errors.append(f"parameters {sorted(params)} must be exactly {sorted(expected_keys)}")
        return errors
    for question_id, verdict in case.get("verdict_requirements", {}).items():
        if row[question_id] != verdict:
            errors.append(f"requires {question_id} {verdict}")
    if "target_verdict_requirement" in case and row[targets[0]] != case["target_verdict_requirement"]:
        errors.append(f"requires target {targets[0]} {case['target_verdict_requirement']}")
    prerequisite = case.get("gold_prerequisites")
    if prerequisite:
        for target in targets:
            value = row[target]
            if prerequisite["kind"] == "CONTAINS_CHARACTER":
                ok = prerequisite.get("character", "") in value[1:-1]
            elif prerequisite["kind"] == "REPEATED_DIGIT_RUN":
                ok = max(maximal_digit_runs(value[1:-1])) >= prerequisite.get("minimum_run_length", 2)
            else:
                ok = numeric_prerequisite_met(prerequisite, value)
            if not ok:
                errors.append(f"{target} gold {value!r} fails {prerequisite['kind']}")
    if "antecedent_speaker" in params and params["antecedent_speaker"] not in case["allowed_antecedent_speakers"]:
        errors.append("antecedent speaker not allowed")
    gold_key = GOLD_PARAMETER.get(case["id"])
    if gold_key is not None and params[gold_key] != row[targets[0]]:
        errors.append(f"{gold_key} {params[gold_key]!r} differs from coverage-matrix gold {row[targets[0]]!r}")
    superseded_key = SUPERSEDED_PARAMETER.get(case["id"])
    if superseded_key is not None:
        values = params[superseded_key] if isinstance(params[superseded_key], list) else [params[superseded_key]]
        excluded = {targets[0]: frozenset(gold[c][targets[0]] for c in gold)} if targets[0] in {domain.k01, domain.k02} else {}
        legal = domain.alternatives(targets[0], row, excluded.get(targets[0], frozenset()))
        if superseded_key == "superseded_values" and len(values) < 2:
            errors.append("needs at least two superseded values")
        if len(set(map(repr, values))) != len(values) or any(v not in legal for v in values):
            errors.append(f"{superseded_key} {values!r} must be distinct, differ from gold and stay realistic")
    builder_checks = {
        "EC_COMPLETE_ORDER_REPLACEMENT": _order_replacement_errors,
        "EC_MULTIPLE_CURRENCIES": _currency_distractor_errors,
        "EC_MULTIPLE_MONETARY_VALUES": _monetary_distractor_errors,
        "EC_PRODUCT_REFERENCE": _product_reference_errors,
        "EC_DOUBLE_TRIPLE_DIGITS": _variant_errors,
        "EC_FINANCIAL_SHORTHAND": lambda *args: _spoken_errors(financial_shorthand, *args),
        "EC_LARGE_NUMBER_EXPRESSION": lambda *args: _spoken_errors(magnitude_expression, *args),
        "EC_LONG_DISTANCE_DEPENDENCY": _long_distance_errors,
    }
    if check := builder_checks.get(case["id"]):
        errors += check(case, targets, params, row, domain)
    return errors


def _order_replacement_errors(case, targets, params, row, domain):
    errors = []
    final, initial = params["final_order"], params["initial_order"]
    if not isinstance(final, dict) or not isinstance(initial, dict) or set(final) != set(domain.order_ids) or set(initial) != set(domain.order_ids):
        return ["orders must contain exactly the transaction fields"]
    if any(final[q] != row[q] for q in domain.order_ids):
        errors.append("final_order differs from coverage-matrix gold")
    if not domain.plausible_order(initial):
        errors.append("initial_order is not a plausible configured order")
    if [q for q in domain.order_ids if initial[q] != final[q]] != targets:
        errors.append("targets must be exactly the changed transaction fields")
    return errors


def _currency_distractor_errors(case, targets, params, row, domain):
    errors = []
    if params["distractor_currency"] not in domain.currencies or params["distractor_currency"] == row[domain.t02]:
        errors.append("distractor_currency must be another configured currency")
    if params["distractor_role"] not in CURRENCY_DISTRACTOR_ROLES:
        errors.append("unknown distractor_role")
    elif params["distractor_role"] in FEE_ROLES and row[FEE_DISCLOSURE_QUESTION] == "FAIL":
        errors.append("fee distractor is not allowed when C04 is FAIL")
    return errors


def _monetary_distractor_errors(case, targets, params, row, domain):
    errors = []
    distractors = params["distractor_values"]
    final, currency = row[domain.t03], row[domain.t02]
    if not isinstance(distractors, list) or len(distractors) < 2:
        return ["needs at least two distractor values"]
    roles = [d.get("role") for d in distractors]
    amounts = [d.get("amount") for d in distractors]
    if len(set(roles)) != len(roles) or len(set(amounts)) != len(amounts):
        errors.append("distractor roles and amounts must be distinct")
    for d in distractors:
        if set(d) != {"amount", "currency", "role"}:
            errors.append("distractor values need exactly amount, currency and role")
            continue
        if d["role"] in FEE_ROLES and row[FEE_DISCLOSURE_QUESTION] == "FAIL":
            errors.append("fee distractor is not allowed when C04 is FAIL")
        elif d["role"] not in MONETARY_DISTRACTOR_ROLES:
            errors.append(f"unknown monetary distractor role {d['role']}")
        if d["currency"] != currency or d["amount"] not in domain.amounts[currency]:
            errors.append("distractor amount must be a configured amount in the transaction currency")
        if d["amount"] == final or (d["role"] in BALANCE_ROLES and not d["amount"] > final):
            errors.append(f"{d['role']} amount {d['amount']} is not realistic against {final}")
    return errors


def _product_reference_errors(case, targets, params, row, domain):
    candidates = params["candidate_products"]
    if (not isinstance(candidates, list) or len(candidates) < 2 or len(set(candidates)) != len(candidates)
            or row[domain.t01] not in candidates
            or any(row[domain.t02] not in domain.products.get(p, {}).get("allowed_currencies", []) for p in candidates)):
        return ["candidate_products must be distinct configured products allowing the currency, including the selection"]
    return []


def _variant_errors(case, targets, params, row, domain):
    run = K01_VARIANT_RUNS.get(params["variant"])
    if run is None or run not in maximal_digit_runs(row[domain.k01][1:-1]):
        return [f"variant {params['variant']!r} has no matching digit run in K01"]
    return []


def _spoken_errors(formatter, case, targets, params, row, domain):
    spoken, value = params["spoken_form"], params["canonical_value"]
    if spoken != formatter(value) or parse_magnitude(spoken, domain.magnitude_words) != value:
        return [f"spoken_form {spoken!r} does not denote {value}"]
    return []


def _long_distance_errors(case, targets, params, row, domain):
    errors = []
    if params["target_question_ids"] != targets:
        errors.append("target_question_ids must equal the assignment targets")
    if params["min_turn_gap"] != case["plan_template"]["default_min_turn_gap"]:
        errors.append("min_turn_gap must equal the catalog default")
    return errors


def _call_rules(c, rows, cases, domain, exempt, rules):
    """Per-call edge-case rules shared with the 1.0 release: one case per target, mutual exclusion, bundle exclusion."""
    errors = []
    targeted = Counter(q for a in rows if a["edge_case_id"] not in exempt for q in a["target_question_ids"])
    if over := sorted(q for q, n in targeted.items() if n > rules["max_edge_cases_per_target_question_per_call"]):
        errors.append(f"edge_case_assignments: {c}: more than one edge case targets {over}")
    present = {a["edge_case_id"] for a in rows}
    for case_id in present:
        if set(cases.get(case_id, {}).get("mutually_exclusive_with", [])) & present:
            errors.append(f"edge_case_assignments: {c}: {case_id} is mutually exclusive with another assigned case")
    if any(cases[a["edge_case_id"]].get("target_scope") == "TRANSACTION_BUNDLE" for a in rows if a["edge_case_id"] in cases):
        for a in rows:
            case = cases.get(a["edge_case_id"], {})
            if case and supplies_final(case) and set(a["target_question_ids"]) & set(domain.order_ids):
                errors.append(f"edge_case_assignments: {c}: {a['edge_case_id']} supplies a transaction FINAL alongside order replacement")
            if case and separates_order_finals(case, set(domain.order_ids)):
                errors.append(f"edge_case_assignments: {c}: {a['edge_case_id']} needs distinct transaction FINAL events, "
                              "which an order replacement merges into one")
    return errors


def _variant_errors_v2(a, case, profile, row, domain, avoided):
    """A language variant is NONE or a READY_V1 catalog variant of the case for this profile; contrasts are exact."""
    errors = []
    variant_id = a["language_variant"]
    variant = None
    if variant_id != NO_VARIANT:
        variant = next((v for v in case.get("language_variants", []) if v["variant_id"] == variant_id), None)
        if variant is None or variant["status"] != "READY_V1" or profile not in variant["profiles"]:
            return [f"language variant {variant_id} is not a READY_V1 variant of {case['id']} for {profile}"]
    if case["id"] != "EC_MULTIPLE_MONETARY_VALUES" or not isinstance(a["parameters"].get("distractor_values"), list):
        return errors
    contrast = magnitude_contrast(variant)
    factor = contrast["factor"] if contrast else (None if variant else avoided.get(case["id"]))
    if factor is None:
        return errors
    final, currency = row[domain.t03], row[domain.t02]
    partners = set(domain.magnitude_partners(currency, final, factor))
    hits = [d for d in a["parameters"]["distractor_values"] if d.get("amount") in partners]
    if contrast:
        if len(hits) != contrast["distractor_count"]:
            errors.append(f"{variant_id} needs exactly {contrast['distractor_count']} x{factor} distractor, found {len(hits)}")
        for d in hits:
            if d["amount"] < final and d["role"] not in contrast["divided_distractor_roles"]:
                errors.append(f"{variant_id}: a divided distractor may not take role {d['role']}")
    elif hits:
        errors.append(f"an unlabelled x{factor} monetary distractor; every contrast in {profile} must carry its language variant")
    return errors


def _validate_multilingual(documents, manifest, matrix, assignments, schema_dir):
    """Independent validation of a multilingual (2.0) plan: every allocation is exact within its language profile."""
    plan = documents["generation_plan"]
    domain = Domain(documents)
    cases = {case["id"]: case for case in documents["edge_case_catalog"]["edge_cases"]}
    case_index = {case_id: index for index, case_id in enumerate(cases)}
    question_index = {item["id"]: index for index, item in enumerate(documents["checklist"])}
    rules = plan["planning_rules"]
    exempt = set(rules["per_question_limit_exempt_edge_cases"])
    language_profiles = documents["language_profiles"]["profiles"]
    plans = {p["language_profile"]: p for p in plan["language_profile_plans"]}
    errors = []

    # Manifest -----------------------------------------------------------------------------------------
    if any("language_profile" not in row for row in manifest):
        return ["call_manifest: a multilingual release needs the language_profile/mixing_style manifest layout"]
    call_ids = [row["call_id"] for row in manifest]
    if call_ids != [f"CALL_{n:03d}" for n in range(1, plan["num_calls"] + 1)]:
        return [f"call_manifest: call IDs must be CALL_001..CALL_{plan['num_calls']:03d} in order"]
    vocabulary = plan["call_intent_vocabulary"]
    durations = {d["profile"]: d for d in plan["duration_profiles"]}
    profile_of = {}
    for row in manifest:
        c, name = row["call_id"], row["language_profile"]
        intents = row["call_intents"]
        if any(i not in vocabulary for i in intents) or intents != sorted(set(intents), key=vocabulary.index):
            errors.append(f"call_manifest: {c}: call_intents must be unique vocabulary terms in vocabulary order")
        duration = durations.get(row["duration_profile"])
        if duration is None or (row["target_duration_min_seconds"], row["target_duration_max_seconds"]) != (
                duration["min_seconds"], duration["max_seconds"]):
            errors.append(f"call_manifest: {c}: duration bounds must match the configured profile")
        if name not in plans:
            errors.append(f"call_manifest: {c}: unknown language profile {name}")
            continue
        styles = language_profiles[name].get("mixing_styles")
        allowed = row["mixing_style"] in styles if styles else row["mixing_style"] == NO_MIXING_STYLE
        if not allowed:
            errors.append(f"call_manifest: {c}: mixing_style {row['mixing_style']} is not allowed for {name}")
        profile_of[c] = name
    if errors:
        return errors
    members = {name: [c for c in call_ids if profile_of[c] == name] for name in plans}
    expected_counts = [("language profile", Counter(profile_of.values()), Counter({n: p["num_calls"] for n, p in plans.items()}))]
    for name, p in plans.items():
        rows = [r for r in manifest if r["language_profile"] == name]
        expected_counts += [
            (f"{name} duration", Counter(r["duration_profile"] for r in rows), Counter(p["duration_counts"])),
            (f"{name} call intent set", Counter(tuple(r["call_intents"]) for r in rows),
             Counter({tuple(s["call_intents"]): s["count"] for s in p["call_intent_set_counts"]})),
            (f"{name} mixing style", Counter(r["mixing_style"] for r in rows),
             Counter(p.get("mixing_style_counts", {NO_MIXING_STYLE: p["num_calls"]}))),
        ]

    # Coverage matrix through the existing loader/compiler -------------------------------------------------
    gold, matrix_errors = typed_gold(documents, matrix, call_ids, schema_dir)
    errors += matrix_errors
    if gold is None or matrix_errors:
        return errors
    for name, p in plans.items():
        ids = members[name]
        expected_counts += [
            (f"{name} transaction direction", Counter(gold[c][domain.t05] for c in ids), Counter(p["transaction_direction_counts"])),
            (f"{name} order type", Counter(gold[c][domain.t06] for c in ids), Counter(p["order_type_counts"])),
            (f"{name} currency", Counter(gold[c][domain.t02] for c in ids), Counter(p["transaction_currency_counts"])),
        ]
    for c in call_ids:
        row = gold[c]
        if not domain.valid_k01(row[domain.k01]):
            errors.append(f"coverage_matrix: {c}/{domain.k01}: not a configured checksum-invalid identifier")
        if not domain.valid_k02(row[domain.k02]):
            errors.append(f"coverage_matrix: {c}/{domain.k02}: DOB outside the configured range")
        if not domain.plausible_order({q: row[q] for q in domain.order_ids}):
            errors.append(f"coverage_matrix: {c}: product/currency/amount/quantity is not a plausible configured order")
    for q in (domain.k01, domain.k02):
        if len({gold[c][q] for c in call_ids}) != len(call_ids):
            errors.append(f"coverage_matrix: {q} values must be unique across calls")

    # Compliance -----------------------------------------------------------------------------------------
    fail_count = {c: sum(gold[c][q] == "FAIL" for q in domain.compliance_ids) for c in call_ids}
    prohibition = set(rules["prohibition_question_ids"])
    for c in call_ids:
        for rule in documents["check_semantics"]["cross_check_rules"]:
            if gold[c][rule["if"]["question_id"]] == rule["if"]["verdict"]:
                for implied in rule["then"]:
                    if gold[c][implied["question_id"]] != implied["verdict"]:
                        errors.append(f"coverage_matrix: {c}: violates {rule['id']}")
        if sum(gold[c][q] == "FAIL" for q in prohibition) > rules["max_prohibition_fails_per_call"]:
            errors.append(f"coverage_matrix: {c}: more than {rules['max_prohibition_fails_per_call']} prohibition FAILs")
    for name, p in plans.items():
        ids = members[name]
        expected_counts.append((f"{name} FAILs per call", Counter(fail_count[c] for c in ids),
                                Counter({int(n): count for n, count in p["fail_count_per_call_counts"].items()})))
        expected_counts.append((f"{name} FAILs per question",
                                Counter({q: sum(gold[c][q] == "FAIL" for c in ids) for q in domain.compliance_ids}),
                                Counter(p["fail_counts_per_check"])))

    # Edge-case assignments --------------------------------------------------------------------------------
    if any("language_variant" not in a for a in assignments):
        return errors + ["edge_case_assignments: a multilingual release needs the language_variant column"]
    by_call = defaultdict(list)
    ids = [a["assignment_id"] for a in assignments]
    if ids != [f"EA_{n:03d}" for n in range(1, len(assignments) + 1)]:
        errors.append("edge_case_assignments: assignment IDs must be EA_001.. in file order")
    order_key = [(a["call_id"], case_index.get(a["edge_case_id"], -1),
                  [question_index.get(q, -1) for q in a["target_question_ids"]]) for a in assignments]
    if order_key != sorted(order_key):
        errors.append("edge_case_assignments: rows must be sorted by call, catalog order and target")
    avoided = {name: contrast_factors_avoided(cases, name) for name in plans}
    for a in assignments:
        if a["call_id"] not in gold:
            errors.append(f"edge_case_assignments: {a['assignment_id']}: unknown call {a['call_id']}")
            continue
        by_call[a["call_id"]].append(a)
        label = f"edge_case_assignments: {a['assignment_id']} ({a['edge_case_id']})"
        problems = _assignment_errors(a, cases, gold, domain, exempt, question_index)
        if not problems and a["edge_case_id"] in cases:
            name = profile_of[a["call_id"]]
            if name not in cases[a["edge_case_id"]].get("language_profiles", []):
                problems.append(f"not eligible for language profile {name}")
            problems += _variant_errors_v2(a, cases[a["edge_case_id"]], name, gold[a["call_id"]], domain, avoided[name])
        errors += [f"{label}: {e}" for e in problems]
    order_fields = set(domain.order_ids)
    capped, cap = rules["load_cap"]["duration_profile"], rules["load_cap"]["max_fails_plus_edge_cases"]
    duration_of = {r["call_id"]: r["duration_profile"] for r in manifest}
    for c in call_ids:
        rows = by_call[c]
        errors += _call_rules(c, rows, cases, domain, exempt, rules)
        has_bundle = any(a["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT" for a in rows)
        pins = {}
        for a in rows:
            variant = next((v for v in cases.get(a["edge_case_id"], {}).get("language_variants", [])
                            if v["variant_id"] == a["language_variant"]), None)
            for key, language in variant_pins(a["edge_case_id"], a["target_question_ids"], variant, has_bundle, order_fields).items():
                if pins.setdefault(key, language) != language:
                    errors.append(f"edge_case_assignments: {c}: language variants pin {key} to two languages")
        if duration_of[c] == capped and fail_count[c] + len(rows) > cap:
            errors.append(f"call_manifest: {c}: {fail_count[c]} FAILs plus {len(rows)} edge cases exceed the {capped} load cap {cap}")
        for rule in rules["dependent_fail_rules"]:
            q, failed = rule["question_id"], rule["fails_only_with"]["failed_question_id"]
            case_id = rule["fails_only_with"]["or_edge_case_id"]
            licensed = any(a["edge_case_id"] == case_id for a in rows)
            if (gold[c][q] == "FAIL" and gold[c][failed] == "PASS") != licensed:
                errors.append(f"coverage_matrix: {c}: {q} FAIL with {failed} PASS must occur exactly when {case_id} is assigned")
    for name, p in plans.items():
        rows = [a for c in members[name] for a in by_call[c]]
        expected_counts += [
            (f"{name} edge case", Counter(a["edge_case_id"] for a in rows), Counter(p["edge_case_counts"])),
            (f"{name} language variant",
             Counter((a["edge_case_id"], a["language_variant"]) for a in rows if a["language_variant"] != NO_VARIANT),
             Counter({(case_id, v): n for case_id, by in p["language_variant_counts"].items() for v, n in by.items()})),
            (f"{name} edge cases per call", Counter(len(by_call[c]) for c in members[name]),
             Counter({int(n): count for n, count in p["edge_cases_per_call_counts"].items()})),
            (f"{name} call composition", Counter(
                ("all_pass" if not fail_count[c] else "some_fail") + ("_with_edge_cases" if by_call[c] else "_without_edge_cases")
                for c in members[name]), Counter(p["call_composition_counts"])),
        ]
        for q, limit in p["target_exposure_caps"].items():
            exposed = len({a["call_id"] for a in rows if q in a["target_question_ids"]})
            if exposed > limit:
                errors.append(f"edge_case_assignments: {name}: {exposed} calls target {q}, above its exposure cap {limit}")
    for label, actual, expected in expected_counts:
        if +actual != +expected:
            errors.append(f"{label} counts {dict(actual)} differ from the plan {dict(expected)}")
    return errors
