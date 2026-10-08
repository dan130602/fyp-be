"""Independent G1 output validation.

Re-derives every approved G0/G1 constraint from the three tabular outputs
(manifest rows, raw coverage-matrix cells and edge-case assignment rows). It
does not reuse allocator state, so it also validates files read back from disk.
"""

from collections import Counter, defaultdict

from fyp_evaluation.authoring.coverage import check_matrix, compile_expected
from fyp_evaluation.validation.schemas import validate_checklist
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR, numeric_prerequisite_met

from .adapter import legacy_normalization
from .domain import (
    BALANCE_ROLES, CURRENCY_DISTRACTOR_ROLES, FEE_ROLES, MONETARY_DISTRACTOR_ROLES, Domain,
    financial_shorthand, magnitude_expression, maximal_digit_runs, parse_magnitude,
)
from .edge_params import (
    FEE_DISCLOSURE_QUESTION, GOLD_PARAMETER, K01_VARIANT_RUNS, SUPERSEDED_PARAMETER,
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
