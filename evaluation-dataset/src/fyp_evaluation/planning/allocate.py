"""G1 allocator: build one complete, deterministic dataset plan in memory.

Order: call-level attributes -> edge-case placement with verdict locks ->
compliance verdicts -> KYC gold -> transaction gold with edge-case parameters.
Every random choice uses a named stream derived from (seed, attempt, name).
An attempt that cannot satisfy a constraint is discarded and the next numbered
attempt runs; constraints are never relaxed.
"""

import hashlib
import random
from collections import Counter
from datetime import timedelta

from fyp_evaluation.validation.stage0 import is_multilingual, numeric_prerequisite_met, validate_stage0

from .domain import Domain, format_dob, maximal_digit_runs
from .edge_params import (
    BUILDERS, K01_VARIANT_RUNS, Balancer, BuildContext, Infeasible, missing_builders,
)

MAX_ATTEMPTS = 200
MAX_IDENTIFIER_DRAWS = 10_000


class PlanningError(ValueError):
    """G0 inputs are invalid or no attempt produced a feasible plan."""


def stream(seed, attempt, name):
    digest = hashlib.sha256(f"{seed}|{attempt}|{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def shuffled(items, rng):
    items = list(items)
    rng.shuffle(items)
    return items


def expand(pairs):
    return [label for label, count in pairs for _ in range(count)]


def forces_fail(case):
    return "FAIL" in case.get("verdict_requirements", {}).values()


class _CallState:
    def __init__(self, capacity, all_pass):
        self.slots, self.all_pass = capacity, all_pass
        self.cases, self.occupied, self.locks = [], set(), {}

    def correction_on(self, question_ids, cases):
        return any(
            cases[case_id]["family"] == "CORRECTION_SUPERSESSION" and set(targets) & question_ids
            for case_id, targets in self.cases
        )


def _closure(question_id, rules):
    """Questions that must FAIL when `question_id` FAILs (contrapositive of X PASS => Y PASS)."""
    result, changed = {question_id}, True
    while changed:
        changed = False
        for rule in rules:
            if rule["if"]["question_id"] not in result and any(t["question_id"] in result for t in rule["then"]):
                result.add(rule["if"]["question_id"])
                changed = True
    return result


def _attempt(documents, domain, seed, attempt):
    plan = documents["generation_plan"]
    catalog = documents["edge_case_catalog"]
    rules = documents["check_semantics"]["cross_check_rules"]
    planning_rules = plan["planning_rules"]
    exempt = set(planning_rules["per_question_limit_exempt_edge_cases"])
    cases = {case["id"]: case for case in catalog["edge_cases"]}
    case_index = {case_id: index for index, case_id in enumerate(cases)}
    question_index = {item["id"]: index for index, item in enumerate(documents["checklist"])}
    order_fields = set(domain.order_ids)

    def rng(name):
        return stream(seed, attempt, name)

    # Call-level attributes ------------------------------------------------
    call_ids = [f"CALL_{number:03d}" for number in range(1, plan["num_calls"] + 1)]
    composition = dict(zip(call_ids, shuffled(expand(plan["call_composition_counts"].items()), rng("composition"))))
    all_pass = {c: composition[c].startswith("all_pass") for c in call_ids}
    has_edges = {c: "_with_edge_cases" in composition[c] for c in call_ids}
    durations = {d["profile"]: d for d in plan["duration_profiles"]}
    duration = dict(zip(call_ids, shuffled(
        expand((d["profile"], d["count"]) for d in plan["duration_profiles"]), rng("duration"))))
    language = dict(zip(call_ids, shuffled(expand(plan["language_counts"].items()), rng("language"))))
    intents = dict(zip(call_ids, shuffled(
        expand((tuple(s["call_intents"]), s["count"]) for s in plan["call_intent_set_counts"]), rng("intents"))))

    # Edge-case count per call: exact configured distribution, seeded shuffle ---
    edge_counts = expand((int(number), count) for number, count in
                         plan["edge_case_distribution"]["edge_cases_per_call_counts"].items())
    edge_counts = sorted((n for n in edge_counts if n), reverse=True)
    edge_calls = [c for c in call_ids if has_edges[c]]
    if len(edge_counts) != len(edge_calls) or sum(edge_counts) != sum(plan["edge_case_distribution"]["family_quotas"].values()):
        raise PlanningError("Edge-case counts per call do not match the edge-case calls or family quotas")
    capacity = dict(zip(shuffled(edge_calls, rng("edge_counts")), edge_counts))

    # Edge-case instances: balanced within family, remainder in catalog order --
    instances = []
    quotas = plan["edge_case_distribution"]["family_quotas"]
    for family, quota in quotas.items():
        members = [case_id for case_id, case in cases.items() if case["family"] == family]
        base, remainder = divmod(quota, len(members))
        for position, case_id in enumerate(members):
            instances += [case_id] * (base + (position < remainder))

    def priority(case_id):
        case = cases[case_id]
        if case.get("target_scope") == "TRANSACTION_BUNDLE":
            rank = 0
        elif forces_fail(case):
            rank = 1
        elif case_id in exempt:
            rank = 4
        elif len(case["allowed_target_question_ids"]) == 1:
            rank = 2
        else:
            rank = 3
        return rank, case_index[case_id]

    instances.sort(key=priority)

    # Edge-case placement with verdict locks --------------------------------
    states = {c: _CallState(capacity.get(c, 0), all_pass[c]) for c in call_ids}
    placement_rng = rng("placement")
    balancer = Balancer(rng("targets"))
    bundle_count = 0

    def legal_targets(state, case):
        case_id = case["id"]
        if not state.slots or (forces_fail(case) and state.all_pass):
            return None
        if any(other in {cid for cid, _ in state.cases} for other in case.get("mutually_exclusive_with", [])):
            return None
        for question_id, verdict in case.get("verdict_requirements", {}).items():
            if state.locks.get(question_id, verdict) != verdict:
                return None
        allowed = case["allowed_target_question_ids"]
        if case.get("target_scope") == "TRANSACTION_BUNDLE":
            if state.correction_on(order_fields, cases):
                return None
            return [q for q in allowed if q not in state.occupied] or None
        if case_id in exempt:
            return list(allowed)
        has_bundle = any(cases[cid].get("target_scope") == "TRANSACTION_BUNDLE" for cid, _ in state.cases)
        targets = []
        for question_id in allowed:
            if question_id in state.occupied:
                continue
            if has_bundle and case["family"] == "CORRECTION_SUPERSESSION" and question_id in order_fields:
                continue
            required = case.get("target_verdict_requirement")
            if required and state.locks.get(question_id, required) != required:
                continue
            targets.append(question_id)
        return targets or None

    for case_id in instances:
        case = cases[case_id]
        chosen = None
        for call in sorted(shuffled(edge_calls, placement_rng), key=lambda c: -states[c].slots):
            targets = legal_targets(states[call], case)
            if case.get("target_scope") == "TRANSACTION_BUNDLE" and targets:
                changed = min(2 + bundle_count % 2, len(case["allowed_target_question_ids"]))
                if len(targets) < changed:
                    continue
            if targets:
                chosen = call, targets
                break
        if chosen is None:
            raise Infeasible(f"cannot place {case_id}")
        call, options = chosen
        state = states[call]
        if case.get("target_scope") == "TRANSACTION_BUNDLE":
            picks = []
            for _ in range(changed):
                picks.append(balancer.choose(f"target:{case_id}", [q for q in options if q not in picks]))
            bundle_count += 1
        elif case_id in exempt:
            picks = []
            for _ in range(2):
                picks.append(balancer.choose(f"target:{case_id}", [q for q in options if q not in picks]))
        else:
            picks = [balancer.choose(f"target:{case_id}", options)]
        for pick in picks:
            balancer.counts[(f"target:{case_id}", pick)] += 1
        picks.sort(key=question_index.get)
        state.slots -= 1
        state.cases.append((case_id, picks))
        if case_id not in exempt:
            state.occupied.update(picks)
        state.locks.update(case.get("verdict_requirements", {}))
        if "target_verdict_requirement" in case:
            state.locks[picks[0]] = case["target_verdict_requirement"]

    # Compliance verdicts: exact FAIL count per call -------------------------
    minimum_fails = plan["compliance_distribution"]["min_fail_cases_per_check"]
    some_fail = [c for c in call_ids if not all_pass[c]]
    fails = {c: {q for q, v in states[c].locks.items() if v == "FAIL"} for c in call_ids}
    for call in call_ids:
        for question_id in list(fails[call]):
            fails[call] |= _closure(question_id, rules)
    compliance_rng = rng("compliance")
    units = sorted((int(number) for number, count in
                    plan["compliance_distribution"]["fail_count_per_call_counts"].items()
                    for _ in range(count) if int(number)), reverse=True)
    if len(units) != len(some_fail):
        raise PlanningError("FAIL-count distribution does not match the number of some-FAIL calls")
    # Match budgets to calls: edge-case-forced FAILs first, then a seeded draw among budgets that fit.
    budget = {c: 0 for c in call_ids}
    for call in sorted(some_fail, key=lambda c: (-len(fails[c]), compliance_rng.random())):
        fitting = [unit for unit in units if unit >= len(fails[call])]
        if not fitting:
            raise Infeasible(f"{call}: no FAIL budget covers its {len(fails[call])} edge-case-forced FAILs")
        budget[call] = compliance_rng.choice(fitting)
        units.remove(budget[call])
    counts = Counter(q for c in call_ids for q in fails[c])

    def addition(call, question_id):
        add = _closure(question_id, rules) - fails[call]
        if any(states[call].locks.get(q) == "PASS" for q in add):
            return None
        if len(fails[call]) + len(add) > budget[call]:
            return None
        return add

    for question_id in sorted(domain.compliance_ids, key=lambda q: (-len(_closure(q, rules)), question_index[q])):
        while counts[question_id] < minimum_fails:
            options = [c for c in shuffled(some_fail, compliance_rng)
                       if question_id not in fails[c] and addition(c, question_id) is not None]
            if not options:
                raise Infeasible(f"cannot reach {minimum_fails} FAIL cases for {question_id}")
            call = max(options, key=lambda c: budget[c] - len(fails[c]))
            for q in addition(call, question_id):
                fails[call].add(q)
                counts[q] += 1
    for call in shuffled(some_fail, compliance_rng):
        while len(fails[call]) < budget[call]:
            options = [q for q in shuffled(domain.compliance_ids, compliance_rng)
                       if q not in fails[call] and addition(call, q) is not None]
            if not options:
                raise Infeasible(f"{call}: cannot fill its FAIL budget of {budget[call]}")
            question_id = min(options, key=lambda q: (counts[q], len(addition(call, q))))
            for q in addition(call, question_id):
                fails[call].add(q)
                counts[q] += 1
    verdicts = {c: {q: "FAIL" if q in fails[c] else "PASS" for q in domain.compliance_ids} for c in call_ids}

    # KYC gold ---------------------------------------------------------------
    kyc_rng = rng("kyc")
    nric = domain.nric
    gold = {c: dict(verdicts[c]) for c in call_ids}
    variants = {}
    used_k01 = set()
    for call in call_ids:
        prerequisites = [
            cases[case_id].get("gold_prerequisites") for case_id, targets in states[call].cases
            if domain.k01 in targets and cases[case_id].get("gold_prerequisites")
        ]
        run_length = None
        if any(case_id == "EC_DOUBLE_TRIPLE_DIGITS" for case_id, _ in states[call].cases):
            variants[call] = balancer.choose("k01_variant", list(K01_VARIANT_RUNS))
            balancer.counts[("k01_variant", variants[call])] += 1
            run_length = K01_VARIANT_RUNS[variants[call]]
        prefix = balancer.choose("k01_prefix", nric["prefixes"])
        balancer.counts[("k01_prefix", prefix)] += 1
        for _ in range(MAX_IDENTIFIER_DRAWS):
            digits = [str(kyc_rng.randrange(10)) for _ in range(nric["digit_count"])]
            for prerequisite in prerequisites:
                if prerequisite["kind"] == "CONTAINS_CHARACTER" and prerequisite["character"] not in digits:
                    digits[kyc_rng.randrange(len(digits))] = prerequisite["character"]
            if run_length:
                start = kyc_rng.randrange(len(digits) - run_length + 1)
                digits[start:start + run_length] = [str(kyc_rng.randrange(1, 10))] * run_length
            value = prefix + "".join(digits) + kyc_rng.choice(nric["suffix_letters"])
            runs = maximal_digit_runs(value[1:-1])
            if (not domain.valid_k01(value) or value in used_k01
                    or (run_length and run_length not in runs)
                    or not all(_k01_prerequisite_met(p, value) for p in prerequisites)):
                continue
            break
        else:
            raise Infeasible(f"{call}: no unique checksum-invalid K01 after {MAX_IDENTIFIER_DRAWS} draws")
        used_k01.add(value)
        gold[call][domain.k01] = value
    span = (domain.dob_max - domain.dob_min).days
    used_k02 = set()
    for call in call_ids:
        while True:
            value = format_dob(domain.dob_min + timedelta(days=kyc_rng.randrange(span + 1)))
            if value not in used_k02:
                break
        used_k02.add(value)
        gold[call][domain.k02] = value

    # Transaction gold with edge-case parameters ------------------------------
    extraction = plan["extraction_distribution"]
    transaction_rng = rng("transaction")
    currency = dict(zip(call_ids, shuffled(expand(extraction["transaction_currency_counts"].items()), transaction_rng)))
    direction = dict(zip(call_ids, shuffled(expand(extraction["transaction_direction_counts"].items()), transaction_rng)))
    order_type = dict(zip(call_ids, shuffled(expand(extraction["order_type_counts"].items()), transaction_rng)))
    usage = Counter()
    excluded = {domain.k01: frozenset(used_k01), domain.k02: frozenset(used_k02)}
    parameters = {}
    for call in call_ids:
        cur = currency[call]
        prerequisites = {
            q: [cases[case_id]["gold_prerequisites"] for case_id, targets in states[call].cases
                if q in targets and cases[case_id].get("gold_prerequisites")]
            for q in (domain.t03, domain.t04)
        }
        products = [p for p in domain.product_names if cur in domain.products[p]["allowed_currencies"]]
        products = sorted(shuffled(products, transaction_rng), key=lambda p: usage[("product", p)])
        pairs = [
            (amount, quantity) for amount, quantity in domain.coherent_pairs(cur)
            if all(numeric_prerequisite_met(p, amount) for p in prerequisites[domain.t03])
            and all(numeric_prerequisite_met(p, quantity) for p in prerequisites[domain.t04])
        ]
        pairs = sorted(shuffled(pairs, transaction_rng),
                       key=lambda pair: usage[("amount", cur, pair[0])] + usage[("quantity", pair[1])])
        accepted = None
        for product in products:
            for amount, quantity in pairs:
                row = {**gold[call], domain.t01: product, domain.t02: cur, domain.t03: amount,
                       domain.t04: quantity, domain.t05: direction[call], domain.t06: order_type[call]}
                ctx = BuildContext(domain, row, verdicts[call], stream(seed, attempt, f"params:{call}"),
                                   balancer, excluded, variants.get(call), catalog)
                try:
                    built = [BUILDERS[case_id](ctx, cases[case_id], targets)
                             for case_id, targets in states[call].cases]
                except Infeasible:
                    continue
                accepted = row, built, ctx
                break
            if accepted:
                break
        if accepted is None:
            raise Infeasible(f"{call}: no product/amount/quantity satisfies its edge cases")
        row, built, ctx = accepted
        ctx.commit()
        usage.update([("product", row[domain.t01]), ("amount", cur, row[domain.t03]), ("quantity", row[domain.t04])])
        gold[call] = row
        parameters[call] = built

    assignments = []
    for call in call_ids:
        for (case_id, targets), params in zip(states[call].cases, parameters[call]):
            assignments.append({
                "call_id": call, "edge_case_id": case_id, "family": cases[case_id]["family"],
                "target_question_ids": targets, "parameters": params,
            })
    assignments.sort(key=lambda a: (a["call_id"], case_index[a["edge_case_id"]],
                                    [question_index[q] for q in a["target_question_ids"]]))
    for number, assignment in enumerate(assignments, start=1):
        assignment["assignment_id"] = f"EA_{number:03d}"
    calls = [{
        "call_id": c,
        "call_intents": list(intents[c]),
        "language": language[c],
        "duration_profile": duration[c],
        "target_duration_min_seconds": durations[duration[c]]["min_seconds"],
        "target_duration_max_seconds": durations[duration[c]]["max_seconds"],
    } for c in call_ids]
    return {"calls": calls, "gold": gold, "assignments": assignments}


def _k01_prerequisite_met(prerequisite, value):
    if prerequisite["kind"] == "CONTAINS_CHARACTER":
        return prerequisite["character"] in value[1:-1]
    if prerequisite["kind"] == "REPEATED_DIGIT_RUN":
        return max(maximal_digit_runs(value[1:-1])) >= prerequisite["minimum_run_length"]
    return True


def build_plan(documents, *, max_attempts=MAX_ATTEMPTS):
    """Return the first feasible plan for the configured seed, or raise PlanningError."""
    if errors := validate_stage0(documents):
        raise PlanningError("G0 contracts are invalid:\n" + "\n".join(errors))
    if missing := missing_builders(documents["edge_case_catalog"]):
        raise PlanningError(f"No G1 parameter builder registered for {missing}")
    domain = Domain(documents)
    seed = documents["generation_plan"]["random_seed"]
    attempt_plan = _attempt
    if is_multilingual(documents):
        from .multilingual import multilingual_attempt  # deferred: that module reuses this module's helpers
        attempt_plan = multilingual_attempt
    rejected = []
    for attempt in range(max_attempts):
        try:
            result = attempt_plan(documents, domain, seed, attempt)
        except Infeasible as reason:
            rejected.append(f"attempt {attempt}: {reason}")
            continue
        result["attempt"] = attempt
        result["rejected_attempts"] = rejected
        return result
    raise PlanningError(f"No feasible plan in {max_attempts} attempts. Last reasons:\n" + "\n".join(rejected[-5:]))
