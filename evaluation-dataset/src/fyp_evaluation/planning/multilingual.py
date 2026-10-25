"""G1 kernel for a multilingual (generation plan 2.0) release: exact allocation within each language profile.

Each language profile is planned on its own, in this order:
  1. layout: every call's (FAIL count, edge-case count, duration) from the profile's exact histograms, the
     all-PASS / edge-free overlap and the short-call load cap;
  2. edge-case placement with verdict locks and exact per-case counts (target-exposure caps respected);
  3. language variants with exact per-variant counts, never pinning one event to two languages;
  4. compliance verdicts: exact per-question FAIL counts filled by a max-flow over the per-call budgets, with the
     dependent-FAIL rule (C03 FAILs only with C02 or its edge case) and at most N prohibition FAILs per call;
  5. KYC and transaction gold with edge-case parameters, currency chosen with edge-case eligibility first.
The profiles then share the global call IDs through a seeded interleave. K01/K02 uniqueness and product/amount
balancing are release-wide. Every random choice uses a named stream derived from (seed, attempt, profile, name);
an attempt that cannot satisfy a constraint is discarded and the next numbered attempt runs. Constraints are never
relaxed.
"""

from collections import Counter, deque
from datetime import timedelta

from fyp_evaluation.validation.stage0 import numeric_prerequisite_met

from .allocate import MAX_IDENTIFIER_DRAWS, _closure, _k01_prerequisite_met, expand, forces_fail, shuffled, stream
from .domain import format_dob, maximal_digit_runs
from .edge_params import BUILDERS, K01_VARIANT_RUNS, Balancer, BuildContext, Infeasible, magnitude_contrast

NO_VARIANT = "NONE"
NO_MIXING_STYLE = "NONE"
# Cases whose cross-language or EN_SURFACE variant pins the FINAL event of their target, which they do not create
# themselves (the plain FINAL, or the shared final ORDER_STATEMENT of an order replacement).
FINAL_PINNING_CASES = frozenset({
    "EC_MULTIPLE_CURRENCIES", "EC_MULTIPLE_MONETARY_VALUES", "EC_FALSE_START",
    "EC_DOUBLE_TRIPLE_DIGITS", "EC_OH_VS_ZERO", "EC_FINANCIAL_SHORTHAND", "EC_LARGE_NUMBER_EXPRESSION",
})
BUNDLE_FINAL = "ORDER_REPLACEMENT_FINAL"


# Families whose cases create their target's FINAL event; an order replacement already supplies every transaction FINAL.
SUPPLYING_FAMILIES = frozenset({"CORRECTION_SUPERSESSION", "REFERENCE_RESOLUTION"})


def supplies_final(case):
    return case["family"] in SUPPLYING_FAMILIES and case.get("target_scope") != "TRANSACTION_BUNDLE"


def separates_order_finals(case, order_fields):
    """True for a multi-question case that needs distinct FINAL events on transaction fields (a long-distance dependency).

    An order replacement gives every transaction field one shared final ORDER_STATEMENT, so the two cannot share a call.
    """
    return case.get("target_scope") == "MULTI_QUESTION" and len(set(case["allowed_target_question_ids"]) - order_fields) < 2


class _CallState:
    def __init__(self, fails, edges):
        self.fails, self.slots = fails, edges
        self.cases, self.occupied, self.locks = [], set(), {}
        self.exposed = set()

    def fail_locks(self):
        return {q for q, v in self.locks.items() if v == "FAIL"}

    def supplier_on(self, question_ids, cases):
        return any(supplies_final(cases[c]) and set(t) & question_ids for c, t in self.cases)


def _layout(p, rng, load_cap):
    """[(fails, edges, duration)] satisfying the histograms, the overlap and the load cap on capped calls."""
    composition = p["call_composition_counts"]
    pure, pass_with = composition["all_pass_without_edge_cases"], composition["all_pass_with_edge_cases"]
    fail_without = composition["some_fail_without_edge_cases"]
    fails = shuffled(expand((int(k), v) for k, v in p["fail_count_per_call_counts"].items() if int(k)), rng)
    edges = shuffled(expand((int(k), v) for k, v in p["edge_cases_per_call_counts"].items() if int(k)), rng)
    pairs = ([[0, 0] for _ in range(pure)] + [[0, e] for e in edges[:pass_with]] + [[f, 0] for f in fails[:fail_without]]
             + [list(pair) for pair in zip(fails[fail_without:], edges[pass_with:])])
    order = shuffled(range(len(pairs)), rng)
    ranked = sorted(order, key=lambda i: -(pairs[i][0] + pairs[i][1]))
    durations = expand(p["duration_counts"].items())
    capped, cap = load_cap["duration_profile"], load_cap["max_fails_plus_edge_cases"]
    uncapped = [d for d in durations if d != capped]
    duration = {}
    for position, index in enumerate(ranked):
        duration[index] = uncapped[position] if position < len(uncapped) else capped
    # Repair overloaded capped calls by swapping edge-case counts between calls that both have FAILs and edge cases.
    both = [i for i in order if pairs[i][0] and pairs[i][1]]
    for i in [i for i in order if duration[i] == capped]:
        while sum(pairs[i]) > cap:
            partner = next((j for j in both if j != i and pairs[j][1] < pairs[i][1] and pairs[i][0] + pairs[j][1] <= cap
                            and (duration[j] != capped or pairs[j][0] + pairs[i][1] <= cap)), None)
            if partner is None:
                raise Infeasible(f"{p['language_profile']}: no layout keeps capped calls within {cap}")
            pairs[i][1], pairs[partner][1] = pairs[partner][1], pairs[i][1]
    return [(pairs[i][0], pairs[i][1], duration[i]) for i in order]


def _instances(p, cases, case_index, exempt, order_fields):
    """Instances in placement order: the most constrained mechanisms first (order replacements, then the cases that
    cannot share a call with them, then forced-FAIL, single-target, multi-target and exempt cases)."""
    def priority(case_id):
        case = cases[case_id]
        if case.get("target_scope") == "TRANSACTION_BUNDLE":
            rank = 0
        elif separates_order_finals(case, order_fields):
            rank = 1
        elif forces_fail(case):
            rank = 2
        elif case_id in exempt:
            rank = 5
        elif len(case["allowed_target_question_ids"]) == 1:
            rank = 3
        else:
            rank = 4
        return rank, case_index[case_id]
    return sorted(expand(p["edge_case_counts"].items()), key=priority)


def _place(p, calls, states, cases, case_index, exempt, order_fields, rng, balancer, question_index):
    """Place every instance of the profile's exact per-case counts into the call states (cases, locks, exposure)."""
    caps = p["target_exposure_caps"]
    exposure = Counter()
    bundle_count = 0

    def exposable(state, q):
        return q not in caps or q in state.exposed or exposure[q] < caps[q]

    def legal_targets(state, case):
        case_id = case["id"]
        if not state.slots:
            return None
        if any(other in {c for c, _ in state.cases} for other in case.get("mutually_exclusive_with", [])):
            return None
        new_fails = {q for q, v in case.get("verdict_requirements", {}).items() if v == "FAIL"}
        if len(state.fail_locks() | new_fails) > state.fails:
            return None
        for question_id, verdict in case.get("verdict_requirements", {}).items():
            if state.locks.get(question_id, verdict) != verdict:
                return None
        allowed = [q for q in case["allowed_target_question_ids"] if exposable(state, q)]
        has_bundle = any(cases[c].get("target_scope") == "TRANSACTION_BUNDLE" for c, _ in state.cases)
        if case.get("target_scope") == "TRANSACTION_BUNDLE":
            if state.supplier_on(order_fields, cases) or any(separates_order_finals(cases[c], order_fields) for c, _ in state.cases):
                return None
            return [q for q in allowed if q not in state.occupied] or None
        if has_bundle and separates_order_finals(case, order_fields):
            return None
        if case_id in exempt:
            return allowed if len(allowed) >= 2 else None
        targets = []
        for question_id in allowed:
            if question_id in state.occupied:
                continue
            if has_bundle and supplies_final(case) and question_id in order_fields:
                continue
            required = case.get("target_verdict_requirement")
            if required and state.locks.get(question_id, required) != required:
                continue
            targets.append(question_id)
        return targets or None

    for case_id in _instances(p, cases, case_index, exempt, order_fields):
        case = cases[case_id]
        chosen = None
        for call in sorted(shuffled(calls, rng), key=lambda c: -states[c].slots):
            targets = legal_targets(states[call], case)
            if case.get("target_scope") == "TRANSACTION_BUNDLE" and targets:
                changed = min(2 + bundle_count % 2, len(case["allowed_target_question_ids"]))
                if len(targets) < changed:
                    continue
            if targets:
                chosen = call, targets
                break
        if chosen is None:
            raise Infeasible(f"{p['language_profile']}: cannot place {case_id}")
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
        for q in picks:
            if q in caps and q not in state.exposed:
                state.exposed.add(q)
                exposure[q] += 1
        state.slots -= 1
        state.cases.append((case_id, picks))
        if case_id not in exempt:
            state.occupied.update(picks)
        state.locks.update(case.get("verdict_requirements", {}))
        if "target_verdict_requirement" in case:
            state.locks[picks[0]] = case["target_verdict_requirement"]


def contrast_factors_avoided(cases, profile):
    """{case_id: factor} for cases whose magnitude-contrast variant is planned in this profile: their variant-free
    instances must not contain such a contrast by accident, so every contrast in the profile is a labelled variant."""
    avoided = {}
    for case_id, case in cases.items():
        for variant in case.get("language_variants", []):
            constraint = magnitude_contrast(variant)
            if constraint and variant["status"] == "READY_V1" and profile in variant["profiles"]:
                avoided[case_id] = constraint["factor"]
    return avoided


def variant_pins(case_id, targets, variant, has_bundle, order_fields):
    """{FINAL-event key: language} that one assignment's variant pins on events it does not create."""
    languages = (variant or {}).get("event_languages", {})
    pins = {}
    if case_id == "EC_COMPLETE_ORDER_REPLACEMENT" and "target" in languages:
        pins[BUNDLE_FINAL] = languages["target"]
    elif case_id in FINAL_PINNING_CASES and "target" in languages:
        for q in targets:
            pins[BUNDLE_FINAL if has_bundle and q in order_fields else q] = languages["target"]
    return pins


def _assign_variants(p, calls, states, cases, order_fields, rng):
    """{(call, position): variant_id}: exact per-case variant counts on instances that cannot pin conflicting languages."""
    chosen = {}
    pins = {call: {} for call in calls}
    defined = {c: {v["variant_id"]: v for v in case.get("language_variants", [])} for c, case in cases.items()}
    # Mandatory variants (count == every instance) first, so optional ones avoid them.
    requests = sorted(p["language_variant_counts"].items(),
                      key=lambda item: -(sum(item[1].values()) == p["edge_case_counts"][item[0]]))
    for case_id, by_variant in requests:
        slots = [(call, i) for call in calls for i, (c, _) in enumerate(states[call].cases) if c == case_id]
        slots = shuffled(slots, rng)
        for variant_id, count in by_variant.items():
            variant = defined[case_id][variant_id]
            placed = 0
            for call, i in slots:
                if placed == count:
                    break
                if (call, i) in chosen:
                    continue
                state = states[call]
                has_bundle = any(c == "EC_COMPLETE_ORDER_REPLACEMENT" for c, _ in state.cases)
                new = variant_pins(case_id, state.cases[i][1], variant, has_bundle, order_fields)
                if any(pins[call].get(key, language) != language for key, language in new.items()):
                    continue
                pins[call].update(new)
                chosen[(call, i)] = variant_id
                placed += 1
            if placed < count:
                raise Infeasible(f"{p['language_profile']}: cannot assign {count} {case_id}/{variant_id} variants without conflicts")
    return chosen


def _max_flow(capacity, source, sink):
    """Edmonds-Karp on an adjacency dict {u: {v: cap}} (mutated into the residual graph); returns the flow value."""
    flow = 0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v, cap in capacity[u].items():
                if cap > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            return flow
        v, bottleneck = sink, float("inf")
        while parent[v] is not None:
            bottleneck = min(bottleneck, capacity[parent[v]][v])
            v = parent[v]
        v = sink
        while parent[v] is not None:
            u = parent[v]
            capacity[u][v] -= bottleneck
            capacity[v].setdefault(u, 0)
            capacity[v][u] += bottleneck
            v = u
        flow += bottleneck


def _verdicts(p, calls, states, compliance_ids, rules, plan_rules, rng):
    """Exact per-question FAIL counts; returns {call: {question: verdict}}."""
    targets = p["fail_counts_per_check"]
    fails = {call: set(states[call].fail_locks()) for call in calls}
    for call in calls:
        for q in list(fails[call]):
            fails[call] |= _closure(q, rules)
    dependents = plan_rules["dependent_fail_rules"]
    paired = {rule["fails_only_with"]["failed_question_id"]: rule["question_id"] for rule in dependents}
    dependent = set(paired.values())
    prohibition = set(plan_rules["prohibition_question_ids"])
    max_prohibition = plan_rules["max_prohibition_fails_per_call"]
    for call in calls:
        if len(fails[call]) > states[call].fails:
            raise Infeasible(f"{call}: edge cases force more FAILs than its budget")
    counts = Counter(q for call in calls for q in fails[call])
    budget = {call: states[call].fails - len(fails[call]) for call in calls}

    def free(call, q):
        return q not in fails[call] and states[call].locks.get(q) != "PASS"

    # Paired FAILs (C02 with its dependent C03) take two budget units; place them first on the roomiest calls.
    for failed, q in paired.items():
        need = targets[failed] - counts[failed]
        if targets[q] - counts[q] != need:
            raise Infeasible(f"{p['language_profile']}: {q} FAILs cannot follow {failed} FAILs exactly")
        options = sorted(shuffled([c for c in calls if budget[c] >= 2 and free(c, failed) and free(c, q)], rng),
                         key=lambda c: -budget[c])
        if len(options) < need:
            raise Infeasible(f"{p['language_profile']}: not enough room for {need} {failed}+{q} FAIL pairs")
        for call in options[:need]:
            fails[call] |= {failed, q}
            budget[call] -= 2
            counts[failed] += 1
            counts[q] += 1
    singles = [q for q in compliance_ids if q not in paired and q not in dependent]
    demand = {q: targets[q] - counts[q] for q in singles}
    if any(d < 0 for d in demand.values()) or sum(demand.values()) != sum(budget.values()):
        raise Infeasible(f"{p['language_profile']}: FAIL budgets and per-question targets disagree")
    source, sink = "SOURCE", "SINK"
    graph = {source: {}, sink: {}}
    for q in singles:
        graph[("Q", q)] = {sink: demand[q]}
    for call in shuffled(calls, rng):
        if not budget[call]:
            continue
        node, group = ("CALL", call), ("GROUP", call)
        graph[source][node] = budget[call]
        graph[node] = {}
        graph[group] = {}
        room = max_prohibition - len(fails[call] & prohibition)
        if room > 0:
            graph[node][group] = room
        for q in shuffled(singles, rng):
            if free(call, q):
                if q in prohibition:
                    graph[group][("Q", q)] = 1
                else:
                    graph[node][("Q", q)] = 1
    original = {u: dict(edges) for u, edges in graph.items()}
    if _max_flow(graph, source, sink) != sum(budget.values()):
        raise Infeasible(f"{p['language_profile']}: no exact FAIL fill satisfies the per-call budgets and per-question targets")
    for call in calls:
        node, group = ("CALL", call), ("GROUP", call)
        for start in (node, group):
            for target, cap in original.get(start, {}).items():
                if isinstance(target, tuple) and target[0] == "Q" and cap - graph[start][target] == 1:
                    fails[call].add(target[1])
    return {call: {q: "FAIL" if q in fails[call] else "PASS" for q in compliance_ids} for call in calls}


def _currencies(p, calls, states, cases, variants, domain, rng):
    """Exact per-profile currency counts; calls whose edge cases narrow the eligible currencies are served first."""
    remaining = Counter({c: n for c, n in p["transaction_currency_counts"].items() if n})

    def eligible(call):
        prerequisites = {q: [cases[c]["gold_prerequisites"] for c, t in states[call].cases
                             if q in t and cases[c].get("gold_prerequisites")] for q in (domain.t03, domain.t04)}
        factor = next((magnitude_contrast(variants[(call, i)])["factor"] for i, _ in enumerate(states[call].cases)
                       if magnitude_contrast(variants.get((call, i)))), None)
        result = []
        for currency in remaining:
            if not any(currency in domain.products[name]["allowed_currencies"] for name in domain.product_names):
                continue
            if any(all(numeric_prerequisite_met(r, a) for r in prerequisites[domain.t03])
                   and all(numeric_prerequisite_met(r, q) for r in prerequisites[domain.t04])
                   and (factor is None or domain.magnitude_partners(currency, a, factor))
                   for a, q in domain.coherent_pairs(currency)):
                result.append(currency)
        return result

    options = {call: eligible(call) for call in calls}
    assigned = {}
    for call in sorted(shuffled(calls, rng), key=lambda c: len(options[c])):
        choices = [c for c in options[call] if remaining[c]]
        if not choices:
            raise Infeasible(f"{call}: no planned currency remains that satisfies its edge cases")
        best = max(remaining[c] for c in choices)
        currency = rng.choice(sorted(c for c in choices if remaining[c] == best))
        assigned[call] = currency
        remaining[currency] -= 1
    return assigned


def _plan_structure(p, calls, documents, domain, seed, attempt, shared):
    """Layout, edge cases, variants, verdicts and KYC gold for one profile (KYC uniqueness is release-wide)."""
    plan = documents["generation_plan"]
    catalog = documents["edge_case_catalog"]
    cases = {case["id"]: case for case in catalog["edge_cases"]}
    case_index = {case_id: index for index, case_id in enumerate(cases)}
    question_index = {item["id"]: index for index, item in enumerate(documents["checklist"])}
    rules = documents["check_semantics"]["cross_check_rules"]
    plan_rules = plan["planning_rules"]
    exempt = set(plan_rules["per_question_limit_exempt_edge_cases"])
    name = p["language_profile"]
    order_fields = set(domain.order_ids)

    def rng(label):
        return stream(seed, attempt, f"{name}/{label}")

    layout = _layout(p, rng("layout"), plan_rules["load_cap"])
    states = {call: _CallState(f, e) for call, (f, e, _) in zip(calls, layout)}
    duration = {call: d for call, (_, _, d) in zip(calls, layout)}
    balancer = Balancer(rng("targets"))
    _place(p, calls, states, cases, case_index, exempt, order_fields, rng("placement"), balancer, question_index)
    variant_ids = _assign_variants(p, calls, states, cases, order_fields, rng("variants"))
    defined = {c: {v["variant_id"]: v for v in case.get("language_variants", [])} for c, case in cases.items()}
    variants = {key: defined[states[key[0]].cases[key[1]][0]][v] for key, v in variant_ids.items()}
    verdicts = _verdicts(p, calls, states, domain.compliance_ids, rules, plan_rules, rng("compliance"))
    intents = dict(zip(calls, shuffled(expand((tuple(s["call_intents"]), s["count"]) for s in p["call_intent_set_counts"]),
                                       rng("intents"))))
    styles = dict(zip(calls, shuffled(expand(p.get("mixing_style_counts", {}).items()), rng("mixing")))) if "mixing_style_counts" in p else {}

    # KYC gold (release-wide uniqueness) -------------------------------------------------------------------
    kyc_rng = rng("kyc")
    nric = domain.nric
    gold = {call: dict(verdicts[call]) for call in calls}
    k01_variants = {}
    for call in calls:
        prerequisites = [cases[c].get("gold_prerequisites") for c, targets in states[call].cases
                         if domain.k01 in targets and cases[c].get("gold_prerequisites")]
        run_length = None
        if any(c == "EC_DOUBLE_TRIPLE_DIGITS" for c, _ in states[call].cases):
            k01_variants[call] = balancer.choose("k01_variant", list(K01_VARIANT_RUNS))
            balancer.counts[("k01_variant", k01_variants[call])] += 1
            run_length = K01_VARIANT_RUNS[k01_variants[call]]
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
            if (not domain.valid_k01(value) or value in shared["k01"]
                    or (run_length and run_length not in runs)
                    or not all(_k01_prerequisite_met(r, value) for r in prerequisites)):
                continue
            break
        else:
            raise Infeasible(f"{call}: no unique checksum-invalid K01 after {MAX_IDENTIFIER_DRAWS} draws")
        shared["k01"].add(value)
        gold[call][domain.k01] = value
    span = (domain.dob_max - domain.dob_min).days
    for call in calls:
        while True:
            value = format_dob(domain.dob_min + timedelta(days=kyc_rng.randrange(span + 1)))
            if value not in shared["k02"]:
                break
        shared["k02"].add(value)
        gold[call][domain.k02] = value
    return {"p": p, "calls": calls, "rng": rng, "states": states, "duration": duration, "balancer": balancer,
            "variant_ids": variant_ids, "variants": variants, "verdicts": verdicts, "intents": intents, "styles": styles,
            "gold": gold, "k01_variants": k01_variants}


def _plan_values(s, documents, domain, seed, attempt, shared):
    """Transaction gold and edge-case parameters for one profile, after every profile's KYC gold exists."""
    p, calls, rng, states = s["p"], s["calls"], s["rng"], s["states"]
    balancer, variants, verdicts, gold, k01_variants = s["balancer"], s["variants"], s["verdicts"], s["gold"], s["k01_variants"]
    duration, variant_ids, intents, styles = s["duration"], s["variant_ids"], s["intents"], s["styles"]
    plan = documents["generation_plan"]
    catalog = documents["edge_case_catalog"]
    cases = {case["id"]: case for case in catalog["edge_cases"]}
    name = p["language_profile"]

    avoided = contrast_factors_avoided(cases, name)

    # Transaction gold with edge-case parameters --------------------------------------------------------------
    transaction_rng = rng("transaction")
    currency = _currencies(p, calls, states, cases, variants, domain, rng("currency"))
    direction = dict(zip(calls, shuffled(expand(p["transaction_direction_counts"].items()), transaction_rng)))
    order_type = dict(zip(calls, shuffled(expand(p["order_type_counts"].items()), transaction_rng)))
    usage = shared["usage"]
    excluded = {domain.k01: frozenset(shared["k01"]), domain.k02: frozenset(shared["k02"])}
    parameters = {}
    for call in calls:
        cur = currency[call]
        prerequisites = {q: [cases[c]["gold_prerequisites"] for c, t in states[call].cases if q in t and cases[c].get("gold_prerequisites")]
                         for q in (domain.t03, domain.t04)}
        factors = [magnitude_contrast(variants.get((call, i)))["factor"] for i in range(len(states[call].cases))
                   if magnitude_contrast(variants.get((call, i)))]
        products = [n for n in domain.product_names if cur in domain.products[n]["allowed_currencies"]]
        products = sorted(shuffled(products, transaction_rng), key=lambda n: usage[("product", n)])
        pairs = [
            (amount, quantity) for amount, quantity in domain.coherent_pairs(cur)
            if all(numeric_prerequisite_met(r, amount) for r in prerequisites[domain.t03])
            and all(numeric_prerequisite_met(r, quantity) for r in prerequisites[domain.t04])
            and all(domain.magnitude_partners(cur, amount, f) for f in factors)
        ]
        pairs = sorted(shuffled(pairs, transaction_rng), key=lambda pair: usage[("amount", cur, pair[0])] + usage[("quantity", pair[1])])
        accepted = None
        for product in products:
            for amount, quantity in pairs:
                row = {**gold[call], domain.t01: product, domain.t02: cur, domain.t03: amount,
                       domain.t04: quantity, domain.t05: direction[call], domain.t06: order_type[call]}
                ctx = BuildContext(domain, row, verdicts[call], stream(seed, attempt, f"{name}/params:{call}"),
                                   balancer, excluded, k01_variants.get(call), catalog)
                try:
                    built = []
                    for i, (case_id, targets) in enumerate(states[call].cases):
                        ctx.language_variant = variants.get((call, i))
                        ctx.avoided_magnitude_factor = None if ctx.language_variant else avoided.get(case_id)
                        built.append(BUILDERS[case_id](ctx, cases[case_id], targets))
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

    durations = {d["profile"]: d for d in plan["duration_profiles"]}
    calls_out = [{
        "call_id": call,
        "call_intents": list(intents[call]),
        "language_profile": name,
        "mixing_style": styles.get(call, NO_MIXING_STYLE),
        "duration_profile": duration[call],
        "target_duration_min_seconds": durations[duration[call]]["min_seconds"],
        "target_duration_max_seconds": durations[duration[call]]["max_seconds"],
    } for call in calls]
    assignments = [{
        "call_id": call, "edge_case_id": case_id, "family": cases[case_id]["family"], "target_question_ids": targets,
        "language_variant": variant_ids.get((call, i), NO_VARIANT), "parameters": params,
    } for call in calls for i, ((case_id, targets), params) in enumerate(zip(states[call].cases, parameters[call]))]
    return calls_out, gold, assignments


def multilingual_attempt(documents, domain, seed, attempt):
    """One complete multilingual plan in memory, or Infeasible."""
    plan = documents["generation_plan"]
    plans = plan["language_profile_plans"]
    cases = {case["id"]: case for case in documents["edge_case_catalog"]["edge_cases"]}
    case_index = {case_id: index for index, case_id in enumerate(cases)}
    question_index = {item["id"]: index for index, item in enumerate(documents["checklist"])}
    call_ids = [f"CALL_{number:03d}" for number in range(1, plan["num_calls"] + 1)]
    labels = shuffled(expand((p["language_profile"], p["num_calls"]) for p in plans), stream(seed, attempt, "profile_interleave"))
    members = {p["language_profile"]: [] for p in plans}
    for call, label in zip(call_ids, labels):
        members[label].append(call)
    shared = {"k01": set(), "k02": set(), "usage": Counter()}
    structures = [_plan_structure(p, members[p["language_profile"]], documents, domain, seed, attempt, shared) for p in plans]
    calls, gold, assignments = [], {}, []
    for structure in structures:
        out_calls, out_gold, out_assignments = _plan_values(structure, documents, domain, seed, attempt, shared)
        calls += out_calls
        gold.update(out_gold)
        assignments += out_assignments
    calls.sort(key=lambda c: c["call_id"])
    assignments.sort(key=lambda a: (a["call_id"], case_index[a["edge_case_id"]], [question_index[q] for q in a["target_question_ids"]]))
    for number, assignment in enumerate(assignments, start=1):
        assignment["assignment_id"] = f"EA_{number:03d}"
    return {"calls": calls, "gold": gold, "assignments": assignments}
