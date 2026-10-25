"""Deterministic parameter builders, registered by edge_case_id.

Each builder derives one assignment's parameters from the call's locked gold
row. Final values are copied from gold; superseded, wrong and distractor
values are realistic alternatives from the configured domains. A builder
raises Infeasible when the call's gold leaves no legal parameter choice.
"""

from collections import Counter

from .domain import (
    CURRENCY_DISTRACTOR_ROLES, FEE_ROLES, MONETARY_DISTRACTOR_ROLES,
    financial_shorthand, magnitude_expression,
)

FEE_DISCLOSURE_QUESTION = "C04"
K01_VARIANT_RUNS = {"DOUBLE": 2, "TRIPLE": 3}

# Parameter that must equal the target question's gold value.
GOLD_PARAMETER = {
    "EC_CORRECTION_SAME_TURN": "final_value",
    "EC_CORRECTION_CROSS_TURN": "final_value",
    "EC_MULTIPLE_CORRECTIONS": "final_value",
    "EC_EMPLOYEE_READBACK_ERROR": "final_value",
    "EC_ONE_FIELD_READBACK_ERROR": "final_value",
    "EC_MULTIPLE_CURRENCIES": "final_currency",
    "EC_MULTIPLE_MONETARY_VALUES": "final_transaction_amount",
    "EC_PREVIOUS_CURRENCY_REFERENCE": "final_currency",
    "EC_PRODUCT_REFERENCE": "selected_product",
    "EC_PREVIOUS_QUANTITY_REFERENCE": "final_quantity",
    "EC_DOUBLE_TRIPLE_DIGITS": "canonical_value",
    "EC_OH_VS_ZERO": "canonical_value",
    "EC_FINANCIAL_SHORTHAND": "canonical_value",
    "EC_LARGE_NUMBER_EXPRESSION": "canonical_value",
    "EC_FALSE_START": "final_value",
}
# Parameters holding superseded or wrong values that must differ from gold.
SUPERSEDED_PARAMETER = {
    "EC_CORRECTION_SAME_TURN": "initial_value",
    "EC_CORRECTION_CROSS_TURN": "initial_value",
    "EC_MULTIPLE_CORRECTIONS": "superseded_values",
    "EC_EMPLOYEE_READBACK_ERROR": "employee_wrong_value",
    "EC_ONE_FIELD_READBACK_ERROR": "employee_wrong_value",
}
ALTERNATIVES_NEEDED = {case_id: 1 for case_id in SUPERSEDED_PARAMETER} | {"EC_MULTIPLE_CORRECTIONS": 2}


class Infeasible(Exception):
    """The current candidate cannot satisfy a constraint; the caller may try another."""


class Balancer:
    """Least-used choice per named decision, with a seeded and stable tie-break."""

    def __init__(self, rng):
        self.rng = rng
        self.counts = Counter()
        self._ranks = {}

    def _rank(self, name, option):
        key = (name, option)
        if key not in self._ranks:
            self._ranks[key] = self.rng.random()
        return self._ranks[key]

    def choose(self, name, options, local=None):
        if not options:
            raise Infeasible(f"no option for {name}")
        local = local or Counter()
        return min(options, key=lambda option: (
            self.counts[(name, option)] + local[(name, option)], self._rank(name, option), str(option),
        ))


class BuildContext:
    """Per-call builder state; balancing picks are committed only on success."""

    def __init__(self, domain, gold, verdicts, rng, balancer, excluded, k01_variant, catalog):
        self.domain, self.gold, self.verdicts, self.rng = domain, gold, verdicts, rng
        self.balancer, self.excluded, self.k01_variant, self.catalog = balancer, excluded, k01_variant, catalog
        self.local = Counter()
        # The catalog language variant of the assignment being built (multilingual releases set it per assignment), and the
        # magnitude factor a variant-free monetary distractor must avoid where the profile plans that contrast deliberately.
        self.language_variant = None
        self.avoided_magnitude_factor = None

    def pick(self, name, options):
        option = self.balancer.choose(name, options, self.local)
        self.local[(name, option)] += 1
        return option

    def commit(self):
        self.balancer.counts.update(self.local)

    def alternatives(self, question_id, count):
        values = self.domain.alternatives(question_id, self.gold, self.excluded.get(question_id, frozenset()))
        if len(values) < count:
            raise Infeasible(f"{question_id}: needs {count} realistic alternative(s), found {len(values)}")
        chosen = []
        for _ in range(count):
            chosen.append(self.pick(f"alternative:{question_id}", [v for v in values if v not in chosen]))
        return chosen


def _correction(ctx, case, targets):
    (target,) = targets
    values = ctx.alternatives(target, ALTERNATIVES_NEEDED[case["id"]])
    key = SUPERSEDED_PARAMETER[case["id"]]
    return {key: values if key == "superseded_values" else values[0], "final_value": ctx.gold[target]}


def _order_replacement(ctx, case, targets):
    orders = ctx.domain.initial_orders(ctx.gold, targets)
    if not orders:
        raise Infeasible(f"no plausible initial order changing {targets}")
    return {
        "initial_order": orders[ctx.rng.randrange(len(orders))],
        "final_order": {qid: ctx.gold[qid] for qid in ctx.domain.order_ids},
    }


def _multiple_currencies(ctx, case, targets):
    final = ctx.gold[ctx.domain.t02]
    roles = [
        role for role in CURRENCY_DISTRACTOR_ROLES
        if not (role in FEE_ROLES and ctx.verdicts[FEE_DISCLOSURE_QUESTION] == "FAIL")
    ]
    return {
        "distractor_currency": ctx.pick("distractor_currency", [c for c in ctx.domain.currencies if c != final]),
        "distractor_role": ctx.pick("currency_distractor_role", roles),
        "final_currency": final,
    }


def magnitude_contrast(variant):
    """The DISTRACTOR_MAGNITUDE_FACTOR constraint of a catalog language variant, or None."""
    constraint = (variant or {}).get("constraint")
    return constraint if constraint and constraint["kind"] == "DISTRACTOR_MAGNITUDE_FACTOR" else None


def _multiple_monetary_values(ctx, case, targets):
    final, currency = ctx.gold[ctx.domain.t03], ctx.gold[ctx.domain.t02]
    distractors, used, partners = [], set(), []
    contrast = magnitude_contrast(ctx.language_variant)
    if contrast:
        # Exactly one distractor is the final amount x or / the factor; the other distractor is never such a partner.
        partners = ctx.domain.magnitude_partners(currency, final, contrast["factor"])
        if not partners:
            raise Infeasible(f"no x{contrast['factor']} partner of {final} {currency}")
        amount = ctx.pick(f"magnitude_partner:{currency}", partners)
        roles = [role for role in MONETARY_DISTRACTOR_ROLES if ctx.domain.balance_distractor_ok(role, amount, final)
                 and (amount > final or role in contrast["divided_distractor_roles"])]
        role = ctx.pick("monetary_distractor_role", roles)
        used.add(amount)
        distractors.append({"amount": amount, "currency": currency, "role": role})
    elif ctx.avoided_magnitude_factor:
        partners = ctx.domain.magnitude_partners(currency, final, ctx.avoided_magnitude_factor)
    for _ in range(2 - len(distractors)):
        options = {
            role: [a for a in ctx.domain.amounts[currency]
                   if a not in used and a not in partners and ctx.domain.balance_distractor_ok(role, a, final)]
            for role in MONETARY_DISTRACTOR_ROLES if role not in {d["role"] for d in distractors}
        }
        role = ctx.pick("monetary_distractor_role", [r for r, values in options.items() if values])
        amount = ctx.pick(f"monetary_distractor_amount:{currency}", options[role])
        used.add(amount)
        distractors.append({"amount": amount, "currency": currency, "role": role})
    return {"distractor_values": distractors, "final_transaction_amount": final}


def _reference(gold_key, question_attr):
    def build(ctx, case, targets):
        return {
            "antecedent_speaker": ctx.pick("antecedent_speaker", case["allowed_antecedent_speakers"]),
            gold_key: ctx.gold[getattr(ctx.domain, question_attr)],
        }
    return build


def _product_reference(ctx, case, targets):
    selected, currency = ctx.gold[ctx.domain.t01], ctx.gold[ctx.domain.t02]
    others = [
        name for name in ctx.domain.product_names
        if name != selected and currency in ctx.domain.products[name]["allowed_currencies"]
    ]
    other = ctx.pick("reference_candidate_product", others)
    candidates = [selected, other]
    ctx.rng.shuffle(candidates)
    return {
        "antecedent_speaker": ctx.pick("antecedent_speaker", case["allowed_antecedent_speakers"]),
        "candidate_products": candidates,
        "selected_product": selected,
    }


def _double_triple(ctx, case, targets):
    return {"canonical_value": ctx.gold[ctx.domain.k01], "variant": ctx.k01_variant}


def _canonical(question_attr, spoken=None):
    def build(ctx, case, targets):
        value = ctx.gold[getattr(ctx.domain, question_attr)]
        params = {"canonical_value": value}
        if spoken is not None:
            params["spoken_form"] = spoken(value)
        return params
    return build


def _false_start(ctx, case, targets):
    (target,) = targets
    return {"final_value": ctx.gold[target]}


def _long_distance(ctx, case, targets):
    return {"min_turn_gap": case["plan_template"]["default_min_turn_gap"], "target_question_ids": list(targets)}


def _no_parameters(ctx, case, targets):
    return {}


BUILDERS = {
    "EC_CORRECTION_SAME_TURN": _correction,
    "EC_CORRECTION_CROSS_TURN": _correction,
    "EC_MULTIPLE_CORRECTIONS": _correction,
    "EC_EMPLOYEE_READBACK_ERROR": _correction,
    "EC_ONE_FIELD_READBACK_ERROR": _correction,
    "EC_COMPLETE_ORDER_REPLACEMENT": _order_replacement,
    "EC_MULTIPLE_CURRENCIES": _multiple_currencies,
    "EC_MULTIPLE_MONETARY_VALUES": _multiple_monetary_values,
    "EC_PREVIOUS_CURRENCY_REFERENCE": _reference("final_currency", "t02"),
    "EC_PRODUCT_REFERENCE": _product_reference,
    "EC_PREVIOUS_QUANTITY_REFERENCE": _reference("final_quantity", "t04"),
    "EC_DOUBLE_TRIPLE_DIGITS": _double_triple,
    "EC_OH_VS_ZERO": _canonical("k01"),
    "EC_FINANCIAL_SHORTHAND": _canonical("t03", financial_shorthand),
    "EC_LARGE_NUMBER_EXPRESSION": _canonical("t03", magnitude_expression),
    "EC_FALSE_START": _false_start,
    "EC_LONG_DISTANCE_DEPENDENCY": _long_distance,
    "EC_GUARANTEE_REQUEST_REFUSAL": _no_parameters,
    "EC_PARAPHRASED_COMPLIANCE_STATEMENT": _no_parameters,
    "EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED": _no_parameters,
    "EC_WEAK_CONFIRMATION": _no_parameters,
}


def missing_builders(catalog):
    return sorted({case["id"] for case in catalog["edge_cases"]} - BUILDERS.keys())
