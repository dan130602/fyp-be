"""Per-call context and required scenario slots, derived only from the call spec and frozen G0.

`CallContext` is the single place that interprets a call spec for G3: the brief builder, the validator and the
synthetic-fixture generator all read it, so they cannot disagree about which slots a call needs.
"""

from fyp_evaluation.validation.stage0 import is_multilingual

from .products import derive_products

COMPLIANCE_FAIL = "FAIL"


def _currency_values(edge, targets):
    """Currency codes named by one assignment, for the allowed-currency set."""
    params, found = edge["parameters"], []
    if "T02" in targets or edge["edge_case_id"] == "EC_MULTIPLE_CURRENCIES":
        for key in ("initial_value", "final_value", "employee_wrong_value", "final_currency", "distractor_currency"):
            if isinstance(params.get(key), str):
                found.append(params[key])
        for value in params.get("superseded_values", []):
            if isinstance(value, str):
                found.append(value)
    for key in ("initial_order", "final_order"):
        if key in params:
            found.append(params[key]["T02"])
    for item in params.get("distractor_values", []):
        found.append(item["currency"])
    return found


class CallContext:
    """Everything G3 needs to know about one call, computed once."""

    def __init__(self, inputs, call_id):
        spec = inputs.specs[call_id]
        self.call_id, self.spec = call_id, spec
        self.multilingual = is_multilingual(inputs.documents)
        if self.multilingual:  # spoken names are required in every language each speaker speaks (frozen call spec)
            requirements = spec["language_requirements"]
            self.language_requirements = requirements
            self.name_languages = {speaker.lower(): list(requirements["speakers"][speaker]["spoken_languages"])
                                   for speaker in ("CLIENT", "EMPLOYEE")}
            names = inputs.documents["normalization"]["profiles"]["synthetic_product_name"].get("language_names") or {}
            self.zh_product_names = sorted(entry["zh"]["surface"] for entry in names.values() if entry.get("zh"))
        self.gold = {entry["question_id"]: entry["expected"] for entry in spec["gold"]}
        self.edges = spec["edge_cases"]
        self.products = derive_products(spec, inputs.product_category)
        self.price_ranges = inputs.price_ranges
        profile = inputs.currency_profile
        self.currency_codes = list(profile["canonical"]["allowed_values"])
        self.currency_aliases = profile["aliases"]
        self.product_pool = list(inputs.product_category)

        replacement = next((e for e in self.edges if e["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT"), None)
        self.order_states = {"final": self._state(self._final_order())}
        if replacement is not None:
            self.order_states["initial"] = self._state(replacement["parameters"]["initial_order"])

        self.fee_required = self.verdict("C04") == "PASS"
        distractor_fee = [e["parameters"]["distractor_currency"] for e in self.edges
                          if e["edge_case_id"] == "EC_MULTIPLE_CURRENCIES"
                          and e["parameters"].get("distractor_role") == "FEE_CURRENCY"]
        self.fee_currencies = list(dict.fromkeys([self.gold["T02"]["expected_value"]] + distractor_fee))
        self.monetary_values = self._monetary_values()
        self.holding_required = any(state["direction"] == "SELL" for state in self.order_states.values())
        self.red_flag_required = {q: self.verdict(q) == COMPLIANCE_FAIL for q in ("C09", "C10")}
        self.other_product_refs = [p["product_ref"] for p in self.products if "FINAL" not in p["roles"]]
        self.edge_context_ids = [
            e["assignment_id"] for e in self.edges
            if e["family"] != "SPOKEN_NUMERIC_NORMALISATION"
            and e["edge_case_id"] not in ("EC_LONG_DISTANCE_DEPENDENCY", "EC_PARAPHRASED_COMPLIANCE_STATEMENT")]
        allowed = list(self.fee_currencies)
        for edge in self.edges:
            allowed += _currency_values(edge, edge["target_question_ids"])
        allowed.append(self.gold["T02"]["expected_value"])
        self.allowed_currencies = [c for c in self.currency_codes if c in set(allowed)]

    def verdict(self, question_id):
        return self.gold[question_id]["expected_verdict"]

    def value(self, question_id):
        return self.gold[question_id]["expected_value"]

    def _final_order(self):
        return {q: self.value(q) for q in ("T01", "T02", "T03", "T04", "T05", "T06")}

    def _state(self, order):
        currency, amount, quantity = order["T02"], order["T03"], order["T04"]
        bounds = self.price_ranges[currency]
        return {
            "direction": order["T05"], "order_type": order["T06"], "currency": currency, "amount": amount, "quantity": quantity,
            "implied_unit_price": round(amount / quantity, 4), "min_unit_price": bounds["min"], "max_unit_price": bounds["max"],
            "limit_required": order["T06"] == "LIMIT",
        }

    def _monetary_values(self):
        """[(currency, amount)] of every monetary value the call already contains (order amounts and distractors)."""
        values = [(state["currency"], state["amount"]) for state in self.order_states.values()]
        for edge in self.edges:
            for item in edge["parameters"].get("distractor_values", []):
                values.append((item["currency"], item["amount"]))
        return values

    @staticmethod
    def _limit_slot(state):
        if state is None or not state["limit_required"]:
            return {"required": False}
        return {"required": True, "currency": state["currency"], "implied_unit_price": state["implied_unit_price"],
                "min_unit_price": state["min_unit_price"], "max_unit_price": state["max_unit_price"]}

    def required_slots(self):
        """The slot requirements as plain data (brief, validator and fixtures all use this)."""
        return {
            "fee_disclosure": {"required": self.fee_required, "allowed_amount_currencies": self.fee_currencies if self.fee_required else []},
            "limit_prices": {key: self._limit_slot(self.order_states.get(key)) for key in ("final", "initial")},
            "holding_context": {"required": self.holding_required},
            "red_flags": {q: {"required": required} for q, required in self.red_flag_required.items()},
            "other_product_refs": list(self.other_product_refs),
            "edge_case_context_ids": list(self.edge_context_ids),
        }
