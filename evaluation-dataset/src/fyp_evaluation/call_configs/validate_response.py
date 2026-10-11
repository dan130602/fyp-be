"""Deterministic validation of one G3 authoring response against the call's frozen truth.

Small on purpose: schema, gold-conditioned nullness, exact echo of Python-owned references, structured-value
checks, and a handful of obvious lexical scans. It is not an NLP scanner: number words, paraphrased facts and
semantic leakage are left to the G5/G7 semantic checks.
"""

import re
from decimal import Decimal

from jsonschema import Draft202012Validator

NRIC_LIKE = re.compile(r"\b[STFGM]\d{7}[A-Z]\b")
CURRENCY_SYMBOLS = r"S\$|US\$|[$€¥£]"
INTEGER_SCAN_THRESHOLD = 1000  # smaller integers collide with ordinary prose, so only larger ones are scanned


def _decimal(value):
    return Decimal(str(value))


def _plain(value):
    text = format(_decimal(value).normalize(), "f")
    return text


def _surface_forms(value):
    """Digit renderings of a structured value: plain and thousands-grouped."""
    number = _decimal(value)
    if number == number.to_integral_value():
        integer = int(number)
        return {str(integer), f"{integer:,}"}
    plain = _plain(value)
    whole, _, fraction = plain.partition(".")
    return {plain, f"{int(whole):,}.{fraction}"}


def _is_scannable(value):
    number = _decimal(value)
    return number != number.to_integral_value() or abs(number) >= INTEGER_SCAN_THRESHOLD


def collect_texts(response):
    """[(path, text)] of every free-text field."""
    texts = [(f"client.{k}", response["client"][k]) for k in ("name", "background", "investment_objective", "loss_attitude")]
    texts.append(("employee.name", response["employee"]["name"]))
    texts += [(f"setting.{k}", response["setting"][k]) for k in ("call_reason", "conversation_style")]
    product = response["final_product"]
    texts += [("final_product.description", product["description"]), ("final_product.recommendation_basis", product["recommendation_basis"])]
    texts += [(f"final_product.concrete_risks[{i}]", text) for i, text in enumerate(product["concrete_risks"])]
    texts += [(f"other_products[{i}].description", item["description"]) for i, item in enumerate(response["other_products"])]
    if response["holding_context"] is not None:
        texts.append(("holding_context", response["holding_context"]))
    texts += [(f"red_flags.{k}", v) for k, v in response["red_flags"].items() if v is not None]
    texts += [(f"edge_case_context[{i}].motivation", item["motivation"]) for i, item in enumerate(response["edge_case_context"])]
    texts += [(f"additional_discussion_topics[{i}]", text) for i, text in enumerate(response["additional_discussion_topics"])]
    return texts


class ResponseValidator:
    def __init__(self, response_schema):
        Draft202012Validator.check_schema(response_schema)
        self._schema = Draft202012Validator(response_schema)

    # ---- entry point -----------------------------------------------------------------------------------
    def errors(self, response, ctx):
        """Every violation found in `response` for this call ([] when valid)."""
        schema_errors = [f"schema: {e.json_path}: {e.message}" for e in self._schema.iter_errors(response)]
        if schema_errors:
            return schema_errors  # the rules below assume the schema shape
        errors = []
        errors += self._nonempty(response)
        errors += self._nullness(response, ctx)
        errors += self._references(response, ctx)
        errors += self._structured(response, ctx)
        errors += self._lexical(response, ctx)
        return errors

    # ---- rules -----------------------------------------------------------------------------------------
    @staticmethod
    def _nonempty(response):
        return [f"{path}: must not be empty" for path, text in collect_texts(response) if not text.strip()] + (
            [] if response["final_product"]["concrete_risks"] else ["final_product.concrete_risks: needs at least one risk"])

    @staticmethod
    def _slot(errors, name, required, value):
        if required and value is None:
            errors.append(f"{name}: required for this call but null")
        if not required and value is not None:
            errors.append(f"{name}: must be null for this call")

    def _nullness(self, response, ctx):
        errors = []
        self._slot(errors, "holding_context", ctx.holding_required, response["holding_context"])
        self._slot(errors, "fee_disclosure", ctx.fee_required, response["fee_disclosure"])
        for key in ("final", "initial"):
            state = ctx.order_states.get(key)
            self._slot(errors, f"limit_prices.{key}", bool(state and state["limit_required"]), response["limit_prices"][key])
        for question, required in ctx.red_flag_required.items():
            self._slot(errors, f"red_flags.{question}", required, response["red_flags"][question])
        return errors

    @staticmethod
    def _references(response, ctx):
        errors = []
        refs = [item["product_ref"] for item in response["other_products"]]
        if sorted(refs) != sorted(ctx.other_product_refs):
            errors.append(f"other_products: product_ref set {sorted(refs)} must equal {sorted(ctx.other_product_refs)}")
        ids = [item["assignment_id"] for item in response["edge_case_context"]]
        if sorted(ids) != sorted(ctx.edge_context_ids):
            errors.append(f"edge_case_context: assignment_id set {sorted(ids)} must equal {sorted(ctx.edge_context_ids)}")
        return errors

    @staticmethod
    def _collides(value, currency, ctx):
        return any(c == currency and _decimal(a) == _decimal(value) for c, a in ctx.monetary_values)

    def _structured(self, response, ctx):
        errors = []
        fee = response["fee_disclosure"]
        if fee is not None:
            if fee["value"] <= 0:
                errors.append("fee_disclosure.value: must be positive")
            if fee["basis"] == "RATE" and fee["currency"] is not None:
                errors.append("fee_disclosure.currency: must be null for a RATE")
            if fee["basis"] == "AMOUNT":
                if fee["currency"] not in ctx.fee_currencies:
                    errors.append(f"fee_disclosure.currency: {fee['currency']!r} not in {ctx.fee_currencies}")
                elif fee["value"] > 0 and self._collides(fee["value"], fee["currency"], ctx):
                    errors.append("fee_disclosure.value: equals an order amount or monetary distractor in the same currency")
        for key in ("final", "initial"):
            price, state = response["limit_prices"][key], ctx.order_states.get(key)
            if price is None or state is None:
                continue
            if price <= 0:
                errors.append(f"limit_prices.{key}: must be positive")
                continue
            if not state["min_unit_price"] <= price <= state["max_unit_price"]:
                errors.append(f"limit_prices.{key}: {price} outside the {state['currency']} unit-price range "
                              f"[{state['min_unit_price']}, {state['max_unit_price']}]")
            if self._collides(price, state["currency"], ctx):
                errors.append(f"limit_prices.{key}: equals an order amount or monetary distractor in {state['currency']}")
        return errors

    def _lexical(self, response, ctx):
        errors = []
        names = sorted((p["name"] for p in ctx.products), key=len, reverse=True)
        outsiders = [name for name in ctx.product_pool if name not in {p["name"] for p in ctx.products}]
        allowed = set(ctx.allowed_currencies)
        every_alias = sorted({alias for code in ctx.currency_codes for alias in ctx.currency_aliases[code]} | {"dollars", "dollar"},
                             key=len, reverse=True)
        word_tokens = "|".join(re.escape(token) for token in every_alias)
        before = re.compile(rf"(?<![A-Za-z])(?:{CURRENCY_SYMBOLS}|(?:{word_tokens})\b)\s*\d", re.IGNORECASE)
        after = re.compile(rf"\d\s*(?:{word_tokens})\b", re.IGNORECASE)
        forbidden = []  # (label, compiled regex) for unapproved currency codes and aliases
        for code in ctx.currency_codes:
            if code in allowed:
                continue
            forbidden.append((code, re.compile(rf"\b{code}\b")))
            for alias in ctx.currency_aliases[code]:
                if alias != code:
                    forbidden.append((alias, re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)))
        structured = self._structured_values(response, ctx)
        for path, text in collect_texts(response):
            lowered = text.lower()
            for name in outsiders:
                if name.lower() in lowered:
                    errors.append(f"{path}: mentions product {name!r} that is not part of this call")
            masked = text
            for name in names:  # authorized product names are exempt from currency and number detection
                masked = re.sub(re.escape(name), "§", masked, flags=re.IGNORECASE)
            if NRIC_LIKE.search(masked):
                errors.append(f"{path}: contains an NRIC/FIN-like identifier")
            for label, pattern in forbidden:
                if pattern.search(masked):
                    errors.append(f"{path}: mentions currency {label!r} that is not part of this call")
            if before.search(masked) or after.search(masked):
                errors.append(f"{path}: contains an explicit currency-plus-number expression")
            for label, forms in structured:
                for form in forms:
                    if re.search(rf"(?<![\d.,]){re.escape(form)}(?![\d]|[.,]\d)", masked):
                        errors.append(f"{path}: restates structured value {label} ({form}) in narrative")
                        break
        return errors

    @staticmethod
    def _structured_values(response, ctx):
        """[(label, digit forms)] of the call's structured numbers that must not appear in narrative."""
        labelled = []
        for key, state in ctx.order_states.items():
            labelled += [(f"{key} order amount", state["amount"]), (f"{key} order quantity", state["quantity"])]
        labelled += [("monetary distractor", amount) for _, amount in ctx.monetary_values]
        fee = response["fee_disclosure"]
        if fee is not None and fee["basis"] == "AMOUNT":
            labelled.append(("fee amount", fee["value"]))
        labelled += [(f"{key} limit price", price) for key, price in response["limit_prices"].items() if price is not None]
        seen, result = set(), []
        for label, value in labelled:
            if _is_scannable(value) and (label, str(value)) not in seen:
                seen.add((label, str(value)))
                result.append((label, _surface_forms(value)))
        return result
