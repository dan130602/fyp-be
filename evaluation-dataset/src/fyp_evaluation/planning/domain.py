"""Configured value domains and realism predicates shared by G1 planning and validation.

Everything here derives from the approved G0 documents. Predicates are pure;
enumerations return deterministic, sorted candidate lists.
"""

import re
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction

from fyp_evaluation.validation.dates import ENGLISH_MONTHS
from fyp_evaluation.validation.identification import is_invalid_test_identifier

TRANSACTION_ORDER_FIELDS = (
    "transaction_instrument", "transaction_currency", "transaction_amount",
    "transaction_quantity", "transaction_direction", "order_type",
)
CURRENCY_DISTRACTOR_ROLES = (
    "ACCOUNT_CURRENCY", "EXISTING_HOLDING_CURRENCY", "COMPARISON_CURRENCY", "FEE_CURRENCY",
)
FEE_ROLES = frozenset({"FEE_CURRENCY", "FEE_AMOUNT", "FEE_RATE"})
# No fee role is offered for amounts: a realistic fee value would need a fee schedule.
MONETARY_DISTRACTOR_ROLES = (
    "PORTFOLIO_BALANCE", "ACCOUNT_VALUE", "PREVIOUS_INVESTMENT_AMOUNT", "ILLUSTRATIVE_AMOUNT",
)
BALANCE_ROLES = frozenset({"PORTFOLIO_BALANCE", "ACCOUNT_VALUE"})


def format_dob(day):
    return f"{day.day:02d} {ENGLISH_MONTHS[day.month - 1]} {day.year}"


def parse_dob(value):
    day, month, year = value.split(" ")
    return date(int(year), ENGLISH_MONTHS.index(month) + 1, int(day))


def maximal_digit_runs(identifier):
    """Lengths of maximal runs of identical consecutive digits."""
    return [len(match.group(0)) for match in re.finditer(r"([0-9])\1*", identifier)]


def _decimal_text(value):
    return f"{value.normalize():f}"


def financial_shorthand(amount):
    """'50k' below one million, otherwise '1.2m'."""
    if amount >= 1_000_000:
        return _decimal_text(Decimal(amount) / Decimal(1_000_000)) + "m"
    return _decimal_text(Decimal(amount) / Decimal(1000)) + "k"


def magnitude_expression(amount):
    """'1.2 million' for amounts of at least one million."""
    return _decimal_text(Decimal(amount) / Decimal(1_000_000)) + " million"


def parse_magnitude(spoken, magnitude_words):
    """Parse '<number><word>' or '<number> <word>' using the profile's magnitude words."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?) ?([A-Za-z]+)", spoken)
    if match is None or match.group(2) not in magnitude_words:
        return None
    value = Decimal(match.group(1)) * magnitude_words[match.group(2)]
    return int(value) if value == value.to_integral_value() else None


class Domain:
    """Release value domains, keyed by checklist question ID."""

    def __init__(self, documents):
        self.checklist = documents["checklist"]
        plan = documents["generation_plan"]
        extraction = plan["extraction_distribution"]
        self.question = {item["field"]: item["id"] for item in self.checklist if item["type"] != "COMPLIANCE"}
        self.compliance_ids = [item["id"] for item in self.checklist if item["type"] == "COMPLIANCE"]
        self.extraction_ids = [item["id"] for item in self.checklist if item["type"] != "COMPLIANCE"]
        self.order_ids = [self.question[field] for field in TRANSACTION_ORDER_FIELDS]
        self.k01 = self.question["client_identification_number"]
        self.k02 = self.question["dob"]
        (self.t01, self.t02, self.t03, self.t04, self.t05, self.t06) = self.order_ids
        self.products = {product["name"]: product for product in extraction["product_pool"]}
        self.product_names = [product["name"] for product in extraction["product_pool"]]
        # A 1.0 plan names its vocabularies through its count maps; a multilingual 2.0 plan lists them once (counts are per profile).
        self.currencies = list(extraction.get("transaction_currencies") or extraction["transaction_currency_counts"])
        self.directions = list(extraction.get("transaction_directions") or extraction["transaction_direction_counts"])
        self.order_types = list(extraction.get("order_types") or extraction["order_type_counts"])
        self.amounts = extraction["transaction_amount_values_by_currency"]
        self.quantities = extraction["transaction_quantity_values"]
        self.ranges = {
            currency: (Fraction(str(bounds["min"])), Fraction(str(bounds["max"])))
            for currency, bounds in extraction["amount_quantity_coherence"]["ranges"].items()
        }
        kyc = extraction["synthetic_kyc"]
        self.nric = kyc["nric_fin"]
        self.dob_min = date.fromisoformat(kyc["dob"]["min_date"])
        self.dob_max = date.fromisoformat(kyc["dob"]["max_date"])
        profiles = documents["normalization"]["profiles"]
        profile_of = {
            entry["question_id"]: entry["normalization_profile"]
            for entry in documents["check_semantics"]["checks"] if "normalization_profile" in entry
        }
        self.magnitude_words = {
            word: Decimal(value) for word, value in profiles[profile_of[self.t03]]["magnitude_words"].items()
        }

    # Order realism -----------------------------------------------------
    def coherent(self, currency, amount, quantity):
        low, high = self.ranges[currency]
        return low <= Fraction(amount, quantity) <= high

    def plausible_order(self, order):
        """True when product/currency/amount/quantity/direction/type form a configured, coherent order."""
        product = self.products.get(order[self.t01])
        currency, amount, quantity = order[self.t02], order[self.t03], order[self.t04]
        return (
            product is not None and currency in product["allowed_currencies"]
            and amount in self.amounts.get(currency, []) and quantity in self.quantities
            and type(amount) is int and type(quantity) is int
            and self.coherent(currency, amount, quantity)
            and order[self.t05] in self.directions and order[self.t06] in self.order_types
        )

    def coherent_pairs(self, currency):
        return [
            (amount, quantity)
            for amount in self.amounts[currency] for quantity in self.quantities
            if self.coherent(currency, amount, quantity)
        ]

    def field_values(self, question_id):
        """Every configured value of one transaction field, in configuration order."""
        if question_id == self.t01:
            return list(self.product_names)
        if question_id == self.t02:
            return list(self.currencies)
        if question_id == self.t03:
            return sorted({amount for values in self.amounts.values() for amount in values})
        if question_id == self.t04:
            return list(self.quantities)
        if question_id == self.t05:
            return list(self.directions)
        if question_id == self.t06:
            return list(self.order_types)
        raise ValueError(f"{question_id} is not a transaction order field")

    # KYC validity ------------------------------------------------------
    def valid_k01(self, value):
        return (
            isinstance(value, str) and is_invalid_test_identifier(value)
            and value[0] in self.nric["prefixes"] and value[-1] in self.nric["suffix_letters"]
            and len(value) == self.nric["digit_count"] + 2
        )

    def valid_k02(self, value):
        try:
            parsed = parse_dob(value)
        except (ValueError, AttributeError):
            return False
        return format_dob(parsed) == value and self.dob_min <= parsed <= self.dob_max

    # Realistic alternatives (single field changed, everything else at gold) --
    def alternatives(self, question_id, gold, excluded=frozenset()):
        """Sorted realistic values for one field that differ from that field's gold."""
        final = gold[question_id]
        if question_id == self.k01:
            candidates = {
                final[:index] + digit + final[index + 1:]
                for index in range(1, 1 + self.nric["digit_count"])
                for digit in "0123456789" if digit != final[index]
            }
            return sorted(value for value in candidates if self.valid_k01(value) and value not in excluded)
        if question_id == self.k02:
            day = parse_dob(final)
            candidates = set()
            for other_day in range(1, 32):
                try:
                    candidates.add(day.replace(day=other_day))
                except ValueError:
                    pass
            for shift in (-2, -1, 1, 2):
                try:
                    candidates.add(day.replace(year=day.year + shift))
                except ValueError:
                    pass
            values = {format_dob(c) for c in candidates if self.dob_min <= c <= self.dob_max}
            return sorted(value for value in values if value != final and value not in excluded)
        values = []
        for value in self.field_values(question_id):
            if value == final:
                continue
            order = {qid: gold[qid] for qid in self.order_ids}
            order[question_id] = value
            if self.plausible_order(order):
                values.append(value)
        return values

    def initial_orders(self, gold, changed):
        """Plausible initial orders differing from gold exactly on the `changed` fields."""
        orders = [{qid: gold[qid] for qid in self.order_ids}]
        for question_id in changed:
            orders = [
                {**order, question_id: value}
                for order in orders for value in self.field_values(question_id) if value != gold[question_id]
            ]
        return [order for order in orders if self.plausible_order(order)]

    def balance_distractor_ok(self, role, amount, final):
        return amount > final if role in BALANCE_ROLES else amount != final

    def magnitude_partners(self, currency, amount, factor):
        """Configured same-currency amounts equal to `amount` times or divided by `factor` (larger first)."""
        values = self.amounts.get(currency, [])
        partners = [amount * factor] if amount * factor in values else []
        if amount % factor == 0 and amount // factor in values:
            partners.append(amount // factor)
        return partners


def offset_date(start, days):
    return start + timedelta(days=days)
