"""Per-call context for G4: the frozen skeleton, the accepted G3 scenario and the placeholder rules of every event.

`PlanContext` is the one place that interprets a skeleton for G4. The brief builder, the response validator, the
compiler and the synthetic-fixture generator all read it, so they cannot disagree about which placeholders an event may,
must or must not use, which forms it may choose, or which value each placeholder renders.
"""

from dataclasses import dataclass

from .config import QUESTION_PLACEHOLDERS

ACT_GOLD_PLACEHOLDERS = {  # D2: controlled gold restatements in employee compliance acts (role ACT only)
    "C08": ("T01", "T02", "T03", "T04", "T05", "T06"),
    "C02": ("T01",), "C03": ("T01",), "C05": ("T01",), "C07": ("T01",),
}
VALUE_KINDS = ("VALUE_STATEMENT", "VALUE_CORRECTION", "DISTRACTOR_MENTION", "ANTECEDENT_MENTION", "ORDER_STATEMENT")
NO_PLACEHOLDER_KINDS = ("REFERENCE_SELECTION", "FALSE_START", "GUARANTEE_REQUEST", "GUARANTEE_REFUSAL",
                        "WEAK_ACKNOWLEDGEMENT", "ORDER_INVALIDATION")


class PlanInputError(ValueError):
    """The authoritative upstream inputs contradict each other; G4 cannot proceed for this call."""


@dataclass(frozen=True)
class SlotRules:
    required: tuple  # placeholders that must appear exactly once
    optional: tuple  # placeholders that may appear at most once
    readback: bool  # the model chooses the included fields (READBACK events)

    def allowed(self):
        return set(self.required) | set(self.optional)


def forms_of(spoken_form):
    """Allowed form IDs of a spoken-form requirement, or None when the value must not be spoken explicitly."""
    requirement = spoken_form["requirement"]
    if requirement == "ONE_OF":
        return list(spoken_form["forms"])
    if requirement == "REQUIRED":
        return [spoken_form["form"]]
    return None


class PlanContext:
    def __init__(self, inputs, call_id):
        self.inputs, self.call_id = inputs, call_id
        self.skeleton, self.config = inputs.skeletons[call_id], inputs.configs[call_id]
        self.spec = inputs.specs[call_id]
        self.scenario = self.config["scenario"]
        self.events = self.skeleton["planned_events"]
        self.by_id = {event["event_id"]: event for event in self.events}
        self.checks = {check["question_id"]: check for check in self.skeleton["checks"]}
        self.gold = {entry["question_id"]: entry["expected"] for entry in self.spec["gold"]}
        self.aliases = inputs.currency_profile["aliases"]
        self.semantics = {entry["question_id"]: entry for entry in inputs.documents["check_semantics"]["checks"]}
        self.checklist = {item["id"]: item for item in inputs.documents["checklist"]}
        self.fee, self.limit_prices = self.scenario["fee_disclosure"], self.scenario["limit_prices"]
        self.product_pool = list(inputs.product_category)
        self.final_event = {q: c["final_event_id"] for q, c in self.checks.items() if c["kind"] == "EXTRACTION"}
        self.limit_currency = {"final": self.gold["T02"]["expected_value"]}
        for edge in self.spec["edge_cases"]:
            if edge["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT":
                self.limit_currency["initial"] = edge["parameters"]["initial_order"]["T02"]
        self._check_inputs()

    # ---- facts ---------------------------------------------------------------------------------------
    def verdict(self, question_id):
        return self.checks[question_id]["expected"]["expected_verdict"]

    def gold_value(self, question_id):
        return self.gold[question_id]["expected_value"]

    def limit_slot(self, event):
        """'final' / 'initial' when the event states a LIMIT order that must carry the scenario limit price, else None."""
        t06 = event.get("values", {}).get("T06")
        if not t06 or t06["spoken_form"]["requirement"] == "NOT_EXPLICIT" or t06["value"] != "LIMIT":
            return None
        if event["status"] == "FINAL":
            return "final"
        if event["kind"] == "ORDER_STATEMENT" and event["status"] == "SUPERSEDED":
            return "initial"
        return None

    def _check_inputs(self):
        c04_acts = [e for e in self.events if e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C04"]]
        if self.verdict("C04") == "PASS":
            if len(c04_acts) != 1 or self.fee is None:
                raise PlanInputError(f"{self.call_id}: C04 PASS needs exactly one C04 act and a G3 fee_disclosure")
        elif c04_acts or self.fee is not None:
            raise PlanInputError(f"{self.call_id}: C04 FAIL must have no C04 event and no G3 fee_disclosure")
        for event in self.events:
            slot = self.limit_slot(event)
            if slot and self.limit_prices.get(slot) is None:
                raise PlanInputError(f"{self.call_id}/{event['event_id']}: a {slot} LIMIT order needs a G3 limit price")
            for question in event.get("values", {}):
                if question not in self.checks:
                    raise PlanInputError(f"{self.call_id}/{event['event_id']}: value for unknown question {question}")

    # ---- placeholder rules ------------------------------------------------------------------------------
    def readback_candidates(self):
        """Questions a readback may read back: every extraction question with a FINAL event, in checklist order."""
        return [q for q in QUESTION_PLACEHOLDERS if q in self.final_event]

    def slot_rules(self, event):
        kind, questions = event["kind"], event["question_ids"]
        if kind == "READBACK":
            return SlotRules((), (), True)
        if kind in VALUE_KINDS:
            required = [q for q, v in event.get("values", {}).items() if forms_of(v["spoken_form"]) is not None]
            if "currency" in event:
                required.append("CURRENCY")
            if self.limit_slot(event):
                required.append("LIMIT_PRICE")
            return SlotRules(tuple(required), (), False)
        if kind == "COMPLIANCE_ACT" and event.get("role") == "ACT":
            question = questions[0]
            if question == "C04":
                return SlotRules(("FEE",), (), False)
            if question in ACT_GOLD_PLACEHOLDERS:
                optional = list(ACT_GOLD_PLACEHOLDERS[question])
                if question == "C08" and self.gold_value("T06") == "LIMIT":
                    optional.append("LIMIT_PRICE")
                return SlotRules((), tuple(optional), False)
        return SlotRules((), (), False)

    def binding(self, event, placeholder):
        """(role, spoken_form, value) for a question placeholder in this event, or None when it has no value here."""
        if event["kind"] == "READBACK":
            readback = event["readback"]
            if placeholder == readback["wrong_question_id"]:
                return "READBACK_WRONG", readback["wrong_value_spoken_form"], readback["wrong_value"]
            return "READBACK_FINAL", self.checks[placeholder]["gold_restatement"]["spoken_form"], self.gold_value(placeholder)
        own = event.get("values", {}).get(placeholder)
        if own is not None:
            return "EVENT_VALUE", own["spoken_form"], own["value"]
        if placeholder in self.checks and self.checks[placeholder]["kind"] == "EXTRACTION":
            return "GOLD_RESTATEMENT", self.checks[placeholder]["gold_restatement"]["spoken_form"], self.gold_value(placeholder)
        return None

    def form_choices(self, event, placeholder):
        """Allowed form IDs for a question placeholder (more than one means the model must choose), or None."""
        binding = self.binding(event, placeholder)
        return None if binding is None else forms_of(binding[1])

    def scenario_surface_notes(self):
        """Human-readable fee / limit-price facts for the brief (the values themselves stay Python-rendered)."""
        return {"fee": self.fee, "limit_prices": self.limit_prices}

    # ---- derived constraints (D3) -------------------------------------------------------------------------
    def order_edges(self):
        return [(c["before"], c["after"]) for c in self.skeleton["constraints"] if c["type"] == "ORDER"]
