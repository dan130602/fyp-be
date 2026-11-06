"""Per-call context for G4: the frozen skeleton, the accepted G3 scenario and the placeholder rules of every event.

`PlanContext` is the one place that interprets a skeleton for G4. The brief builder, the response validator, the
compiler and the synthetic-fixture generator all read it, so they cannot disagree about which placeholders an event may,
must or must not use, which forms it may choose, or which value each placeholder renders.
"""

from dataclasses import dataclass

from fyp_evaluation.validation.stage0 import is_multilingual

from .config import QUESTION_PLACEHOLDERS
from .surfaces import SurfaceCatalog

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
class FormChoice:
    """One spoken form a placeholder may take in an event, with the language it speaks and the exact surface Python writes."""

    form: str
    language: str  # normalization spoken_forms[].language ("en" / "zh")
    surface: str


def get_catalog(inputs):
    """The release's SurfaceCatalog (schema 2.0 only), built once per inputs object."""
    if getattr(inputs, "_g4_catalog", None) is None:
        inputs._g4_catalog = SurfaceCatalog(inputs.documents["normalization"])
    return inputs._g4_catalog


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
        self.multilingual = is_multilingual(inputs.documents)  # schema 2.0: language-aware forms, guards and checks
        self.catalog = get_catalog(inputs) if self.multilingual else None
        self.language_requirements = self.spec.get("language_requirements") if self.multilingual else None
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
        if self.multilingual:  # every slot an event must use keeps at least one valid form (raises PlanInputError otherwise)
            for event in self.events:
                rules = self.slot_rules(event)
                if rules.readback:
                    readback = event["readback"]
                    for name in {readback["wrong_question_id"], *readback["must_include_question_ids"]}:
                        self.form_choices(event, name)
                    if len(self.readback_candidates(event)) < readback["min_field_count"]:
                        raise PlanInputError(f"{self.call_id}/{event['event_id']}: fewer than {readback['min_field_count']} fields "
                                             "can be read back in this event's languages")
                for name in rules.required:
                    self.form_choices(event, name) if name in QUESTION_PLACEHOLDERS else self.scenario_forms(event, name)

    # ---- placeholder rules ------------------------------------------------------------------------------
    def readback_candidates(self, event=None):
        """Questions a readback may read back: every extraction question with a FINAL event, in checklist order.

        Schema 2.0 with an `event`: only the questions that still have a valid form in that event's languages (an English-only
        REQUIRED form, for example, cannot be read back in a Mandarin-pinned readback)."""
        candidates = [q for q in QUESTION_PLACEHOLDERS if q in self.final_event]
        if event is None or not self.multilingual:
            return candidates
        return [q for q in candidates if self._speakable(event, q)]

    def _speakable(self, event, name):
        """Whether a placeholder keeps at least one valid form in this event (schema 2.0)."""
        try:
            return bool(self.form_choices(event, name) if name in QUESTION_PLACEHOLDERS else self.scenario_forms(event, name))
        except PlanInputError:
            return False

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
                if self.multilingual:  # an optional restatement with no valid form in this event's languages is not offered
                    optional = [name for name in optional if self._speakable(event, name)]
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

    def allowed_languages(self, event):
        """Languages this event may speak (schema 2.0): its pinned language, else every language of its allowed speakers."""
        if not self.multilingual:
            return None
        if event.get("required_language"):
            return {event["required_language"]}
        speakers = self.language_requirements["speakers"]
        return {language for speaker in event["allowed_speakers"] for language in speakers[speaker]["spoken_languages"]}

    def form_choices(self, event, placeholder):
        """Allowed form IDs for a question placeholder (more than one means the model must choose), or None.

        Schema 1.0: the skeleton's forms. Schema 2.0: those forms narrowed to the event's allowed languages and to the forms
        that can express this particular value; a REQUIRED form that does not survive is a contract conflict.
        """
        binding = self.binding(event, placeholder)
        if binding is None:
            return None
        forms = forms_of(binding[1])
        if forms is None or not self.multilingual:
            return forms
        allowed = self.allowed_languages(event)
        kept = [form for form in forms
                if self.catalog.form_language(placeholder, form) in allowed and self.catalog.form_available(placeholder, binding[2], form)]
        if not kept or (binding[1]["requirement"] == "REQUIRED" and kept != forms):
            raise PlanInputError(f"{self.call_id}/{event['event_id']}: no permitted form for {placeholder} "
                                 f"(skeleton {forms}, languages {sorted(allowed)}, value {binding[2]!r})")
        return kept

    def scenario_forms(self, event, name):
        """Schema 2.0 form IDs for FEE / LIMIT_PRICE (amount forms) and CURRENCY (currency forms) in this event; None on 1.0."""
        if not self.multilingual:
            return None
        allowed = self.allowed_languages(event)
        forms = [form for form in self.catalog.default_forms(name) if self.catalog.form_language(name, form) in allowed]
        if name == "CURRENCY":
            forms = [form for form in forms if self.catalog.form_available(name, event["currency"], form)]
        if not forms:
            raise PlanInputError(f"{self.call_id}/{event['event_id']}: no permitted form for {name} in {sorted(allowed)}")
        return forms

    def choices(self, event, placeholder):
        """[FormChoice] for a question or scenario placeholder (None when it must not be spoken explicitly here).

        The single source of choices: request previews, response validation and rendering all read this list, so they
        cannot disagree. Schema 1.0 keeps its fixed surfaces (languages are None).
        """
        from .analysis import question_surface, scenario_surface

        if placeholder in QUESTION_PLACEHOLDERS:
            binding = self.binding(event, placeholder)
            forms = self.form_choices(event, placeholder)
            if forms is None:
                return None
            return [FormChoice(form, self.catalog.form_language(placeholder, form) if self.multilingual else None,
                               question_surface(self, placeholder, binding[2], form, binding[1])) for form in forms]
        forms = self.scenario_forms(event, placeholder)
        if forms is None:
            return None
        return [FormChoice(form, self.catalog.form_language(placeholder, form), scenario_surface(self, event, placeholder, form)[0])
                for form in forms]

    def scenario_surface_notes(self):
        """Human-readable fee / limit-price facts for the brief (the values themselves stay Python-rendered)."""
        return {"fee": self.fee, "limit_prices": self.limit_prices}

    # ---- derived constraints (D3) -------------------------------------------------------------------------
    def order_edges(self):
        return [(c["before"], c["after"]) for c in self.skeleton["constraints"] if c["type"] == "ORDER"]
