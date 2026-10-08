"""Closed vocabularies, the per-call context derived from G0, and the plan builder.

Nothing here invents a business value: gold comes from the call spec, and every spoken-form,
speaker, ordering and gap fact is read from the frozen G0 documents or the G1 parameters
carried in the call spec. The few structural conventions that G0 leaves implicit are marked
CONVENTION and listed in the README.
"""

from fyp_evaluation.call_specs.compile import parse_json_text  # noqa: F401  (re-exported for callers)

EVENT_KINDS = (
    "COMPLIANCE_ACT", "VALUE_STATEMENT", "VALUE_CORRECTION", "READBACK", "ORDER_STATEMENT",
    "ORDER_INVALIDATION", "DISTRACTOR_MENTION", "ANTECEDENT_MENTION", "REFERENCE_SELECTION",
    "FALSE_START", "GUARANTEE_REQUEST", "GUARANTEE_REFUSAL", "WEAK_ACKNOWLEDGEMENT",
)
EVENT_STATUSES = ("FINAL", "STATED", "SUPERSEDED", "REJECTED", "DISTRACTOR", "ANTECEDENT", "ABANDONED", "CONTEXT")
CONSTRAINT_TYPES = ("ORDER", "SAME_TURN", "NOT_SAME_TURN", "MIN_INTERVENING_TURNS")
KIND_STATUSES = {
    "COMPLIANCE_ACT": {"CONTEXT"},
    "VALUE_STATEMENT": {"FINAL", "STATED", "SUPERSEDED"},
    "VALUE_CORRECTION": {"FINAL", "SUPERSEDED"},
    "READBACK": {"REJECTED"},
    "ORDER_STATEMENT": {"FINAL", "SUPERSEDED"},
    "ORDER_INVALIDATION": {"CONTEXT"},
    "DISTRACTOR_MENTION": {"DISTRACTOR"},
    "ANTECEDENT_MENTION": {"ANTECEDENT"},
    "REFERENCE_SELECTION": {"FINAL"},
    "FALSE_START": {"ABANDONED"},
    "GUARANTEE_REQUEST": {"CONTEXT"},
    "GUARANTEE_REFUSAL": {"CONTEXT"},
    "WEAK_ACKNOWLEDGEMENT": {"CONTEXT"},
}
VALUE_KINDS = frozenset({
    "VALUE_STATEMENT", "VALUE_CORRECTION", "ORDER_STATEMENT", "DISTRACTOR_MENTION",
    "ANTECEDENT_MENTION", "REFERENCE_SELECTION",
})
CLIENT, EMPLOYEE = "CLIENT", "EMPLOYEE"
BOTH = [CLIENT, EMPLOYEE]
EVENT_FIELD_ORDER = (
    "event_id", "kind", "status", "allowed_speakers", "question_ids", "assignment_ids", "role", "condition_ref",
    "values", "corrects_event_id", "invalidates_event_id", "refers_to_event_id", "readback", "distractor_role",
    "currency", "incomplete", "strength",
)
NOT_EXPLICIT = {"requirement": "NOT_EXPLICIT"}


def same_scalar(left, right):
    """Equality that also distinguishes JSON types (1 vs 1.0 vs true)."""
    return type(left) is type(right) and left == right


class Context:
    """Everything one call's compilation needs, derived from the call spec and the frozen G0 documents."""

    def __init__(self, spec, documents):
        self.spec, self.documents = spec, documents
        self.checklist = documents["checklist"]
        self.items = {item["id"]: item for item in self.checklist}
        self.semantics = {e["question_id"]: e for e in documents["check_semantics"]["checks"]}
        self.ordering_rules = documents["check_semantics"]["ordering_rules"]
        self.catalog = {case["id"]: case for case in documents["edge_case_catalog"]["edge_cases"]}
        self.profiles = documents["normalization"]["profiles"]
        self.expected = {g["question_id"]: g["expected"] for g in spec["gold"]}
        self.gold = {q: e.get("expected_verdict", e.get("expected_value")) for q, e in self.expected.items()}
        self.extraction_ids = [i["id"] for i in self.checklist if i["type"] != "COMPLIANCE"]
        self.compliance_ids = [i["id"] for i in self.checklist if i["type"] == "COMPLIANCE"]
        self.order_ids = [i["id"] for i in self.checklist if i["type"] == "TRANSACTION"]
        self.assignments = spec["edge_cases"]
        self.profile_of = {q: self.semantics[q]["normalization_profile"] for q in self.extraction_ids}
        self._spoken_forms()

    # ---- spoken forms: derived entirely from normalization.json and the assignments ------------
    def default_forms(self, q):
        return [f["id"] for f in self.profiles[self.profile_of[q]]["spoken_forms"] if f["usage"] in {"DEFAULT", "PERMITTED"}]

    def _spoken_forms(self):
        self.edge_form = {}
        for a in self.assignments:
            for q in a["target_question_ids"]:
                if q not in self.profile_of:
                    continue
                for form in self.profiles[self.profile_of[q]]["spoken_forms"]:
                    if form.get("enabled_by_edge_case") == a["edge_case_id"]:
                        if q in self.edge_form:
                            raise ValueError(f"{self.spec['call_id']}/{q}: more than one spoken-form edge case")
                        self.edge_form[q] = (form, a)
        self.non_gold_form, self.gold_form, self.forbidden_forms = {}, {}, {}
        for q in self.extraction_ids:
            permitted = {"requirement": "ONE_OF", "forms": self.default_forms(q)}
            self.non_gold_form[q] = permitted
            self.gold_form[q] = self._required(*self.edge_form[q]) if q in self.edge_form else permitted
            assigned = {self.edge_form[q][0]["id"]} if q in self.edge_form else set()
            self.forbidden_forms[q] = [
                f["id"] for f in self.profiles[self.profile_of[q]]["spoken_forms"]
                if f["usage"] == "EDGE_CASE_ONLY" and f["id"] not in assigned
            ]

    @staticmethod
    def _required(form, assignment):
        result = {"requirement": "REQUIRED", "form": form["id"], "applies_to": form["applies_to"]}
        if "remaining_characters_form" in form:
            result["remaining_characters_form"] = form["remaining_characters_form"]
        parameters = assignment["parameters"]
        if "variant" in parameters:
            result["variant"] = parameters["variant"]
        if "spoken_form" in parameters:
            result["text"] = parameters["spoken_form"]
        return result

    # ---- compliance derivation ------------------------------------------------------------------
    def compliance_mode(self, q):
        """PRESENCE or ABSENCE, from the checklist requirement type and the frozen verdict."""
        requirement = self.items[q]["requirement_type"]
        passed = self.gold[q] == "PASS"
        return "PRESENCE" if (requirement == "OBLIGATION") == passed else "ABSENCE"

    def condition_ref(self, q):
        """OBLIGATION questions refer to their pass condition; PROHIBITION questions to their fail condition."""
        return f"{q}.{'pass_condition' if self.items[q]['requirement_type'] == 'OBLIGATION' else 'fail_condition'}"


class Plan:
    """Mutable builder for one skeleton: allocates event IDs and records deduplicated constraints."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.events, self.constraints = [], []
        self.final_of = {}
        self.presence_events = {}
        self.wording = {}
        self._constraint_keys = set()

    # ---- events ------------------------------------------------------------------------------------
    def event(self, kind, status, speakers, question_ids, assignment=None, **fields):
        if status not in KIND_STATUSES[kind]:
            raise ValueError(f"{kind} cannot have status {status}")
        event_id = f"EV_{len(self.events) + 1:03d}"
        record = {
            "event_id": event_id, "kind": kind, "status": status, "allowed_speakers": sorted(speakers),
            "question_ids": list(question_ids), "assignment_ids": [assignment["assignment_id"]] if assignment else [],
            **fields,
        }
        self.events.append({key: record[key] for key in EVENT_FIELD_ORDER if key in record})
        if status == "FINAL":
            for q in question_ids:
                if q in self.final_of:
                    raise ValueError(f"{self.ctx.spec['call_id']}/{q}: more than one FINAL event "
                                     f"({self.final_of[q]} and {event_id}); conflicting edge cases")
                self.final_of[q] = event_id
        return event_id

    def get(self, event_id):
        return self.events[int(event_id[3:]) - 1]

    def attach(self, event_ids, assignment):
        for event_id in event_ids:
            ids = self.get(event_id)["assignment_ids"]
            if assignment["assignment_id"] not in ids:
                ids.append(assignment["assignment_id"])
                ids.sort()

    # ---- values and spoken forms --------------------------------------------------------------------
    def values(self, q, value):
        """One explicit value mention: a gold-valued mention follows the gold-restatement policy."""
        ctx = self.ctx
        form = ctx.gold_form[q] if same_scalar(value, ctx.gold[q]) else ctx.non_gold_form[q]
        return {q: {"value": value, "spoken_form": form}}

    def nongold(self, assignment, q, value):
        """A superseded, rejected or distractor value: it must differ from gold."""
        if same_scalar(value, self.ctx.gold[q]):
            raise ValueError(f"{assignment['assignment_id']} ({assignment['edge_case_id']}): "
                             f"{q} value {value!r} equals gold but must differ")
        return self.values(q, value)

    def require_gold(self, assignment, q, value, name="final value"):
        if not same_scalar(value, self.ctx.gold[q]):
            raise ValueError(f"{assignment['assignment_id']} ({assignment['edge_case_id']}): "
                             f"{name} {value!r} contradicts frozen gold {self.ctx.gold[q]!r} for {q}")

    def template(self, assignment):
        return self.ctx.catalog[assignment["edge_case_id"]]["plan_template"]

    # ---- constraints -------------------------------------------------------------------------------
    @staticmethod
    def edge_source(assignment):
        return {"type": "EDGE_ASSIGNMENT", "id": assignment["assignment_id"]}

    def _add(self, constraint):
        key = repr(sorted((k, repr(v)) for k, v in constraint.items()))
        if key not in self._constraint_keys:
            self._constraint_keys.add(key)
            self.constraints.append(constraint)

    def order(self, before, after, source):
        if before == after:
            raise ValueError(f"ORDER constraint between an event and itself ({before})")
        self._add({"type": "ORDER", "before": before, "after": after, "source": source})

    def same_turn(self, first, second, source):
        self._add({"type": "SAME_TURN", "events": sorted([first, second]), "source": source})

    def not_same_turn(self, first, second, source):
        self._add({"type": "NOT_SAME_TURN", "events": sorted([first, second]), "source": source})

    def min_gap(self, first, second, turns, source):
        if first == second:
            raise ValueError(f"MIN_INTERVENING_TURNS needs two distinct events; both resolve to {first}")
        self._add({"type": "MIN_INTERVENING_TURNS", "events": sorted([first, second]), "turns": turns, "source": source})
