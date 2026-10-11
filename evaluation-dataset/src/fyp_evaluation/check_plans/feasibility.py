"""Deterministic synthetic responses and the all-calls feasibility pass.

`synthesize_response` builds a response that satisfies exactly the structural contract of a call's skeleton and nothing
more. It proves STRUCTURAL feasibility (every one of the frozen events can be authored, rendered and compiled, and the
plan projects back to the skeleton) before any paid request. It says nothing about the semantic quality of real model
wording, and it is never sent to a provider nor written as an artifact.
"""

from .analysis import find_cycle
from .compile import assemble, get_lexicon, projection_errors
from .context import forms_of
from .validate_response import PlanResponseValidator

_ACT = {
    ("C01", "ACT"): "Before we go further, could you remind me what you are hoping to achieve with these investments?",
    ("C02", "ACT"): "One real risk with this particular fund is that its unit price can drop when its markets fall.",
    ("C03", "ACT"): "Have those risks come across clearly, and can you tell me in your own words what they mean for you?",
    ("C03", "RESPONSE"): "Yes, the main point is that the price can fall and I could end up with less than I started with.",
    ("C04", "ACT"): "The charge for this dealing is [[FEE]].",
    ("C05", "ACT"): "You should be aware that you could lose some or all of the money you put in.",
    ("C06", "ACT"): "Are you prepared to accept that the investment may lose value?",
    ("C06", "RESPONSE"): "Yes, I am prepared to accept that possibility.",
    ("C07", "ACT"): "I am suggesting this because it fits what you have told me about your needs.",
    ("C08", "ACT"): "So, to confirm the final instruction, shall I go ahead exactly as you described?",
    ("C08", "RESPONSE"): "Yes, please go ahead with that.",
    ("C09", "ACT"): "The money for this came through some arrangements I would rather not document.",
    ("C10", "ACT"): "I heard about this from someone with confidential knowledge, which is why I want to act now.",
    ("C11", "ACT"): "You should say you are comfortable with large losses, whatever you really think.",
    ("C12", "ACT"): "I can promise you that these returns will definitely be achieved.",
}
_STATEMENT = {
    "K01": "My identification number is [[K01]].", "K02": "My date of birth is [[K02]].", "T01": "I want to deal in [[T01]].",
    "T02": "The currency is [[T02]].", "T03": "The amount is [[T03]].", "T04": "The quantity is [[T04]] units.",
    "T05": "I want to [[T05]].", "T06": "I would like [[T06]].",
}
_NO_VALUE = {
    "GUARANTEE_REQUEST": "Can you tell me whether the return is certain?",
    "GUARANTEE_REFUSAL": "I am not able to promise any particular outcome for the returns.",
    "WEAK_ACKNOWLEDGEMENT": "Mm, okay, I see.",
    "ORDER_INVALIDATION": "Please ignore my earlier instruction, I am replacing it.",
    "REFERENCE_SELECTION": "Let us go with the first one you mentioned.",
    "FALSE_START": "I would like to place a—",
}


def _value_text(event, required):
    kind, questions = event["kind"], [name for name in required if name not in ("LIMIT_PRICE", "CURRENCY")]
    if kind == "ORDER_STATEMENT":
        text = "Please process this order: [[T05]] [[T04]] units of [[T01]], amount [[T03]], currency [[T02]], as [[T06]]"
    elif kind == "VALUE_CORRECTION":
        text = "Sorry, I need to correct that, it should be " + " and ".join(f"[[{q}]]" for q in questions)
    elif kind == "DISTRACTOR_MENTION":
        text = "For reference only, something else came up: " + " and ".join(f"[[{q}]]" for q in questions)
        if "CURRENCY" in required:
            text += " in [[CURRENCY]]"
    elif kind == "ANTECEDENT_MENTION":
        text = "We also talked earlier about " + " and ".join(f"[[{q}]]" for q in questions)
    else:
        text = " ".join(_STATEMENT[q] for q in questions)
    if "LIMIT_PRICE" in required:
        text += ", with a limit price of [[LIMIT_PRICE]]"
    return text if text.endswith((".", "?")) else text + "."


def _entry(ctx, event, form_offset, speaker_offset, readback_ids=None):
    rules = ctx.slot_rules(event)
    allowed = event["allowed_speakers"]
    speaker = None if len(allowed) == 1 else allowed[speaker_offset % len(allowed)]
    kind, forms, readback = event["kind"], [], None
    if rules.readback:
        readback = readback_ids
        text = "Let me read that back to you: " + "; ".join(f"[[{q}]]" for q in readback) + "."
        used = list(readback)
    elif kind == "COMPLIANCE_ACT":
        text = _ACT[(event["question_ids"][0], event["role"])]
        used = list(rules.required)
    elif kind in rules_value_kinds():
        text = _value_text(event, rules.required)
        used = list(rules.required)
    else:
        text, used = _NO_VALUE[kind], []
    for index, name in enumerate(used):
        if name in ctx.checks:
            choices = ctx.form_choices(event, name)
            if choices and len(choices) > 1:
                forms.append({"placeholder": name, "form": choices[(form_offset + index) % len(choices)]})
    return {"event_id": event["event_id"], "speaker": speaker, "text": text, "value_forms": forms, "readback_question_ids": readback}


def rules_value_kinds():
    from .context import VALUE_KINDS

    return VALUE_KINDS


def _readback_selection(ctx, event, taken_edges):
    readback = event["readback"]
    chosen = list(dict.fromkeys(readback["must_include_question_ids"]))
    for question in ctx.readback_candidates():
        if len(chosen) >= readback["min_field_count"]:
            break
        if question in chosen:
            continue
        edge = (ctx.final_event[question], event["event_id"])
        if find_cycle(ctx.order_edges() + taken_edges + [edge]) is None:
            chosen.append(question)
    for question in chosen:
        if question != readback["wrong_question_id"]:
            taken_edges.append((ctx.final_event[question], event["event_id"]))
    return [q for q in ctx.readback_candidates() if q in chosen]


def synthesize_response(ctx, *, form_offset=0, speaker_offset=0):
    taken, entries = [], []
    for event in ctx.events:
        ids = _readback_selection(ctx, event, taken) if event["kind"] == "READBACK" else None
        entries.append(_entry(ctx, event, form_offset, speaker_offset, ids))
    return {"events": entries}


def feasibility_report(builder, validator, inputs):
    """{call_id: [errors]} for the synthetic response of every call (empty lists mean structurally feasible)."""
    report = {}
    for call_id in inputs.call_ids:
        ctx = builder.context(call_id)
        report[call_id] = validator.errors(synthesize_response(ctx), ctx)
    return report


def make_validator(builder):
    return PlanResponseValidator(builder.response_schema, builder.artifact_schema, builder.inputs)
