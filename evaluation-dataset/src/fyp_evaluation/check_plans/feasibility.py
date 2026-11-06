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
# The pilot fixture sentence for C05 shares a five-word run with the checklist; paraphrase-constrained 2.0 calls need another.
_ACT_MULTILINGUAL = {**_ACT, ("C05", "ACT"): "Please bear in mind that part or even all of what you invest could be lost."}
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


# Schema 2.0: the same structural contract in Mandarin. Wording avoids numerals, Latin letters and value words on purpose.
_ZH_ACT = {
    ("C01", "ACT"): "在继续之前，请您说说希望通过这些投资实现什么目标。",
    ("C02", "ACT"): "这只基金有一个实际风险，就是市场下跌时它的单位价格可能下跌。",
    ("C03", "ACT"): "这些风险您听明白了吗？请用您自己的话说说它们对您意味着什么。",
    ("C03", "RESPONSE"): "明白，主要就是价格可能下跌，我最后拿回的钱可能比本金少。",
    ("C04", "ACT"): "这笔交易的费用是[[FEE]]。",
    ("C05", "ACT"): "请您注意，您投入的钱可能部分或全部亏掉。",
    ("C06", "ACT"): "您是否愿意接受这项投资可能亏损的可能性？",
    ("C06", "RESPONSE"): "愿意，我接受这种可能性。",
    ("C07", "ACT"): "我推荐这个产品，是因为它符合您告诉我的需求。",
    ("C08", "ACT"): "那么，为了核对最终指示，我是否完全按您说的办理？",
    ("C08", "RESPONSE"): "对，请这样办理。",
    ("C09", "ACT"): "这笔钱来自一些我不太想写进记录的安排。",
    ("C10", "ACT"): "我是听一位掌握机密信息的人说的，所以想马上行动。",
    ("C11", "ACT"): "您应该说您能接受很大的亏损，不管您真实想法如何。",
    ("C12", "ACT"): "我可以向您保证，这些回报一定能实现。",
}
_ZH_STATEMENT = {
    "K01": "我的身份证号码是[[K01]]。", "K02": "我的出生日期是[[K02]]。", "T01": "我想办理的产品是[[T01]]。",
    "T02": "币种是[[T02]]。", "T03": "金额是[[T03]]。", "T04": "数量是[[T04]]。", "T05": "我想要[[T05]]。", "T06": "订单类型是[[T06]]。",
}
_ZH_NO_VALUE = {
    "GUARANTEE_REQUEST": "您能告诉我这个回报是否有把握吗？",
    "GUARANTEE_REFUSAL": "我没办法承诺任何特定的回报结果。",
    "WEAK_ACKNOWLEDGEMENT": "嗯，我先听着。",
    "ORDER_INVALIDATION": "请忽略我之前的指示，我要换一个。",
    "REFERENCE_SELECTION": "就选您刚才提到的第一只吧。",
    "FALSE_START": "我想要办理一——",
}


def _value_text_zh(event, required):
    kind, questions = event["kind"], [name for name in required if name not in ("LIMIT_PRICE", "CURRENCY")]
    if kind == "ORDER_STATEMENT":
        text = "请按这样办理这笔订单：方向[[T05]]，数量[[T04]]，产品[[T01]]，金额[[T03]]，币种[[T02]]，类型[[T06]]"
    elif kind == "VALUE_CORRECTION":
        text = "不好意思，我需要更正一下，应该是" + "和".join(f"[[{q}]]" for q in questions)
    elif kind == "DISTRACTOR_MENTION":
        text = "仅供参考，另外提到过" + "和".join(f"[[{q}]]" for q in questions)
        if "CURRENCY" in required:
            text += "，币种是[[CURRENCY]]"
    elif kind == "ANTECEDENT_MENTION":
        text = "我们之前也聊到过" + "和".join(f"[[{q}]]" for q in questions)
    else:
        text = "".join(_ZH_STATEMENT[q] for q in questions)
    if "LIMIT_PRICE" in required:
        text += "，限价是[[LIMIT_PRICE]]"
    return text if text.endswith(("。", "？")) else text + "。"


def _event_language(ctx, event, position):
    """The language an event's synthetic wording uses: its pin, else its only language, else alternating for mixed calls."""
    allowed = ctx.allowed_languages(event)
    if not allowed or allowed == {"en"}:
        return "en"
    if allowed == {"zh"}:
        return "zh"
    return ("zh", "en")[position % 2]


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


def _entry(ctx, event, form_offset, speaker_offset, readback_ids=None, position=0):
    rules = ctx.slot_rules(event)
    allowed = event["allowed_speakers"]
    speaker = None if len(allowed) == 1 else allowed[speaker_offset % len(allowed)]
    kind, forms, readback = event["kind"], [], None
    zh = ctx.multilingual and _event_language(ctx, event, position) == "zh"
    if rules.readback:
        readback = readback_ids
        text = (("我把内容跟您复述一遍：" + "；".join(f"[[{q}]]" for q in readback) + "。") if zh
                else "Let me read that back to you: " + "; ".join(f"[[{q}]]" for q in readback) + ".")
        used = list(readback)
    elif kind == "COMPLIANCE_ACT":
        text = (_ZH_ACT if zh else _ACT_MULTILINGUAL if ctx.multilingual else _ACT)[(event["question_ids"][0], event["role"])]
        used = list(rules.required)
    elif kind in rules_value_kinds():
        text = _value_text_zh(event, rules.required) if zh else _value_text(event, rules.required)
        used = list(rules.required)
    else:
        text, used = (_ZH_NO_VALUE if zh else _NO_VALUE)[kind], []
    for index, name in enumerate(used):
        if ctx.multilingual:
            options = ctx.form_choices(event, name) if name in ctx.checks else ctx.scenario_forms(event, name)
            if options and len(options) > 1:  # prefer a form in the wording's language, rotating through the rest
                preferred = [f for f in options if ctx.catalog.form_language(name, f) == ("zh" if zh else "en")] or options
                forms.append({"placeholder": name, "form": preferred[(form_offset + index) % len(preferred)]})
        elif name in ctx.checks:
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
    for question in ctx.readback_candidates(event):
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
    return [q for q in ctx.readback_candidates(event) if q in chosen]


def synthesize_response(ctx, *, form_offset=0, speaker_offset=0):
    taken, entries = [], []
    for position, event in enumerate(ctx.events):
        ids = _readback_selection(ctx, event, taken) if event["kind"] == "READBACK" else None
        entries.append(_entry(ctx, event, form_offset, speaker_offset, ids, position))
    return {"events": entries}


def feasibility_report(builder, validator, inputs):
    """{call_id: [errors]} for the synthetic response of every call (empty lists mean structurally feasible)."""
    report = {}
    for call_id in getattr(inputs, "ready_call_ids", inputs.call_ids):
        ctx = builder.context(call_id)
        report[call_id] = validator.errors(synthesize_response(ctx), ctx)
    return report


def make_validator(builder):
    return PlanResponseValidator(builder.response_schema, builder.artifact_schema, builder.inputs)
