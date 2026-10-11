"""Deterministic analysis of one G4 authoring response: every structural rule, placeholder rendering and value spans.

`analyze` returns (authoring blocks by event_id, derived constraints, errors). It never judges whether wording
establishes a compliance behaviour (that is G5's semantic verification); it only enforces mechanical rules.
"""

from collections import Counter

from . import render as R
from .config import ALL_PLACEHOLDERS, QUESTION_PLACEHOLDERS
from .context import forms_of
from .template import (
    FALSE_START_END, GUARANTEE_WORD, INSIDER_WORDS, LAUNDERING_WORDS, WEAK_ACK_BANNED, parse_template, shared_word_run,
)

ROLE_TO_SOURCE = {"EVENT_VALUE": "EVENT_VALUE", "GOLD_RESTATEMENT": "GOLD_RESTATEMENT", "READBACK_WRONG": "READBACK_WRONG",
                  "READBACK_FINAL": "GOLD_RESTATEMENT"}


# ---- surfaces -----------------------------------------------------------------------------------------
def question_surface(ctx, question, value, form, spoken_form):
    detail = spoken_form if spoken_form["requirement"] == "REQUIRED" else None
    if question == "K01":
        return R.render_nric(value, form, detail)
    if question == "K02":
        return R.render_dob(value, form)
    if question == "T01":
        return R.render_product(value, form)
    if question == "T02":
        return R.render_currency(value, form, ctx.aliases)
    if question in ("T03", "T04"):
        return R.render_amount(value, form, detail)
    if question == "T05":
        return R.render_direction(value, form)
    return R.render_order_type(value, form)


def _decode(ctx, name, surface, kind=None):
    if name == "K01":
        return R.decode_nric(surface)
    if name == "K02":
        return R.decode_dob(surface)
    if name == "T01":
        return surface
    if name in ("T02", "CURRENCY"):
        return R.decode_currency(surface, ctx.aliases)
    if name in ("T03", "T04"):
        return R.decode_number(surface)
    if name == "T05":
        return {"buy": "BUY", "sell": "SELL"}[surface]
    if name == "T06":
        return {"a market order": "MARKET", "a limit order": "LIMIT"}[surface]
    if name == "FEE":
        return R.decode_number(surface.rstrip("%")) if kind == "RATE" else R.decode_number(surface.split(" ")[0])
    return R.decode_number(surface.split(" ")[0])  # LIMIT_PRICE


def _same(left, right):
    return left == right or (isinstance(left, (int, float)) and isinstance(right, (int, float)) and float(left) == float(right))


def scenario_surface(ctx, event, name):
    """(surface, value, form, source) for FEE / LIMIT_PRICE / CURRENCY."""
    if name == "FEE":
        return R.render_fee(ctx.fee, ctx.aliases), ctx.fee, ctx.fee["basis"], "SCENARIO_FEE"
    if name == "CURRENCY":
        code = event["currency"]
        return R.render_currency(code, "NATURAL_NAME", ctx.aliases), code, "NATURAL_NAME", "DISTRACTOR_CURRENCY"
    slot = ctx.limit_slot(event) or "final"
    price = ctx.limit_prices[slot]
    return R.render_limit_price(price, ctx.limit_currency[slot], ctx.aliases), price, "NUMBER_WITH_CURRENCY", "SCENARIO_LIMIT_PRICE"


def _render(ctx, event, text, placeholders, chosen_forms):
    """(planned_text, mentions, errors) with exact character offsets of every rendered surface."""
    errors, mentions, out, cursor, written = [], [], [], 0, 0
    for name, start, end in placeholders:
        out.append(text[cursor:start])
        written += start - cursor
        try:
            if name in QUESTION_PLACEHOLDERS:
                role, spoken_form, value = ctx.binding(event, name)
                form = chosen_forms.get(name) or forms_of(spoken_form)[0]
                surface = question_surface(ctx, name, value, form, spoken_form)
                source, kind = ROLE_TO_SOURCE[role], None
            else:
                surface, value, form, source = scenario_surface(ctx, event, name)
                kind = ctx.fee["basis"] if name == "FEE" else None
            decoded = _decode(ctx, name, surface, kind)
            expected = ctx.fee["value"] if name == "FEE" else value
            if not _same(decoded, expected):
                errors.append(f"{event['event_id']}: internal rendering mismatch for [[{name}]] ({surface!r} decodes to {decoded!r})")
        except (R.RenderError, KeyError, TypeError) as error:
            errors.append(f"{event['event_id']}: cannot render [[{name}]]: {error}")
            return None, [], errors
        out.append(surface)
        mentions.append({"placeholder": name, "question_id": name if name in QUESTION_PLACEHOLDERS else None, "value": value,
                         "form": form, "surface": surface, "start": written, "end": written + len(surface), "source": source})
        written += len(surface)
        cursor = end
    out.append(text[cursor:])
    planned = "".join(out)
    for mention in mentions:
        if planned[mention["start"]:mention["end"]] != mention["surface"]:
            errors.append(f"{event['event_id']}: value span for [[{mention['placeholder']}]] does not reproduce its surface")
    return planned, mentions, errors


# ---- one event ------------------------------------------------------------------------------------------
def _analyze_event(ctx, lexicon, event, entry):
    event_id, errors = event["event_id"], []

    def err(message):
        errors.append(f"{event_id}: {message}")

    allowed = event["allowed_speakers"]
    speaker = entry["speaker"]
    if len(allowed) == 1:
        if speaker is not None:
            err(f"speaker must be null because only {allowed[0]} is allowed")
        resolved = allowed[0]
    else:
        resolved = speaker if speaker in allowed else None
        if resolved is None:
            err(f"speaker must be one of {allowed}")

    placeholders, outside, parse_errors = parse_template(entry["text"])
    errors += [f"{event_id}: {e}" for e in parse_errors]
    names = [name for name, _, _ in placeholders if name in ALL_PLACEHOLDERS]
    counts = Counter(names)
    rules = ctx.slot_rules(event)

    readback_ids = entry["readback_question_ids"]
    if rules.readback:
        required, optional = (), ()
        readback = event["readback"]
        if not isinstance(readback_ids, list):
            err("readback_question_ids is required for a readback")
        else:
            if len(set(readback_ids)) != len(readback_ids):
                err("readback_question_ids contains duplicates")
            missing = [q for q in readback["must_include_question_ids"] if q not in readback_ids]
            if missing:
                err(f"readback must include {missing}")
            if len(set(readback_ids)) < readback["min_field_count"]:
                err(f"readback must include at least {readback['min_field_count']} fields")
            bad = [q for q in readback_ids if q not in ctx.readback_candidates()]
            if bad:
                err(f"readback fields {bad} have no FINAL value to read back")
            required = tuple(q for q in readback_ids if q in ctx.readback_candidates())
    else:
        required, optional = rules.required, rules.optional
        if readback_ids is not None:
            err("readback_question_ids must be null for a non-readback event")

    for name in required:
        if counts[name] != 1:
            err(f"[[{name}]] must appear exactly once (found {counts[name]})")
    for name in optional:
        if counts[name] > 1:
            err(f"[[{name}]] may appear at most once (found {counts[name]})")
    for name in counts:
        if name not in set(required) | set(optional):
            err(f"[[{name}]] is not permitted in this event")

    # spoken-form choices
    chosen, needed = {}, {}
    for name in counts:
        if name in QUESTION_PLACEHOLDERS and name in set(required) | set(optional):
            forms = ctx.form_choices(event, name)
            if forms is None:
                err(f"[[{name}]] must not be spoken explicitly here")
            elif len(forms) > 1:
                needed[name] = forms
    given = Counter(item["placeholder"] for item in entry["value_forms"])
    for name, count in given.items():
        if count > 1:
            err(f"value_forms lists {name} more than once")
        elif name not in needed:
            err(f"value_forms entry for {name} is not allowed (no form choice, or the placeholder is unused)")
    for item in entry["value_forms"]:
        if item["placeholder"] in needed:
            if item["form"] not in needed[item["placeholder"]]:
                err(f"form {item['form']!r} for {item['placeholder']} is not one of {needed[item['placeholder']]}")
            else:
                chosen[item["placeholder"]] = item["form"]
    for name in needed:
        if name not in given:
            err(f"value_forms must choose a form for {name} from {needed[name]}")
    for name, form in chosen.items():
        if form in ctx.checks[name].get("forbidden_spoken_forms", []):
            err(f"form {form} is forbidden for {name}")

    # lexical protections on the model-authored wording
    errors += [f"{event_id}: {e}" for e in lexicon.errors(outside)]
    kind = event["kind"]
    if kind == "FALSE_START" and not FALSE_START_END.search(entry["text"]):
        err("a false start must trail off (end with a dash or an ellipsis)")
    if kind == "WEAK_ACKNOWLEDGEMENT" and WEAK_ACK_BANNED.search(outside):
        err("a weak acknowledgement must not contain confirming words (yes, confirm, correct, right, agree, go ahead...)")
    for question in event["question_ids"]:
        if ctx.checks[question].get("wording_constraint") == "NOT_VERBATIM_CHECKLIST":
            reference = " ".join([ctx.checklist[question]["question"], ctx.semantics[question]["pass_condition"],
                                  ctx.semantics[question]["fail_condition"]])
            run = shared_word_run(outside, reference)
            if run:
                err(f"wording repeats the checklist verbatim ({run!r}); paraphrase it")
    if "EMPLOYEE" in allowed and "C12" not in event["question_ids"] and GUARANTEE_WORD.search(outside):
        err("employee wording must not mention guarantees outside the C12 events")
    if ctx.verdict("C09") == "PASS" and "C09" not in event["question_ids"] and LAUNDERING_WORDS.search(outside):
        err("wording suggests money laundering, but C09 is PASS (absence)")
    if ctx.verdict("C10") == "PASS" and "C10" not in event["question_ids"] and INSIDER_WORDS.search(outside):
        err("wording suggests inside information, but C10 is PASS (absence)")

    if errors:
        return None, errors
    planned, mentions, render_errors = _render(ctx, event, entry["text"], placeholders, chosen)
    if render_errors:
        return None, render_errors
    block = {"speaker": resolved, "text_template": entry["text"], "planned_text": planned, "value_mentions": mentions}
    if rules.readback:
        block["readback_question_ids"] = list(readback_ids)
    return block, []


# ---- whole response ---------------------------------------------------------------------------------------
def derive_constraints(ctx, authoring):
    """D3: each read-back field (other than the deliberately wrong one) must follow its own FINAL event."""
    derived = []
    for event in ctx.events:
        if event["kind"] != "READBACK" or event["event_id"] not in authoring:
            continue
        for question in authoring[event["event_id"]]["readback_question_ids"]:
            if question == event["readback"]["wrong_question_id"]:
                continue
            derived.append({"type": "ORDER", "before": ctx.final_event[question], "after": event["event_id"],
                            "source": {"type": "READBACK_FIELD", "id": event["event_id"], "question_id": question}})
    return derived


def find_cycle(edges):
    """A list of event IDs forming a cycle in the ORDER graph, or None."""
    graph = {}
    for before, after in edges:
        graph.setdefault(before, []).append(after)
        graph.setdefault(after, [])
    state, stack = {}, []

    def visit(node):
        state[node] = 1
        stack.append(node)
        for nxt in graph[node]:
            if state.get(nxt) == 1:
                return stack[stack.index(nxt):] + [nxt]
            if nxt not in state:
                cycle = visit(nxt)
                if cycle:
                    return cycle
        stack.pop()
        state[node] = 2
        return None

    for node in list(graph):
        if node not in state:
            cycle = visit(node)
            if cycle:
                return cycle
    return None


def analyze(ctx, lexicon, parsed):
    """(authoring by event_id, derived constraints, errors) for a schema-valid response."""
    entries = parsed["events"]
    expected = [event["event_id"] for event in ctx.events]
    counts = Counter(entry["event_id"] for entry in entries)
    errors = [f"{event_id}: listed {count} times" for event_id, count in counts.items() if count > 1]
    errors += [f"{event_id}: unknown event id (not in the frozen skeleton)" for event_id in counts if event_id not in ctx.by_id]
    errors += [f"{event_id}: missing from the response" for event_id in expected if event_id not in counts]
    authoring = {}
    for event in ctx.events:
        matching = [entry for entry in entries if entry["event_id"] == event["event_id"]]
        if len(matching) != 1:
            continue
        block, event_errors = _analyze_event(ctx, lexicon, event, matching[0])
        errors += event_errors
        if block is not None:
            authoring[event["event_id"]] = block
    derived = derive_constraints(ctx, authoring)
    cycle = find_cycle(ctx.order_edges())
    if cycle:
        errors.append(f"frozen ORDER constraints are cyclic: {' -> '.join(cycle)}")
    else:
        cycle = find_cycle(ctx.order_edges() + [(c["before"], c["after"]) for c in derived])
        if cycle:
            errors.append(f"readback field selection creates an ordering cycle: {' -> '.join(cycle)}")
    return authoring, derived, errors
