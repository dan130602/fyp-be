"""Answer support: which planned events establish the final answer of an extraction question, and every value a question is given.

Extracted unchanged from the G9 compiler so that G9 and the G6/G7 multilingual checks apply the identical rule. Deterministic:
nothing is guessed, and anything the rule cannot verify raises instead of falling back to the whole planned set.
"""


def value_history(question_id, events, spans):
    """Every event that states a value for the question (readback errors included), in dialogue order."""
    history = []
    for event in events:
        if "values" in event and question_id in event["values"]:
            value = event["values"][question_id]["value"]
        elif (event.get("readback") or {}).get("wrong_question_id") == question_id:
            value = event["readback"]["wrong_value"]
        else:
            continue
        span = spans[event["event_id"]]
        history.append({"turn_id": span["turn_id"], "event_id": event["event_id"], "speaker": span["speaker"], "kind": event["kind"],
                        "status": event["status"], "value": value, "_order": (span["turn_id"], span["start"])})
    history.sort(key=lambda entry: entry["_order"])
    return [{key: value for key, value in entry.items() if key != "_order"} for entry in history]


# Final-event kinds whose own turn explicitly states the value (a reference needs its antecedent as well).
EXPLICIT_FINAL_KINDS = frozenset({"VALUE_STATEMENT", "VALUE_CORRECTION", "ORDER_STATEMENT"})


def mentions_value(event, question_id, value):
    return any(m["question_id"] == question_id and m["value"] == value for m in event["authoring"].get("value_mentions", []))


def required_extraction_events(call_id, question_id, check, events_by_id, expected_value):
    """The planned events a model must cite to answer the extraction question: the final event and, for a reference, its antecedent.

    Deterministic: nothing is guessed. Anything the rule cannot verify raises instead of falling back to the whole planned set.
    """
    where = f"{call_id}/{question_id}"
    final_id = check.get("final_event_id")
    if not final_id:
        raise ValueError(f"{where}: an extraction check needs a final event to derive its required evidence")
    final = events_by_id[final_id]
    if final["status"] != "FINAL" or final["values"][question_id]["value"] != expected_value:
        raise ValueError(f"{where}: final event {final_id} is not the final answer {expected_value!r}")
    if final["kind"] in EXPLICIT_FINAL_KINDS:
        if not mentions_value(final, question_id, expected_value):
            raise ValueError(f"{where}: final event {final_id} does not explicitly mention the expected value")
        required = [final_id]
    elif final["kind"] == "REFERENCE_SELECTION":
        antecedent = events_by_id[final["refers_to_event_id"]]
        if antecedent["kind"] != "ANTECEDENT_MENTION" or not mentions_value(antecedent, question_id, expected_value):
            raise ValueError(f"{where}: reference {final_id} does not point to an explicit mention of the expected value")
        required = [antecedent["event_id"], final_id]
    else:
        raise ValueError(f"{where}: unsupported final event kind {final['kind']}")
    if not set(required) <= set(check["evidence_event_ids"]):
        raise ValueError(f"{where}: required evidence {required} is not part of the planned evidence")
    return required
