"""ARTIFICIAL Stage 4B fixtures only; never actual pilot data or authoring files."""

from pathlib import Path

from fixtures import documents
from fyp_evaluation.authoring.call_configs import build_call_config
from fyp_evaluation.authoring.check_plans import build_check_plan

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


def compilation_inputs(*, null_identifier=False):
    checklist = [{"id": qid, "type": "COMPLIANCE", "requirement_type": "OBLIGATION",
                  "question": "ARTIFICIAL marker check"} for qid in ("FIX_PRESENT", "FIX_ABSENT")]
    fields = [("K01", "CLIENT_KYC", "client_identification_number", "text"),
              ("FIX_DOB", "CLIENT_KYC", "dob", "dob_dd_month_yyyy"),
              ("FIX_INSTRUMENT", "TRANSACTION", "fixture_instrument", "text"),
              ("FIX_CURRENCY", "TRANSACTION", "fixture_currency", "text")]
    normalization = documents()["normalization"]
    for qid, category, field, kind in fields:
        checklist.append({"id": qid, "type": category, "field": field, "question": "ARTIFICIAL extraction"})
        normalization["rules"][kind] = {"operation": "exact", "value_type": kind}
        normalization["field_rules"][field] = kind
    cells = {"FIX_PRESENT": "PASS", "FIX_ABSENT": "FAIL",
             "K01": "NOT_STATED" if null_identifier else "S0000000X",
             "FIX_DOB": "01 May 1982", "FIX_INSTRUMENT": "ARTIFICIAL Copper Basket", "FIX_CURRENCY": "EUR"}
    matrix = {"CALL_999": cells, "CALL_998": {**cells, "FIX_CURRENCY": "GBP"}}
    config = build_call_config("CALL_999", checklist, normalization, matrix)
    config["metadata"] = {"language": "en", "scenario": "ARTIFICIAL COMPILATION FIXTURE ONLY"}
    config["scenario_facts"] = [{"name": "fixture_context", "value": "ARTIFICIAL context"}]
    for answer in config["intended_answers"]:
        answer["scenario_intent"] = "ARTIFICIAL authored intent"
    plan = build_check_plan("CALL_999", checklist, normalization, matrix)
    texts = {"FIX_PRESENT": "ARTIFICIAL affirmative marker.", "K01": "My artificial identifier is S0000000X.",
             "FIX_DOB": "My artificial date is 1 May 1982.", "FIX_INSTRUMENT": "The ARTIFICIAL Copper Basket.",
             "FIX_CURRENCY": "The artificial code is EUR."}
    turns = []
    for check in plan["checks"]:
        evidence = check["evidence_plan"]
        if evidence["mode"] == "absence":
            evidence["absent_event"] = "ARTIFICIAL absent marker"
        else:
            item = evidence["planned_evidence"][0]
            item["text"] = texts[check["question_id"]]
            turns.append({"speaker": item["required_speaker"], "text": item["text"],
                          "critical_tokens": [] if check["question_id"] == "FIX_PRESENT" else [item["text"]]})
    # Speakers must alternate: separate same-speaker neighbours with an artificial connector.
    turns = [turn for index, evidence_turn in enumerate(turns) for turn in (
        [{"speaker": "CLIENT" if evidence_turn["speaker"] == "EMPLOYEE" else "EMPLOYEE",
          "text": "ARTIFICIAL connector.", "critical_tokens": []}]
        if index and turns[index - 1]["speaker"] == evidence_turn["speaker"] else []
    ) + [evidence_turn]]
    dialogue = {"schema_version": "0.1.0", "call_id": "CALL_999", "turns": turns}
    return config, plan, dialogue, checklist, normalization, matrix
