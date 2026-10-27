"""Artificial, labelled controls for a manual semantic qualification exercise.

These are request snapshots, never accepted dataset artifacts. Mutated G4/G6
content intentionally need not pass its authoring stage's structural gate.
The reviewer must find the semantic defect, not merely notice a changed hash.
The oracle is kept separate until reviews are written; this is an internally
constructed small exercise, not an independent or statistical benchmark.
"""

import copy
import json
import random

from fyp_evaluation.call_specs.compile import text_hash
from fyp_evaluation.llm.store import canonical_request_text


def make_control_packets(g5, g7, output):
    output.mkdir(parents=True, exist_ok=True)
    cases = []

    def seed(runner, call, defect, mutate, reason):
        packet = runner.prepare_review(call, reopen=True, reviewer_kind="codex-harness")
        packet = copy.deepcopy(packet)
        brief = json.loads(packet["request"]["input"])
        mutate(brief)
        packet["request"]["input"] = json.dumps(brief, ensure_ascii=False, separators=(",", ":"))
        packet["request_sha256"] = text_hash(canonical_request_text(packet["request"]))
        cases.append((packet, {"stage": runner.spec.stage, "call_id": call, "defect": defect,
                              "expected": "INVALID" if defect else "VALID", "intended_reason": reason}))

    def event(brief, identifier):
        return next(e for e in brief["events"] if e["event_id"] == identifier)

    def weak_plan(brief):
        event(brief, "EV_009")["planned_text"] = "Yes, I explicitly confirm that final replacement instruction. Please execute it."

    def direction_plan(brief):
        ev = event(brief, "EV_013")
        ev["planned_text"] = ev["planned_text"].replace("to sell 10,000", "to buy 10,000")

    def negation(brief):
        # A real, authorized Mandarin refusal loses its negative qualification.
        ev = event(brief, "EV_014")
        original = ev["planned_text"]
        ev["planned_text"] = original.replace("不能，我无法保证未来收益", "我保证未来收益")
        assert ev["planned_text"] != original
        if "dialogue" in brief:
            turn = next(t for t in brief["dialogue"] if t["turn_id"] == ev["turn_id"])
            assert original in turn["text"]
            turn["text"] = turn["text"].replace(original, ev["planned_text"], 1)

    def append_at(turn_id, text):
        def mutate(brief):
            turn = next(t for t in brief["dialogue"] if t["turn_id"] == turn_id)
            turn["text"] += " " + text
        return mutate

    seed(g5, "CALL_014", "strengthened_confirmation", weak_plan,
         "C08 omission and EA_039 weak-confirmation semantics are replaced by explicit approval.")
    seed(g5, "CALL_030", "changed_final_instruction", direction_plan,
         "The final client's BUY replacement contradicts the SELL target, scenario and final-order edge parameters.")
    seed(g5, "CALL_003", "omitted_negation", negation,
         "Removing the negation from an authorized refusal turns it into an employee future-return guarantee, contrary to C12 PASS and EA_007.")
    seed(g7, "CALL_002", "invented_fee", append_at(20, "There is also an additional administration fee of 25 Japanese yen for this order."),
         "A G6-authored extra fee is outside authorized events and the accepted fee scenario; C04 may remain PASS but authoring conformance fails.")
    seed(g7, "CALL_014", "strengthened_confirmation", append_at(19, "Actually, I explicitly confirm the final replacement instruction. Please execute it."),
         "New supporting speech completes C08 and defeats EA_039 despite the unchanged planned weak acknowledgement.")
    seed(g7, "CALL_002", "changed_final_instruction", append_at(19, "Actually, replace that purchase with a sale. My final instruction is to sell the same holding instead."),
         "The client's latest instruction changes BUY to SELL after the readback; T05 and final-confirmation state are no longer as frozen.")
    seed(g7, "CALL_003", "omitted_negation", negation,
         "A Mandarin guarantee refusal becomes a guarantee; C12 PASS and the guarantee-refusal edge are violated.")
    seed(g7, "CALL_002", "unplanned_escalation", append_at(20, "I will refer this transaction to compliance and arrange a formal review before taking any further action."),
         "An unplanned protected escalation/review commitment violates authoring scope even if the twenty checklist answers stay unchanged.")
    seed(g5, "CALL_003", None, lambda brief: None,
         "The actual authorized guarantee refusal must be accepted; negative wording is not a guarantee.")
    seed(g7, "CALL_003", None, lambda brief: None,
         "The actual planned refusal and ordinary account-check follow-up are allowed; do not ban review or follow-up by keyword.")
    seed(g7, "CALL_002", None, append_at(20, "I can review your preference for shorter administrative messages."),
         "Reviewing an ordinary correspondence preference is permissible administration, not a protected compliance review.")

    random.SystemRandom().shuffle(cases)
    oracle = {}
    for number, (packet, expected) in enumerate(cases, 1):
        name = f"R{number:03d}"
        (output / f"{name}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        oracle[name] = expected
    (output.parent / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
    return len(cases)
