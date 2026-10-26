"""Semantic control fixtures for G5 (isolated dataset copies; production files are never touched).

Each control is a full copy of a release in which ONE call's accepted G4 plan is replaced through G4's own acceptance path:
the stored accepted G4 response is mutated and fed back to `CheckPlanRunner` (`--reopen`) through the offline fake
transport, so the structural G4 gate accepts it and writes real provenance. G5 then sees a structurally valid plan whose
meaning is deliberately wrong. No network and no credentials are involved.

  A_unchanged      the accepted plan as is (presumed valid; a VALID verdict is not mandatory)
  B_omission_leak  CALL_021 EV_001 (employee C01 act) also gives the explicit capital-loss warning, while C05 must FAIL
  C_contradiction  CALL_021 EV_015 (client FINAL T05 statement) negates the instruction ("I do not want to ...")
"""

import json
import shutil
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

CONTROL_CALL = "CALL_021"
LEAK_SENTENCE = " Please understand that you could lose some or all of the money you invest."
MUTATIONS = {
    "A_unchanged": None,
    "B_omission_leak": ("EV_001", lambda text: text + LEAK_SENTENCE),
    "C_contradiction": ("EV_015", lambda text: text.replace("I want to [[T05]]", "I do not want to [[T05]]")),
}
SKIP = ("authoring/plan_verification",)


def copy_release(source, destination):
    """Copy file contents only (OneDrive-safe), keeping G0-G4 outputs and dropping any G5 output or lock file."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if relative.startswith(SKIP) or relative.endswith(".lock"):
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination


def _mutate_through_g4(destination, call_id, event_id, change, name, *, schema_dir, audit_root, selected=None):
    """Replace one event's wording in the accepted G4 response of `call_id` and re-accept it through G4's own gate (--reopen)."""
    from fyp_evaluation.check_plans.inputs import load_inputs
    from fyp_evaluation.check_plans.run import CheckPlanRunner
    from fyp_evaluation.llm.fake_transport import FakeTransport, response_body

    g4 = CheckPlanRunner(destination, schema_dir=schema_dir, audit_root=audit_root or destination.parent / f"{destination.name}_g4_audit",
                         inputs=load_inputs(destination, schema_dir, selected), selected_call_ids=selected)
    prov, problem = g4._prov_state(call_id)
    if problem or not prov["accepted"]:
        raise ValueError(f"{call_id}: no usable accepted G4 response to mutate ({problem})")
    response = json.loads(json.dumps(prov["accepted"]["parsed_response"]))
    entry = next(e for e in response["events"] if e["event_id"] == event_id)
    mutated = change(entry["text"])
    if mutated == entry["text"]:
        raise ValueError(f"{name}: the mutation did not change {event_id}")
    entry["text"] = mutated
    g4._transport = FakeTransport(lambda body: response_body(json.dumps(response), response_id=f"resp_control_{name}"))
    attempt = g4.run_sync(call_id, reopen_reason=f"G5 semantic control {name} (isolated copy)")
    if attempt["state"] != "ACCEPTED":
        raise ValueError(f"{name}: G4 did not accept the mutated plan: {attempt['rejection_reason']} {attempt['validation_errors'][:3]}")
    return destination


def build_control(source, destination, name, *, schema_dir=DEFAULT_SCHEMA_DIR, audit_root=None):
    """Create control `name` at `destination`; returns the destination path. Raises if G4 does not accept the mutation."""
    if name not in MUTATIONS:
        raise ValueError(f"unknown control {name!r}; choose one of {sorted(MUTATIONS)}")
    destination = copy_release(source, destination)
    mutation = MUTATIONS[name]
    if mutation is None:
        return destination
    event_id, change = mutation
    return _mutate_through_g4(destination, CONTROL_CALL, event_id, change, name, schema_dir=schema_dir, audit_root=audit_root)


# ---- multilingual controls (schema 2.0 releases) ------------------------------------------------------------------------
# Each is a wording change that G4's structural gate accepts but a faithful verifier must not: the plan is valid, the meaning
# is wrong. They are located by what an event does, never by event ID, because the events of a multilingual plan differ per
# call. Truncated numbers and misleading citations are not plan mutations (Python writes every value, and G4 forbids Han value
# expressions); they are checked through grounding tests here and through G7's negative fixtures.
LEAK = {"ENGLISH": " Please understand that you could lose some or all of the money you invest.",
        "SINGLISH": " Please understand that you could lose some or all of the money you invest.",
        "MANDARIN": "请您明白，您投入的钱可能部分或全部亏掉。",
        "EN_ZH_MIXED": "请您明白，您投入的钱可能部分或全部亏掉。"}
GUARANTEE = {"ENGLISH": "I can guarantee this return, so please do not worry.", "SINGLISH": "I can guarantee this return, so please do not worry.",
             "MANDARIN": "这个回报我可以保证，请您放心。", "EN_ZH_MIXED": "这个回报我可以保证，请您放心。"}
NEGATION = {"ENGLISH": "I do not want to go ahead with this order. ", "SINGLISH": "I do not want to go ahead with this order. ",
            "MANDARIN": "我不想办理这笔订单了。", "EN_ZH_MIXED": "我不想办理这笔订单了。"}
SINGLISH_EMPLOYEE = " Can lah, no problem, we settle it for you."


def _first(events, predicate):
    return next((e["event_id"] for e in events if predicate(e)), None)


def _multilingual_controls():
    """name -> (finder(events) -> event_id | None, change(text, profile) -> text, profiles it applies to)."""
    return {
        "M_omission_leak": (lambda ev: _first(ev, lambda e: e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"]
                                              and e["authoring"]["speaker"] == "EMPLOYEE"),
                            lambda text, profile: text + LEAK[profile], ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")),
        "M_negated_direction": (lambda ev: _first(ev, lambda e: e["kind"] == "ORDER_STATEMENT" and e["status"] == "FINAL"),
                                lambda text, profile: NEGATION[profile] + text, ("MANDARIN", "EN_ZH_MIXED", "ENGLISH", "SINGLISH")),
        "M_omitted_negation": (lambda ev: _first(ev, lambda e: e["kind"] == "GUARANTEE_REFUSAL"),
                               lambda text, profile: GUARANTEE[profile], ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")),
        "S_employee_singlish": (lambda ev: _first(ev, lambda e: e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C01"]
                                                  and e["authoring"]["speaker"] == "EMPLOYEE"),
                                lambda text, profile: text + SINGLISH_EMPLOYEE, ("SINGLISH",)),
    }


MULTILINGUAL_CONTROLS = tuple(_multilingual_controls())


def build_multilingual_control(source, destination, name, call_id, profile, *, schema_dir=DEFAULT_SCHEMA_DIR, audit_root=None, selected=None):
    """Like `build_control` for a multilingual release: mutates the accepted G4 response of `call_id` (language `profile`).

    `selected` names the calls that have accepted G3 configs when G3 is not complete (selected-call mode)."""
    controls = _multilingual_controls()
    if name not in controls:
        raise ValueError(f"unknown multilingual control {name!r}; choose one of {sorted(controls)}")
    finder, change, profiles = controls[name]
    if profile not in profiles:
        raise ValueError(f"{name} does not apply to {profile} calls")
    destination = copy_release(source, destination)
    plan = json.loads((destination / "authoring/check_plans" / f"{call_id}.json").read_text(encoding="utf-8"))
    event_id = finder(plan["planned_events"])
    if event_id is None:
        raise ValueError(f"{name}: {call_id} has no event this control can change")
    return _mutate_through_g4(destination, call_id, event_id, lambda text: change(text, profile), name, schema_dir=schema_dir,
                              audit_root=audit_root, selected=selected)
