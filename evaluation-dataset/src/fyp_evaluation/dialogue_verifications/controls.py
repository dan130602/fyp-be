"""Semantic control fixtures for G7 (isolated dataset copies; production files are never touched).

Each control is a full copy of a release in which ONE call's accepted G6 dialogue is replaced through G6's own acceptance path:
the stored accepted G6 response (model turns with event markers) is mutated and fed back to `DialogueRunner` (`--submit` with a
reopen reason, then `--collect`) through the offline fake Batch transport, so the structural G6 gate accepts it and writes real
provenance. Every planned event is preserved (the mutation only adds speech to a turn without markers), and strict alternation
and the length band still hold. G7 then sees a structurally valid dialogue whose meaning is deliberately wrong. No network and no
credentials are involved in building a control.

  A_unchanged               CALL_016 as accepted (presumed valid; a VALID verdict is not mandatory)
  B_fee_leak                the employee's closing turn (66) also states the sale's transaction fee as a rate, while C04 must FAIL
  C_buy_after_confirmation  the client's closing turn (67) cancels the sale and instructs a BUY after the final confirmation (T05 SELL)
"""

import json
import shutil
from pathlib import Path

from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

CONTROL_CALL = "CALL_016"
FEE_SENTENCE = "For the sale we discussed, the transaction fee is 0.25% of its transaction amount."
BUY_SENTENCE = "Before we end, cancel the sale instruction. My final instruction for that same holding is to buy instead."
# name -> (index of the model turn, the speaker that turn must have, the change applied to its text)
MUTATIONS = {
    "A_unchanged": None,
    "B_fee_leak": (66, "EMPLOYEE", lambda text: text.rstrip() + " " + FEE_SENTENCE),
    "C_buy_after_confirmation": (67, "CLIENT", lambda text: BUY_SENTENCE + " " + text.lstrip()),
}
SKIP = ("authoring/dialogue_verification",)


def copy_release(source, destination):
    """Copy file contents only (OneDrive-safe), keeping G0-G6 outputs and dropping any G7 output or lock file."""
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


def build_control(source, destination, name, *, schema_dir=DEFAULT_SCHEMA_DIR, audit_root=None):
    """Create control `name` at `destination`; returns the destination path. Raises if G6 does not accept the mutation."""
    if name not in MUTATIONS:
        raise ValueError(f"unknown control {name!r}; choose one of {sorted(MUTATIONS)}")
    destination = copy_release(source, destination)
    return apply_mutation(destination, name, schema_dir=schema_dir, audit_root=audit_root)


def apply_mutation(destination, name, *, schema_dir=DEFAULT_SCHEMA_DIR, audit_root=None):
    """Replace CONTROL_CALL's accepted G6 dialogue in the isolated copy `destination` by the mutated one, through G6 itself."""
    from fyp_evaluation.dialogues.inputs import load_inputs
    from fyp_evaluation.dialogues.run import DialogueRunner
    from fyp_evaluation.llm.fake_transport import FakeTransport, response_body

    destination = Path(destination)
    mutation = MUTATIONS[name]
    if mutation is None:
        return destination
    index, speaker, change = mutation
    g6 = DialogueRunner(destination, schema_dir=schema_dir, audit_root=audit_root or destination.parent / f"{destination.name}_g6_audit",
                        inputs=load_inputs(destination, schema_dir))
    prov, problem = g6._prov_state(CONTROL_CALL)
    if problem or not prov["accepted"]:
        raise ValueError(f"{CONTROL_CALL}: no usable accepted G6 response to mutate ({problem})")
    response = json.loads(json.dumps(prov["accepted"]["parsed_response"]))
    turn = response["turns"][index]
    if turn["speaker"] != speaker or "[[" in turn["text"]:
        raise ValueError(f"{name}: model turn {index} is not a marker-free {speaker} turn")
    mutated = change(turn["text"])
    if mutated == turn["text"]:
        raise ValueError(f"{name}: the mutation did not change turn {index}")
    turn["text"] = mutated
    g6._transport = FakeTransport(lambda body: response_body(json.dumps(response), response_id=f"resp_control_{name}"))
    submission = g6.submit([CONTROL_CALL], reopen_reason=f"G7 semantic control {name} (isolated copy)")
    g6.transport().finish_batch(submission["batch_id"])
    g6.collect()
    attempt = g6.call_state(CONTROL_CALL)["latest"]
    if attempt["state"] != "ACCEPTED":
        raise ValueError(f"{name}: G6 did not accept the mutated dialogue: {attempt['rejection_reason']} {attempt['validation_errors'][:3]}")
    return destination
