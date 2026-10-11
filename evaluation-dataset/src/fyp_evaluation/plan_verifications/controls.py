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


def build_control(source, destination, name, *, schema_dir=DEFAULT_SCHEMA_DIR, audit_root=None):
    """Create control `name` at `destination`; returns the destination path. Raises if G4 does not accept the mutation."""
    from fyp_evaluation.check_plans.inputs import load_inputs
    from fyp_evaluation.check_plans.run import CheckPlanRunner
    from fyp_evaluation.llm.fake_transport import FakeTransport, response_body

    if name not in MUTATIONS:
        raise ValueError(f"unknown control {name!r}; choose one of {sorted(MUTATIONS)}")
    destination = copy_release(source, destination)
    mutation = MUTATIONS[name]
    if mutation is None:
        return destination
    event_id, change = mutation
    g4 = CheckPlanRunner(destination, schema_dir=schema_dir, audit_root=audit_root or destination.parent / f"{destination.name}_g4_audit",
                         inputs=load_inputs(destination, schema_dir))
    prov, problem = g4._prov_state(CONTROL_CALL)
    if problem or not prov["accepted"]:
        raise ValueError(f"{CONTROL_CALL}: no usable accepted G4 response to mutate ({problem})")
    response = json.loads(json.dumps(prov["accepted"]["parsed_response"]))
    entry = next(e for e in response["events"] if e["event_id"] == event_id)
    mutated = change(entry["text"])
    if mutated == entry["text"]:
        raise ValueError(f"{name}: the mutation did not change {event_id}")
    entry["text"] = mutated
    g4._transport = FakeTransport(lambda body: response_body(json.dumps(response), response_id=f"resp_control_{name}"))
    attempt = g4.run_sync(CONTROL_CALL, reopen_reason=f"G5 semantic control {name} (isolated copy)")
    if attempt["state"] != "ACCEPTED":
        raise ValueError(f"{name}: G4 did not accept the mutated plan: {attempt['rejection_reason']} {attempt['validation_errors'][:3]}")
    return destination
