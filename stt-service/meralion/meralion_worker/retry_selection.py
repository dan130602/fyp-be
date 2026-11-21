"""Retry selection for one invocation: the selection block of stt-experiments/meralion-runner/handoff/verify_inmemory_replay.py:40-57, with the same logic.

    plan_call + group_triggers over every recording's first merge -> groups sorted by (priority, call id, window start)
    run budget      floor(run_cap x total windows of the recordings of THIS invocation), shared by every kind; groups beyond it are skipped ("run budget").
                    run_cap None (strategy v2): no invocation-wide budget at all: run_budget is None, nothing is skipped for it and the selection is never "bound"
    unresolved cap  max(1, floor(unresolved_call_cap x the call's windows)) unresolved groups per call; further ones are skipped ("unresolved per-call budget")
    low-margin      no per-call cap

The run budget is a property of the set of recordings planned together (the invocation), not of one recording.
"""
import math
from dataclasses import dataclass, field

from meralion_worker import vendored

vendored.ensure_on_path()
import retry_plan                                                # noqa: E402  (vendored, byte-identical to the frozen blob)

LABEL = "worker"                                                 # recorded in each trigger's "from"; any constant works as long as it is the same for every pred


@dataclass
class Selection:
    selected: list = field(default_factory=list)                 # groups: {call_id, window, priority, triggers}
    skipped: list = field(default_factory=list)                  # {call_id, window, reason}
    groups_total: int = 0
    windows_total: int = 0
    run_budget: object = 0                                       # an int, or None when the strategy has no invocation-wide budget
    run_budget_bound: bool = False


def select_retries(preds, *, kinds, unresolved_call_cap, run_cap):
    """preds: {call_id: pred} where pred has call_id, audio_seconds, windows (start_sample, end_sample, text) and seams."""
    found = []
    for pred in preds.values():
        found += retry_plan.plan_call(pred, LABEL, set(kinds))
    groups = retry_plan.group_triggers(found)
    groups.sort(key=lambda g: (g["priority"], g["call_id"], g["window"][0]))
    windows_total = sum(len(pred["windows"]) for pred in preds.values())
    run_budget = None if run_cap is None else math.floor(run_cap * windows_total)
    unresolved_per_call, result = {}, Selection(groups_total=len(groups), windows_total=windows_total, run_budget=run_budget)
    for group in groups:
        kinds_in_group = {kind for trigger in group["triggers"] for kind in trigger["kinds"]}
        cap = max(1, math.floor(unresolved_call_cap * len(preds[group["call_id"]]["windows"])))
        if run_budget is not None and len(result.selected) >= run_budget:
            result.skipped.append({"call_id": group["call_id"], "window": list(group["window"]), "reason": "run budget"})
            result.run_budget_bound = True
        elif "unresolved" in kinds_in_group and unresolved_per_call.get(group["call_id"], 0) >= cap:
            result.skipped.append({"call_id": group["call_id"], "window": list(group["window"]), "reason": "unresolved per-call budget"})
        else:
            if "unresolved" in kinds_in_group:
                unresolved_per_call[group["call_id"]] = unresolved_per_call.get(group["call_id"], 0) + 1
            result.selected.append(group)
    return result
