"""Constrained multi-label allocation: exact language quotas and development locks first, then a deterministic swap search.

Hard constraints (never traded for balance): exact held-out count per language, every locked call DEVELOPMENT, every unit (a single
call or a genuine linked group) assigned whole. Balance is a weighted sum of squared standardised deviations of held-out label counts
from a 30% target, so rare labels (small standard deviation) dominate automatically. The search is a greedy construction followed by
best-improvement swaps; it is deterministic (sorted IDs, ties broken by sha256(seed:call_id)) and makes no optimality claim.
"""

import hashlib
import math

from . import DEVELOPMENT, HELDOUT, SplitError
from .features import GROUP_WEIGHT, labels_of

HELDOUT_FRACTION = 0.3
LONG_TARGET_PER_LANGUAGE = 2  # strong balancing target (8 held-out LONG calls), never a hard constraint
EPSILON = 1e-9
MAX_SWAPS = 5000


def _tie(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def _units(features, groups):
    """Units are single calls or genuine groups; a group must lie inside one language so the language quotas stay exact."""
    grouped = {}
    for group in groups:
        languages = {features[c]["language"] for c in group}
        if len(languages) != 1 or any(c not in features or c in grouped for c in group):
            raise SplitError(f"invalid group {sorted(group)}: unknown or repeated call, or more than one language")
        for call in group:
            grouped[call] = tuple(sorted(group))
    units, seen = [], set()
    for call in sorted(features):
        unit = grouped.get(call, (call,))
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    return units


def check_feasible(features, locked, quotas, units):
    """Raise SplitError when the hard constraints cannot all hold."""
    unknown = sorted(set(locked) - set(features))
    if unknown:
        raise SplitError(f"locked calls not in the release: {unknown}")
    for language, held in quotas.items():
        calls = [c for c in features if features[c]["language"] == language]
        locked_here = [c for c in calls if c in locked]
        if len(calls) - len(locked_here) < held:
            raise SplitError(f"{language}: only {len(calls) - len(locked_here)} unlocked calls remain for {held} held-out slots")
    if sum(quotas.values()) > len(features):
        raise SplitError("held-out quota exceeds the number of calls")
    for unit in units:
        if len(unit) > 1 and any(c in locked for c in unit) and not all(c in locked for c in unit):
            raise SplitError(f"group {unit} is partly locked; a group stays together, so lock all members")


class _Model:
    def __init__(self, features, units, locked, long_target):
        self.features, self.units, self.locked = features, units, locked
        self.unit_labels = []
        totals = {}
        group_of = {}
        for unit in units:
            counts = {}
            for call in unit:
                for label, group in labels_of(features[call]).items():
                    counts[label] = counts.get(label, 0) + 1
                    group_of[label] = group
                    totals[label] = totals.get(label, 0) + 1
            self.unit_labels.append(counts)
        self.total = totals
        self.weight = {label: GROUP_WEIGHT[group_of[label]] for label in totals}
        self.target = {label: HELDOUT_FRACTION * n for label, n in totals.items()}
        if long_target:
            for label in totals:
                if label.startswith("lang_dur:") and label.endswith("|LONG"):
                    self.target[label] = min(float(long_target), float(totals[label]))
            if "dur:LONG" in totals:
                self.target["dur:LONG"] = sum(self.target[l] for l in totals if l.startswith("lang_dur:") and l.endswith("|LONG"))
        self.scale = {label: max(0.5, math.sqrt(n * HELDOUT_FRACTION * (1 - HELDOUT_FRACTION))) for label, n in totals.items()}

    def cost(self, label, count):
        return self.weight[label] * ((count - self.target[label]) / self.scale[label]) ** 2

    def delta(self, counts, add, remove):
        """Objective change when unit `add` joins and unit `remove` leaves the held-out set (either may be None)."""
        change = {}
        if add is not None:
            for label, n in self.unit_labels[add].items():
                change[label] = change.get(label, 0) + n
        if remove is not None:
            for label, n in self.unit_labels[remove].items():
                change[label] = change.get(label, 0) - n
        return sum(self.cost(label, counts.get(label, 0) + d) - self.cost(label, counts.get(label, 0)) for label, d in change.items() if d)

    def objective(self, counts):
        return sum(self.cost(label, counts.get(label, 0)) for label in self.total)


def allocate(features, locked, quotas, *, seed, groups=(), long_target=LONG_TARGET_PER_LANGUAGE):
    """{call_id: DEVELOPMENT|HELDOUT} plus search metadata. Deterministic for any input order."""
    locked = set(locked)
    units = _units(features, groups)
    check_feasible(features, locked, quotas, units)
    model = _Model(features, units, locked, long_target)
    language = [features[u[0]]["language"] for u in units]
    tie = [_tie(seed, u[0]) for u in units]
    free = [i for i, u in enumerate(units) if not any(c in locked for c in u)]
    remaining = {lang: quotas.get(lang, 0) for lang in set(language)}
    held, counts = set(), {}

    def apply(i, sign):
        for label, n in model.unit_labels[i].items():
            counts[label] = counts.get(label, 0) + sign * n

    # 1. construction: repeatedly add the held-out unit that lowers the objective most (rare labels dominate through the scale)
    while any(remaining.values()):
        best = None
        for i in free:
            size = len(units[i])
            if i in held or remaining[language[i]] < size:
                continue
            key = (model.delta(counts, i, None), tie[i])
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:
            raise SplitError("could not fill the held-out quotas with whole units")
        i = best[1]
        held.add(i)
        apply(i, 1)
        remaining[language[i]] -= len(units[i])
    # 2. improvement: best-improvement swaps of equal-size units inside one language
    swaps = 0
    while swaps < MAX_SWAPS:
        best = None
        for h in sorted(held):
            for d in free:
                if d in held or language[d] != language[h] or len(units[d]) != len(units[h]):
                    continue
                key = (model.delta(counts, d, h), tie[h], tie[d])
                if best is None or key < best[0]:
                    best = (key, h, d)
        if best is None or best[0][0] >= -EPSILON:
            break
        _, h, d = best
        held.remove(h)
        held.add(d)
        apply(h, -1)
        apply(d, 1)
        swaps += 1
    assignment = {c: DEVELOPMENT for c in features}
    for i in held:
        for call in units[i]:
            assignment[call] = HELDOUT
    return assignment, {"swaps": swaps, "objective": round(model.objective(counts), 6), "units": len(units),
                        "targets": {k: round(v, 4) for k, v in model.target.items()}, "model": model}


def validate_assignment(features, assignment, locked, quotas, groups=()):
    """Independent hard-constraint check; returns a list of errors (empty when valid)."""
    errors = []
    if sorted(assignment) != sorted(features):
        errors.append("assignments do not cover every call exactly once")
        return errors
    if any(v not in (DEVELOPMENT, HELDOUT) for v in assignment.values()):
        errors.append("an assignment label is neither DEVELOPMENT nor HELDOUT")
    errors += [f"{c}: locked call is not DEVELOPMENT" for c in sorted(locked) if assignment.get(c) != DEVELOPMENT]
    for language, held in quotas.items():
        actual = sum(1 for c, f in features.items() if f["language"] == language and assignment[c] == HELDOUT)
        total = sum(1 for f in features.values() if f["language"] == language)
        if actual != held:
            errors.append(f"{language}: {actual} held-out calls, expected {held}")
        if total - actual != total - held:
            errors.append(f"{language}: development count differs from the quota")
    for group in groups:
        if len({assignment.get(c) for c in group}) != 1:
            errors.append(f"group {sorted(group)} is split across DEVELOPMENT and HELDOUT")
    return errors
