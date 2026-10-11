"""Offline feasibility: a constraint preflight, and a deterministic synthetic dialogue that satisfies every rule.

`synthesize_dialogue` is a test fixture and a feasibility proof, not a scheduler for the model: each SAME_TURN group gets
its own turn in a stable topological order, neutral filler turns keep strict alternation and minimum gaps, and further
filler pairs pad the dialogue into the length band. The model chooses its own placements; Python only validates them.
"""

import heapq

from .render import render, word_count

FILLER = {
    "EMPLOYEE": "Thank you, I have noted that on my side.",
    "CLIENT": "Alright, thank you for walking me through the call today.",
}
OTHER = {"EMPLOYEE": "CLIENT", "CLIENT": "EMPLOYEE"}


def _groups(ctx):
    """{event_id: representative} for SAME_TURN groups (union-find), and {representative: [event_ids in G4 order]}."""
    parent = {event_id: event_id for event_id in ctx.texts}

    def find(event_id):
        while parent[event_id] != event_id:
            parent[event_id] = parent[parent[event_id]]
            event_id = parent[event_id]
        return event_id

    for c in ctx.constraints:
        if c["type"] == "SAME_TURN":
            first, second = find(c["events"][0]), find(c["events"][1])
            if first != second:
                parent[max(first, second, key=ctx.event_index.get)] = min(first, second, key=ctx.event_index.get)
    members = {}
    for event_id in sorted(ctx.texts, key=ctx.event_index.get):
        members.setdefault(find(event_id), []).append(event_id)
    return {event_id: find(event_id) for event_id in ctx.texts}, members


def _topological(nodes, edges, rank):
    """Kahn's algorithm with a stable tie-break by `rank`; returns None when the edges contain a cycle."""
    incoming = {node: 0 for node in nodes}
    for _, after in edges:
        incoming[after] += 1
    heap = [(rank(node), node) for node in nodes if incoming[node] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, node = heapq.heappop(heap)
        order.append(node)
        for before, after in edges:
            if before == node:
                incoming[after] -= 1
                if incoming[after] == 0:
                    heapq.heappush(heap, (rank(after), after))
    return order if len(order) == len(nodes) else None


def _plan(ctx):
    """(errors, group order, members, in-group event order). Errors make the call infeasible before any request."""
    group_of, members = _groups(ctx)
    errors = []
    for root, events in members.items():
        speakers = {ctx.speakers[event_id] for event_id in events}
        if len(speakers) > 1:
            errors.append(f"SAME_TURN group {events} mixes speakers {sorted(speakers)}")
    for c in ctx.constraints:
        if c["type"] in ("NOT_SAME_TURN", "MIN_INTERVENING_TURNS") and group_of[c["events"][0]] == group_of[c["events"][1]]:
            errors.append(f"{c['type']} {c['events']} conflicts with SAME_TURN grouping")
    orders = [(c["before"], c["after"]) for c in ctx.constraints if c["type"] == "ORDER"]
    if _topological(list(ctx.texts), orders, ctx.event_index.get) is None:
        errors.append("ORDER constraints contain a cycle")
    group_edges = sorted({(group_of[b], group_of[a]) for b, a in orders if group_of[b] != group_of[a]})
    order = _topological(list(members), group_edges, ctx.event_index.get)
    if order is None:
        errors.append("ORDER constraints contain a cycle across SAME_TURN groups")
    inner = {}
    for root, events in members.items():
        local = [(b, a) for b, a in orders if b in events and a in events]
        inner[root] = _topological(events, local, ctx.event_index.get) or events
    return errors, order or [], members, inner


def preflight_errors(ctx):
    return _plan(ctx)[0]


def synthesize_dialogue(ctx):
    """A response (`{"turns": [...]}`) that satisfies every structural rule for this call, or raises ValueError."""
    errors, order, members, inner = _plan(ctx)
    if errors:
        raise ValueError("; ".join(errors))
    gaps = {}
    for c in ctx.constraints:
        if c["type"] == "MIN_INTERVENING_TURNS":
            first, second = c["events"]
            gaps.setdefault(first, []).append((second, c["turns"]))
            gaps.setdefault(second, []).append((first, c["turns"]))
    turns, placed = [], {}
    for root in order:
        events = inner[root]
        speaker = ctx.speakers[events[0]]
        required = len(turns)
        for event_id in events:
            for other, n in gaps.get(event_id, []):
                if other in placed:
                    required = max(required, placed[other] + n + 1)
        while len(turns) < required or (turns and turns[-1]["speaker"] == speaker):
            filler_speaker = OTHER[turns[-1]["speaker"]] if turns else OTHER[speaker]
            turns.append({"speaker": filler_speaker, "text": FILLER[filler_speaker]})
        for event_id in events:
            placed[event_id] = len(turns)
        turns.append({"speaker": speaker, "text": " ".join(f"[[{event_id}]]" for event_id in events)})
    words = word_count(render(ctx, {"turns": turns})[0])
    while words < ctx.bounds["target_words"][0]:
        speaker = OTHER[turns[-1]["speaker"]]
        turns.append({"speaker": speaker, "text": FILLER[speaker]})
        words += len(FILLER[speaker].split())
    return {"turns": turns}
