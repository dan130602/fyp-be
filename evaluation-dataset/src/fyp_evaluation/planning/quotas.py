"""Resolve the integer quotas of a resized multilingual plan from its approved reference design.

The design file (`authoring/quota_design.json`) holds the approved reference quotas (the 500-call Option D plan) and the
new top-level counts. This module derives every per-profile integer quota from them by exact rational scaling and joint
rounding, so profile totals and global targets stay consistent; nothing is rounded independently and then assumed to
add up. The result is stored in `authoring/generation_plan.json` (`scripts/resolve_multilingual_quotas.py --write`) and
G1 reads only that file. All choices are deterministic: ties are broken by a hash of the plan seed and the item names.

Rounding primitives:

* `largest_remainder`: integers with a fixed total, closest to the shares.
* `round_matrix`: controlled rounding of a profile x item table to floor/ceil of each cell with exact row and column
  totals (a max-flow, so it exists whenever the totals are consistent).
* `resolve_histogram`: per-profile calls-by-count histograms with an exact call total and an exact total of items, and
  exact global bins, closest to the scaled reference.
"""

import hashlib
import json
import re
from fractions import Fraction
from itertools import product
from math import ceil, floor

from .multilingual import max_flow

HELPER_VERSION = "1.0.0"
CROSS_PREFIX = "CROSS_LANGUAGE_"


class QuotaError(ValueError):
    """The requested resize has no exact integer solution under the approved rules."""


def _tie(seed, *key):
    return int.from_bytes(hashlib.sha256(f"{seed}|quota|{'|'.join(map(str, key))}".encode("utf-8")).digest()[:8], "big")


def _frac(value):
    return value - floor(value)


def largest_remainder(shares, total, seed, label):
    """{key: int} closest to `shares` (Fractions) with sum `total`; ties by hash."""
    base = {key: floor(value) for key, value in shares.items()}
    missing = total - sum(base.values())
    if not 0 <= missing <= len(shares):
        raise QuotaError(f"{label}: shares sum to {float(sum(shares.values()))}, which cannot round to {total}")
    order = sorted(shares, key=lambda key: (-_frac(shares[key]), _tie(seed, label, key)))
    for key in order[:missing]:
        base[key] += 1
    return base


def round_matrix(shares, rows, cols, row_totals, col_totals, seed, label, minimum_positive=0):
    """Controlled rounding of {(row, col): Fraction} to integers within [floor, ceil] of every cell.

    Row and column sums equal the given totals exactly. Cells with a positive share are raised to `minimum_positive`.
    """
    lo, hi = {}, {}
    for key in ((r, c) for r in rows for c in cols):
        value = shares.get(key, Fraction(0))
        lo[key] = floor(value)
        hi[key] = ceil(value)
        if value > 0 and minimum_positive:
            lo[key] = max(lo[key], minimum_positive)
            hi[key] = max(hi[key], lo[key])
    need_row = {r: row_totals[r] - sum(lo[r, c] for c in cols) for r in rows}
    need_col = {c: col_totals[c] - sum(lo[r, c] for r in rows) for c in cols}
    if sum(row_totals.values()) != sum(col_totals.values()):
        raise QuotaError(f"{label}: row totals {sum(row_totals.values())} differ from column totals {sum(col_totals.values())}")
    if any(v < 0 for v in need_row.values()) or any(v < 0 for v in need_col.values()):
        raise QuotaError(f"{label}: totals are below the minimum cell values")
    graph = {"S": {}, "T": {}}
    for r in rows:
        graph[("R", r)] = {}
        graph["S"][("R", r)] = need_row[r]
    for c in cols:
        graph[("C", c)] = {"T": need_col[c]}
    for r in rows:
        cells = [c for c in cols if hi[r, c] > lo[r, c]]
        cells.sort(key=lambda c: (-_frac(shares.get((r, c), Fraction(0))), _tie(seed, label, r, c)))
        for c in cells:
            graph[("R", r)][("C", c)] = hi[r, c] - lo[r, c]
    if max_flow(graph, "S", "T") != sum(need_row.values()):
        raise QuotaError(f"{label}: no integer table fits the row and column totals")
    result = {}
    for r in rows:
        for c in cols:
            extra = 0
            if hi[r, c] > lo[r, c]:
                extra = (hi[r, c] - lo[r, c]) - graph[("R", r)][("C", c)]
            result[r, c] = lo[r, c] + extra
    return result


def round_table(shares, rows, cols, row_totals, seed, label, minimum_positive=0):
    """Round a table whose column totals are the largest-remainder rounding of its column share sums."""
    col_shares = {c: sum((shares.get((r, c), Fraction(0)) for r in rows), Fraction(0)) for c in cols}
    col_totals = largest_remainder(col_shares, sum(row_totals.values()), seed, f"{label}/columns")
    return round_matrix(shares, rows, cols, row_totals, col_totals, seed, label, minimum_positive)


def resolve_histogram(profiles, calls, ref_hist, scale, total_items, global_bins, seed, label):
    """{profile: {bin: calls}}: per-profile call and item totals exact, global bins exact, nearest to the scaled reference."""
    bins = sorted({int(k) for p in profiles for k in ref_hist[p]})
    shares = {p: {k: Fraction(ref_hist[p].get(str(k), 0)) * scale[p] for k in bins} for p in profiles}
    candidates = {}
    for p in profiles:
        ranges = [range(max(0, floor(shares[p][k]) - 1), ceil(shares[p][k]) + 2) for k in bins]
        found = []
        for combo in product(*ranges):
            if sum(combo) == calls[p] and sum(k * n for k, n in zip(bins, combo)) == total_items[p]:
                found.append((sum(abs(n - shares[p][k]) for k, n in zip(bins, combo)), _tie(seed, label, p, combo), combo))
        if not found:
            raise QuotaError(f"{label}: {p} has no histogram with {calls[p]} calls and {total_items[p]} items")
        candidates[p] = sorted(found)
    best = None
    for pick in product(*(candidates[p] for p in profiles)):
        if all(sum(c[2][i] for c in pick) == global_bins[k] for i, k in enumerate(bins)):
            score = (sum(c[0] for c in pick), tuple(c[1] for c in pick))
            if best is None or score < best[0]:
                best = (score, pick)
    if best is None:
        raise QuotaError(f"{label}: no combination of profile histograms reaches the global bins {global_bins}")
    return {p: {str(k): n for k, n in zip(bins, c[2])} for p, c in zip(profiles, best[1])}


def _scaled(ref_plans, key, scale):
    return {p: {k: Fraction(v) * scale[p] for k, v in ref_plans[p][key].items()} for p in ref_plans}


def resolve_quotas(design, catalog, planning_rules):
    """The resolved `language_profile_plans` (list of dicts) and their total call count for one design."""
    seed = design["seed"]
    ref_list = design["reference"]["language_profile_plans"]
    ref = {p["language_profile"]: p for p in ref_list}
    profiles = list(ref)
    target = design["target"]
    calls = {p: target["profile_calls"][p] for p in profiles}
    scale = {p: Fraction(calls[p], ref[p]["num_calls"]) for p in profiles}
    cases = [c["id"] for c in catalog["edge_cases"]]
    family = {c["id"]: c["family"] for c in catalog["edge_cases"]}
    families = list(dict.fromkeys(family.values()))
    compliance = list(ref[profiles[0]]["fail_counts_per_check"])
    out = {p: {"language_profile": p, "num_calls": calls[p]} for p in profiles}

    # ---- durations and call intents --------------------------------------------------------------------
    durations = list(ref[profiles[0]]["duration_counts"])
    table = round_table({(p, d): Fraction(ref[p]["duration_counts"][d]) * scale[p] for p in profiles for d in durations},
                        profiles, durations, calls, seed, "durations")
    for p in profiles:
        out[p]["duration_counts"] = {d: table[p, d] for d in durations}
    sets = [tuple(s["call_intents"]) for s in ref[profiles[0]]["call_intent_set_counts"]]
    shares = {(p, i): Fraction(s["count"]) * scale[p] for p in profiles for i, s in enumerate(ref[p]["call_intent_set_counts"])}
    table = round_table(shares, profiles, range(len(sets)), calls, seed, "intents")
    for p in profiles:
        out[p]["call_intent_set_counts"] = [{"call_intents": list(sets[i]), "count": table[p, i]}
                                            for i in range(len(sets)) if table[p, i]]

    # ---- FAIL and edge-case histograms (calls by count), totals tied to the per-profile item totals -------------
    fail_total = {p: sum(Fraction(v) * scale[p] for v in ref[p]["fail_counts_per_check"].values()) for p in profiles}
    edge_total = {p: sum(Fraction(v) * scale[p] for v in ref[p]["edge_case_counts"].values()) for p in profiles}
    for what, totals in (("fail", fail_total), ("edge-case", edge_total)):
        if any(t.denominator != 1 for t in totals.values()):
            raise QuotaError(f"the scaled {what} totals {[float(t) for t in totals.values()]} are not integers")
    for what, key, totals in (("fail", "fail_count_per_call_counts", fail_total), ("edge", "edge_cases_per_call_counts", edge_total)):
        if what == "edge" and all(ref[p][key] == ref[p]["fail_count_per_call_counts"] and totals[p] == fail_total[p] for p in profiles):
            for p in profiles:  # Option D plans one histogram for both, so a call's FAIL and edge-case counts pair up
                out[p][key] = dict(out[p]["fail_count_per_call_counts"])
            continue
        global_bins = {}
        for p in profiles:
            for k, v in ref[p][key].items():
                global_bins[int(k)] = global_bins.get(int(k), 0) + Fraction(v) * scale[p]
        if any(v.denominator != 1 for v in global_bins.values()):
            raise QuotaError(f"the scaled global {what} bins are not integers")
        histogram = resolve_histogram(profiles, calls, {p: ref[p][key] for p in profiles}, scale,
                                      {p: int(totals[p]) for p in profiles}, {k: int(v) for k, v in global_bins.items()},
                                      seed, f"{what}-histogram")
        for p in profiles:
            out[p][key] = histogram[p]

    # ---- composition (the all-PASS / edge-free overlap) ----------------------------------------------------------
    pure_share = {p: Fraction(ref[p]["call_composition_counts"]["all_pass_without_edge_cases"]) * scale[p] for p in profiles}
    pure = largest_remainder(pure_share, round(sum(pure_share.values())), seed, "pure-controls")
    for p in profiles:
        all_pass, edge_free = out[p]["fail_count_per_call_counts"].get("0", 0), out[p]["edge_cases_per_call_counts"].get("0", 0)
        if pure[p] > min(all_pass, edge_free):
            raise QuotaError(f"{p}: {pure[p]} pure controls exceed its {all_pass} all-PASS / {edge_free} edge-free calls")
        out[p]["call_composition_counts"] = {
            "all_pass_without_edge_cases": pure[p], "all_pass_with_edge_cases": all_pass - pure[p],
            "some_fail_without_edge_cases": edge_free - pure[p],
            "some_fail_with_edge_cases": calls[p] - all_pass - edge_free + pure[p]}

    # ---- edge-case counts: families first, then cases inside each family ----------------------------------------------
    minimum = 1
    fam_share = {(p, f): sum((Fraction(ref[p]["edge_case_counts"][c]) for c in cases if family[c] == f), Fraction(0)) * scale[p]
                 for p in profiles for f in families}
    fam_table = round_table(fam_share, profiles, families, {p: int(edge_total[p]) for p in profiles}, seed, "families", minimum)
    edge_counts = {p: {} for p in profiles}
    for f in families:
        members = [c for c in cases if family[c] == f]
        shares = {}
        for p in profiles:
            ref_family = sum(ref[p]["edge_case_counts"][c] for c in members)
            for c in members:
                shares[p, c] = Fraction(ref[p]["edge_case_counts"][c] * fam_table[p, f], ref_family) if ref_family else Fraction(0)
        table = round_table(shares, profiles, members, {p: fam_table[p, f] for p in profiles}, seed, f"cases/{f}", minimum)
        for p in profiles:
            for c in members:
                edge_counts[p][c] = table[p, c]
    for p in profiles:
        out[p]["edge_case_counts"] = {c: edge_counts[p][c] for c in cases}

    # ---- per-question FAIL counts: the dependent question follows its parent plus the risk edge case --------------------
    rule = planning_rules["dependent_fail_rules"][0]
    dependent, parent = rule["question_id"], rule["fails_only_with"]["failed_question_id"]
    risk = rule["fails_only_with"]["or_edge_case_id"]
    parent_share = {p: Fraction(ref[p]["fail_counts_per_check"][parent]) * scale[p] for p in profiles}
    parent_count = largest_remainder(parent_share, round(sum(parent_share.values())), seed, f"fails/{parent}")
    pinned = {p: {parent: parent_count[p], dependent: parent_count[p] + edge_counts[p][risk]} for p in profiles}
    others = [q for q in compliance if q not in (parent, dependent)]
    shares = {(p, q): Fraction(ref[p]["fail_counts_per_check"][q]) * scale[p] for p in profiles for q in others}
    rows = {p: int(fail_total[p]) - sum(pinned[p].values()) for p in profiles}
    table = round_table(shares, profiles, others, rows, seed, "fails/others")
    for p in profiles:
        merged = {**pinned[p], **{q: table[p, q] for q in others}}
        out[p]["fail_counts_per_check"] = {q: merged[q] for q in compliance}

    # ---- language variants -------------------------------------------------------------------------------------------
    for p in profiles:
        resolved, cross = {}, {}
        for c, by in ref[p]["language_variant_counts"].items():
            count = edge_counts[p][c]
            if sum(by.values()) == ref[p]["edge_case_counts"][c]:  # a variant carried by every instance of the case
                resolved[c] = {v: count for v in by}
            elif all(v.startswith(CROSS_PREFIX) for v in by):  # a switch-direction pair: rounded jointly below
                cross[c] = Fraction(sum(by.values())) * scale[p]
            else:  # any other variant scales with the profile
                shares = {v: Fraction(n) * scale[p] for v, n in by.items()}
                resolved[c] = largest_remainder(shares, round(sum(shares.values())), seed, f"variants/{p}/{c}")
        if cross:
            totals = largest_remainder(cross, round(sum(cross.values())), seed, f"variants/{p}/cross-language")
            for c, total in totals.items():
                directions = sorted(ref[p]["language_variant_counts"][c])
                split = {v: total // len(directions) for v in directions}
                for offset in range(total - sum(split.values())):  # the odd extra examples alternate between directions
                    split[directions[(cases.index(c) + offset) % len(directions)]] += 1
                resolved[c] = {v: split[v] for v in ref[p]["language_variant_counts"][c]}
        for c, by in resolved.items():
            if any(n < 1 for n in by.values()):
                raise QuotaError(f"{p}: {c} would have a language variant with no examples ({by})")
            if sum(by.values()) > edge_counts[p][c]:
                raise QuotaError(f"{p}: {c} has more variants than assignments")
        out[p]["language_variant_counts"] = {c: resolved[c] for c in ref[p]["language_variant_counts"]}

    # ---- currency, direction, order type, mixing style, exposure caps ------------------------------------------------------
    for key in ("transaction_direction_counts", "order_type_counts", "transaction_currency_counts"):
        names = list(ref[profiles[0]][key])
        table = round_table({(p, n): Fraction(ref[p][key][n]) * scale[p] for p in profiles for n in names}, profiles, names, calls,
                            seed, key)
        for p in profiles:
            out[p][key] = {n: table[p, n] for n in names}
    for p in profiles:
        out[p]["target_exposure_caps"] = {q: floor(Fraction(percent * calls[p], 100))
                                          for q, percent in target["exposure_cap_percent"].items()}
        if "mixing_style_counts" in ref[p]:
            shares = {s: Fraction(n) * scale[p] for s, n in ref[p]["mixing_style_counts"].items()}
            out[p]["mixing_style_counts"] = largest_remainder(shares, calls[p], seed, f"mixing/{p}")

    order = ["language_profile", "num_calls", "duration_counts", "call_intent_set_counts", "call_composition_counts",
             "fail_count_per_call_counts", "fail_counts_per_check", "edge_cases_per_call_counts", "edge_case_counts",
             "language_variant_counts", "transaction_direction_counts", "order_type_counts", "transaction_currency_counts",
             "target_exposure_caps", "mixing_style_counts"]
    return [{k: out[p][k] for k in order if k in out[p]} for p in profiles], sum(calls.values())


def design_hash(text):
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def apply_quotas(plan, design, design_text, catalog):
    """A copy of the generation plan carrying the resolved quotas and their provenance."""
    resolved, total = resolve_quotas(design, catalog, plan["planning_rules"])
    new = dict(plan)
    new["description"] = re.sub(r"[0-9]+ calls over", f"{total} calls over", plan["description"], count=1)
    new["num_calls"] = total
    new["language_profile_plans"] = resolved
    new["quota_resolution"] = {"design_file": "authoring/quota_design.json", "design_sha256": design_hash(design_text),
                               "helper": "fyp_evaluation.planning.quotas", "helper_version": HELPER_VERSION}
    return new


def canonical_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
