"""Read-only Markdown reports for a multilingual release, derived from its maintained contracts.

`compatibility_report` renders the edge-case x language-profile matrix straight from edge_case_catalog.json (the
only source of truth; nothing here is maintained by hand). `composition_report` renders the planned composition
from the validated planning_record.json summary. Neither writes files or changes any contract.
"""

from collections import Counter
from pathlib import Path

from fyp_evaluation.validation.schemas import load_json
from fyp_evaluation.validation.stage0 import load_stage0

from .outputs import ASSIGNMENTS, MANIFEST, RECORD, parse_assignments, parse_manifest

VARIANT_HANDLING = {
    "event_languages": "G2B pins the variant's earlier/later events (required_language)",
    "constraint": "G1 builds the contrast distractor; G1 validation checks the factor and roles",
}


def _prerequisites(case):
    prerequisite = case.get("gold_prerequisites")
    parts = []
    if prerequisite:
        parts.append(", ".join(f"{k}={v}" for k, v in prerequisite.items()))
    if case.get("verdict_requirements"):
        parts.append(" ".join(f"{q} {v}" for q, v in case["verdict_requirements"].items()))
    if case.get("target_verdict_requirement"):
        parts.append(f"target {case['target_verdict_requirement']}")
    return "; ".join(parts) or "none"


def _handling(variant):
    if variant["status"] != "READY_V1":
        return "not implemented (excluded from V1)" if variant["status"] == "EXCLUDED_V1" else "awaiting review"
    handled = [text for key, text in VARIANT_HANDLING.items() if key in variant]
    return "; ".join(handled) or "G1 counts only"


def compatibility_report(dataset_root):
    documents = load_stage0(dataset_root)
    profiles = list(documents["language_profiles"]["profiles"])
    lines = [
        "# Edge-case x language-profile compatibility",
        "",
        f"Generated from `edge_case_catalog.json` of `{documents['normalization']['dataset_id']}`. "
        "READY_V1 = deterministic and implemented in G1/G2B; NEEDS_REVIEW = waits for linguistic validation; "
        "EXCLUDED_V1 = not used in this release. Human acceptance of realised variants is a pilot metric, not a catalog fact.",
        "",
        "| Edge case | Family | Targets | Prerequisites | " + " | ".join(profiles) + " | Variants |",
        "|---|---|---|---|" + "---|" * len(profiles) + "---|",
    ]
    for case in documents["edge_case_catalog"]["edge_cases"]:
        eligible = ["yes" if p in case["language_profiles"] else "-" for p in profiles]
        variants = "<br>".join(
            f"{v['variant_id']} ({', '.join(v['profiles'])}; {v['status']})" for v in case["language_variants"]) or "-"
        lines.append(f"| {case['id']} | {case['family']} | {', '.join(case['allowed_target_question_ids'])} | "
                     f"{_prerequisites(case)} | " + " | ".join(eligible) + f" | {variants} |")
    lines += ["", "## Variant definitions", "", "| Edge case | Variant | Profiles | Status | Implementation | Purpose |",
              "|---|---|---|---|---|---|"]
    for case in documents["edge_case_catalog"]["edge_cases"]:
        for v in case["language_variants"]:
            lines.append(f"| {case['id']} | {v['variant_id']} | {', '.join(v['profiles'])} | {v['status']} | {_handling(v)} | {v['purpose']} |")
    return "\n".join(lines) + "\n"


def _table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def composition_report(dataset_root):
    root = Path(dataset_root)
    documents = load_stage0(root)
    summary = load_json(root / RECORD)["summary"]
    blocks = summary["by_language_profile"]
    names = list(blocks)
    plans = {p["language_profile"]: p for p in documents["generation_plan"]["language_profile_plans"]}
    compliance = list(summary["compliance_fail_counts"])
    lines = [
        "# Planned composition",
        "",
        f"From the validated `{RECORD}` of `{documents['normalization']['dataset_id']}`: {summary['calls']} calls, "
        f"{summary['total_compliance_fail_pairs']} compliance FAIL labels, {summary['total_edge_case_assignments']} edge-case "
        f"assignments, {summary['total_language_variants']} language variants.",
        "",
    ]

    def per_profile(key, keys=None):
        keys = keys or list(dict.fromkeys(k for n in names for k in blocks[n][key]))
        return [[k, *(blocks[n][key].get(k, 0) for n in names), sum(blocks[n][key].get(k, 0) for n in names)] for k in keys]

    header = ["", *names, "Total"]
    lines += _table("Calls, durations and composition", header,
                    [["calls", *(blocks[n]["calls"] for n in names), summary["calls"]]]
                    + per_profile("duration_profiles") + per_profile("call_composition"))
    lines += _table("Calls by FAIL count", header, per_profile("fails_per_call"))
    lines += _table("Calls by edge-case count", header, per_profile("edge_cases_per_call"))
    lines += _table("FAIL labels per compliance question", header,
                    [[q, *(blocks[n]["compliance_verdicts"][q]["FAIL"] for n in names), summary["compliance_fail_counts"][q]]
                     for q in compliance])
    lines += _table("C02/C03 outcomes", header, per_profile("c02_c03_outcomes"))
    lines += _table("Edge-case families", header, per_profile("edge_case_families"))
    lines += _table("Edge cases", header, per_profile("edge_case_ids"))
    variant_rows = sorted({(case_id, v) for n in names for case_id, by in blocks[n]["language_variants"].items() for v in by})
    lines += _table("Language variants", ["Edge case", "Variant", *names, "Total"],
                    [[case_id, v, *(blocks[n]["language_variants"].get(case_id, {}).get(v, 0) for n in names),
                      sum(blocks[n]["language_variants"].get(case_id, {}).get(v, 0) for n in names)] for case_id, v in variant_rows])
    lines += _table("Target exposure (distinct calls) against caps", ["Question", *(f"{n} (cap)" for n in names)],
                    [[q, *(f"{blocks[n]['target_exposure_calls'][q]} ({plans[n]['target_exposure_caps'][q]})" for n in names)]
                     for q in plans[names[0]]["target_exposure_caps"]])
    lines += _table("FAILs + edge cases per call, by duration", ["Profile", "Duration", "Load: calls"],
                    [[n, d, ", ".join(f"{k}: {v}" for k, v in loads.items())]
                     for n in names for d, loads in blocks[n]["fails_plus_edge_cases_by_duration"].items()])
    lines += _table("Currency, direction and order type", header,
                    per_profile("transaction_currency") + per_profile("transaction_direction") + per_profile("order_type"))
    lines += _table("Mixing styles", header, per_profile("mixing_styles"))
    lines += _table("Products", header, per_profile("transaction_instrument"))
    return "\n".join(lines)


# ---- coverage of every mechanism, and what a future development / held-out split would leave -------------------------------
def small_group_flag(count):
    """Flag for a group too small to treat as well represented."""
    if count == 1:
        return "SINGLETON"
    return "VERY SMALL" if count <= 3 else ""


def proportional_split(count, held_out_percent):
    """(development, held_out) examples of a group under a proportional split that keeps both sides non-empty when possible."""
    if count < 2:
        return count, 0
    held = min(count - 1, max(1, (count * held_out_percent + 50) // 100))
    return count - held, held


def split_flag(count, development, held_out):
    if count < 2:
        return "CANNOT APPEAR IN BOTH"
    return "ONE EXAMPLE IN A SUBSET" if min(development, held_out) == 1 else ""


def planned_call_support(root, documents):
    """{(language profile, mechanism): set of call IDs} from the G1 outputs, or None when they are missing or do not match the
    stored quotas (then distinct-call coverage is unknown). A mechanism is an edge-case ID or "<edge-case ID> / <variant ID>"."""
    root = Path(root)
    manifest_path, assignments_path = root / MANIFEST, root / ASSIGNMENTS
    if not (manifest_path.is_file() and assignments_path.is_file()):
        return None
    try:
        profile_of = {r["call_id"]: r["language_profile"] for r in parse_manifest(manifest_path.read_text(encoding="utf-8"))}
        assignments = parse_assignments(assignments_path.read_text(encoding="utf-8"))
        support = {}
        assigned = Counter()
        for a in assignments:
            profile = profile_of[a["call_id"]]
            keys = [a["edge_case_id"]] + ([f"{a['edge_case_id']} / {a['language_variant']}"] if a["language_variant"] != "NONE" else [])
            for key in keys:
                support.setdefault((profile, key), set()).add(a["call_id"])
                assigned[(profile, key)] += 1
    except (KeyError, ValueError, OSError):
        return None
    planned = Counter()
    for plan in documents["generation_plan"]["language_profile_plans"]:
        name = plan["language_profile"]
        planned.update({(name, case_id): n for case_id, n in plan["edge_case_counts"].items() if n})
        planned.update({(name, f"{case_id} / {variant}"): n for case_id, by in plan["language_variant_counts"].items()
                        for variant, n in by.items() if n})
    return support if assigned == planned else None


def coverage_groups(documents, support=None):
    """[(scope, mechanism, assignments, distinct calls or None)] for every case and variant, per language profile and across
    profiles (assignments > 0 only). `support` is planned_call_support(); without it the distinct calls are unknown (None),
    never assumed equal to the assignment count."""
    plans = documents["generation_plan"]["language_profile_plans"]
    cases = [c["id"] for c in documents["edge_case_catalog"]["edge_cases"]]
    groups = []

    def calls_of(profile, mechanism):
        return None if support is None else len(support.get((profile, mechanism), ()))

    for p in plans:
        name = p["language_profile"]
        for case_id in cases:
            if p["edge_case_counts"][case_id]:
                groups.append((name, case_id, p["edge_case_counts"][case_id], calls_of(name, case_id)))
        for case_id, by in p["language_variant_counts"].items():
            for variant_id, count in by.items():
                groups.append((name, f"{case_id} / {variant_id}", count, calls_of(name, f"{case_id} / {variant_id}")))
    by_mechanism = {}
    for _, mechanism, assignments, calls in groups:
        by_mechanism.setdefault(mechanism, []).append((assignments, calls))
    for mechanism, parts in by_mechanism.items():
        if len(parts) > 1:  # every call belongs to exactly one profile, so distinct calls add up across profiles
            known = all(calls is not None for _, calls in parts)
            groups.append(("ALL PROFILES", mechanism, sum(a for a, _ in parts), sum(c for _, c in parts) if known else None))
    return groups


def _count_cell(assignments, calls, eligible=True):
    """'assignments / distinct calls' with a size flag taken from the distinct calls (unknown when only quotas are available)."""
    if not eligible or not assignments:
        return "-"
    if calls is None:
        return f"{assignments} / unknown"
    flag = small_group_flag(calls)
    return f"{assignments} / {calls}" + (f" ({flag})" if flag else "")


def _total_cell(cell, mechanism):
    """The across-profile cell of a mechanism: the ALL PROFILES row, or the single profile that has it."""
    if ("ALL PROFILES", mechanism) in cell:
        return cell[("ALL PROFILES", mechanism)]
    return next((value for (scope, key), value in cell.items() if key == mechanism), (0, 0))


def coverage_report(dataset_root):
    root = Path(dataset_root)
    documents = load_stage0(root)
    plans = documents["generation_plan"]["language_profile_plans"]
    names = [p["language_profile"] for p in plans]
    catalog = documents["edge_case_catalog"]["edge_cases"]
    held_percent = 30
    design = root / "authoring/quota_design.json"
    if design.is_file():
        held_percent = load_json(design)["target"].get("future_split", {}).get("held_out_percent", held_percent)
    support = planned_call_support(root, documents)
    groups = coverage_groups(documents, support)
    cell = {(scope, mechanism): (assignments, calls) for scope, mechanism, assignments, calls in groups}
    lines = [
        "# Mechanism coverage and future-split assessment",
        "",
        f"Counts come from the resolved `generation_plan.json` of `{documents['normalization']['dataset_id']}`. Each cell reads "
        "`assignments / distinct calls`: an assignment is one planned challenge, but a split moves whole calls, and one call can "
        "carry several assignments of the same case (for example long-distance dependencies on different question pairs). "
        + ("Distinct calls come from the G1 outputs, which match the stored quotas."
           if support is not None else
           "**Distinct calls are unknown**: the G1 outputs are missing or do not match the stored quotas, so no call-level "
           "coverage or split assessment is produced and assignments are never used as a stand-in."),
        "",
    ]
    missing = [(p["language_profile"], c["id"]) for p in plans for c in catalog
               if p["language_profile"] in c["language_profiles"] and not p["edge_case_counts"][c["id"]]]
    lines.append("Eligible mechanisms with no examples: " + (", ".join(f"{n}/{c}" for n, c in missing) or "none") + ".")
    lines += ["", "### Edge cases per language profile (SINGLETON = 1 call, VERY SMALL = 2-3 calls)", "",
              "| Edge case | " + " | ".join(names) + " | Total |", "|---|" + "---|" * (len(names) + 1)]
    for c in catalog:
        cells = [_count_cell(*cell.get((p["language_profile"], c["id"]), (0, 0)), p["language_profile"] in c["language_profiles"])
                 for p in plans]
        lines.append(f"| {c['id']} | " + " | ".join(cells) + f" | {_count_cell(*_total_cell(cell, c['id']))} |")
    lines += ["", "### Language variants (both switch directions must appear wherever a cross-language case has examples)", "",
              "| Edge case | Variant | " + " | ".join(names) + " | Total |", "|---|---|" + "---|" * (len(names) + 1)]
    for c in catalog:
        for variant in c["language_variants"]:
            if variant["status"] != "READY_V1":
                continue
            key = f"{c['id']} / {variant['variant_id']}"
            cells = [_count_cell(*cell.get((p["language_profile"], key), (0, 0))) for p in plans]
            lines.append(f"| {c['id']} | {variant['variant_id']} | " + " | ".join(cells) + f" | {_count_cell(*_total_cell(cell, key))} |")
    one_direction = [f"{p['language_profile']}/{c}" for p in plans for c, by in p["language_variant_counts"].items()
                     if any(v.startswith("CROSS_LANGUAGE_") for v in by) and any(n < 1 for n in by.values())]
    lines += ["", "Cross-language cases missing a switch direction: " + (", ".join(one_direction) or "none") + ".", ""]

    if support is None:
        lines += ["### Future-split assessment", "",
                  "Not produced: distinct-call coverage is unknown without matching G1 outputs (run `plan_dataset.py` for the release).", ""]
        return "\n".join(lines)
    rows = []
    for scope, mechanism, assignments, calls in groups:
        development, held_out = proportional_split(calls, held_percent)
        flag = split_flag(calls, development, held_out)
        if flag:
            rows.append((scope, mechanism, assignments, calls, development, held_out, flag))
    lines += [f"### Groups a {100 - held_percent}/{held_percent} split of whole calls would leave thin ({len(rows)} of {len(groups)} groups)", "",
              "| Scope | Mechanism | Assignments | Distinct calls | Development | Held-out | Flag |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {scope} | {mechanism} | {a} | {n} | {d} | {h} | {flag} |" for scope, mechanism, a, n, d, h, flag in rows]
    never_both = [r for r in rows if r[6] == "CANNOT APPEAR IN BOTH"]
    lines += ["", "Groups that cannot appear in both subsets: " + (", ".join(f"{r[0]} {r[1]}" for r in never_both) or "none") + ".",
              "A group is flagged ONE EXAMPLE IN A SUBSET when a proportional split of its distinct calls leaves a single call in "
              "development or held-out. Per-profile groups are the strictest view; the ALL PROFILES rows count the same mechanism "
              "across profiles.", "",
              "**Not a feasibility proof.** Each group is estimated on its own. A call carries several assignments (about 2.2 on "
              "average) and belongs to one profile, so the groups overlap: calls placed in one subset for one mechanism also decide "
              "the others. The estimates are therefore only indicators; whether one split can satisfy every group at once is a "
              "joint problem that stays deferred with the splitter.", ""]
    return "\n".join(lines)
