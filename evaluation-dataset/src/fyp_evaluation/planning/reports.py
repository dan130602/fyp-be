"""Read-only Markdown reports for a multilingual release, derived from its maintained contracts.

`compatibility_report` renders the edge-case x language-profile matrix straight from edge_case_catalog.json (the
only source of truth; nothing here is maintained by hand). `composition_report` renders the planned composition
from the validated planning_record.json summary. Neither writes files or changes any contract.
"""

from pathlib import Path

from fyp_evaluation.validation.schemas import load_json
from fyp_evaluation.validation.stage0 import load_stage0

from .outputs import RECORD

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
