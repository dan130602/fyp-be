"""Checkpoint selection with the FROZEN criteria (CPU; reads saved metrics only; never looks at the held-out calls).

    python -X utf8 sft_select.py --baseline <E00 eval dir> --run <fetched runs/RUN_ID> [--reviewed epoch-2 ...] [--stage raw]

Guardrails (raw stage, the 35 validation calls, same frozen inference pipeline; E00 = the matching 35-call baseline):
  G1 overall WER                <= E00 + 0.10 pp
  G2 each language WER          <= E00 + min(0.20 pp, 15 % of E00's rate)
  G3 critical-event spans       errors in event:* entity groups <= E00
  G4 decoding health            token-limit stops and abnormal stops <= E00
  G5 critical insertions        no NEW critical-looking inserted-word unit versus E00; otherwise the checkpoint is PENDING_CRITICAL_INSERTION_REVIEW and cannot be selected until someone lists it in --reviewed
Selection: among eligible checkpoints the lowest raw entity-error count; ties within 5 entity errors -> lower WER errors -> lower teacher-forced validation CE -> earlier epoch. Nothing eligible -> keep the original model.
"Improved" (a claim, separate from eligibility) additionally needs >= 20 fewer entity errors than E00 and (fixed - regressed entity spans) >= 15.
"""
import argparse
import json
from pathlib import Path

WER_TOTAL_PP, WER_LANG_PP, WER_LANG_REL = 0.10, 0.20, 0.15
IMPROVE_ENTITY_ERRORS, IMPROVE_NET_SPANS, TIE_ENTITY_ERRORS = 20, 15, 5
LANGS = ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event_errors(stage):
    return sum(v["errors"] for k, v in stage["entity_groups"].items() if k.startswith("event:"))


def assess(base, cand, stage="raw"):
    b, c = base["stages"][stage], cand["stages"][stage]
    gates, notes = {}, []
    pp = lambda blk: 100 * blk["wer"]["errors"] / blk["wer"]["reference_words"]            # noqa: E731
    gates["G1_overall_wer"] = pp(c["by_scope"]["ALL"]) <= pp(b["by_scope"]["ALL"]) + WER_TOTAL_PP
    for lang in LANGS:
        limit = pp(b["by_scope"][lang]) + min(WER_LANG_PP, WER_LANG_REL * pp(b["by_scope"][lang]))
        gates[f"G2_{lang}_wer"] = pp(c["by_scope"][lang]) <= limit
    gates["G3_critical_event_spans"] = event_errors(c) <= event_errors(b)
    bh, ch = base["decode_health"], cand["decode_health"]
    gates["G4_decoding_health"] = ch["token_limit_stops"] <= bh["token_limit_stops"] and ch["abnormal_stops"] <= bh["abnormal_stops"]
    new_ins = c["critical_insertions"]["new_units"]
    review = new_ins > 0
    entity_b, entity_c = b["by_scope"]["ALL"]["entity"]["errors"], c["by_scope"]["ALL"]["entity"]["errors"]
    rel = c.get("errors_vs_baseline", {})
    net = rel.get("fixed", 0) - rel.get("regression", 0)
    return {"gates": gates, "gates_pass": all(gates.values()), "critical_insertions_new_units": new_ins, "needs_critical_insertion_review": review,
            "entity_errors": entity_c, "entity_errors_e00": entity_b, "wer_errors": c["by_scope"]["ALL"]["wer"]["errors"], "wer_errors_e00": b["by_scope"]["ALL"]["wer"]["errors"],
            "fixed_minus_regressed_spans": net, "improved": (entity_b - entity_c) >= IMPROVE_ENTITY_ERRORS and net >= IMPROVE_NET_SPANS}


def select(base, candidates, val_ce, reviewed=(), stage="raw"):
    """candidates: {name: metrics}; val_ce: {name: teacher-forced validation CE}."""
    table = {name: assess(base, m, stage) for name, m in candidates.items()}
    for name, row in table.items():
        row["status"] = ("REJECTED" if not row["gates_pass"] else "PENDING_CRITICAL_INSERTION_REVIEW" if row["needs_critical_insertion_review"] and name not in reviewed else "ELIGIBLE")
    eligible = [n for n, r in table.items() if r["status"] == "ELIGIBLE"]
    chosen = None
    if eligible:
        best = min(table[n]["entity_errors"] for n in eligible)
        tied = [n for n in eligible if table[n]["entity_errors"] <= best + TIE_ENTITY_ERRORS]
        chosen = sorted(tied, key=lambda n: (table[n]["wer_errors"], val_ce.get(n, 9e9), int(n.split("-")[1])))[0] if len(tied) > 1 else tied[0]
    pending = [n for n, r in table.items() if r["status"].startswith("PENDING")]
    return {"selected": chosen, "decision": (f"select {chosen}" + (" (improvement thresholds met)" if table[chosen]["improved"] else " (eligible, but the improvement thresholds are NOT met)")) if chosen else
            ("undecided: " + ", ".join(pending) + " need the critical-insertion review") if pending else "keep the original model: no checkpoint satisfies the guardrails", "table": table}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reviewed", nargs="*", default=[])
    p.add_argument("--stage", default="raw")
    args = p.parse_args(argv)
    base = load(args.baseline / "metrics.json")
    cands = {p_.parent.name: load(p_) for p_ in sorted(args.run.glob("eval/epoch-*/metrics.json"))}
    epochs = {}
    summary = args.run / "epoch_summary.jsonl"
    if summary.is_file():
        epochs = {f"epoch-{r['epoch']}": r["validation"]["ce"] for r in (json.loads(l) for l in summary.read_text(encoding="utf-8").splitlines() if l.strip())}
    result = select(base, cands, epochs, args.reviewed, args.stage)
    (args.run / "selection.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = [f"# Checkpoint selection: {args.run.name}", "", f"Decision: **{result['decision']}**", "", "| checkpoint | status | entity errors (E00) | WER errors (E00) | fixed-regressed | improved | failed gates |", "|---|---|---:|---:|---:|---|---|"]
    for n, r in result["table"].items():
        lines.append(f"| {n} | {r['status']} | {r['entity_errors']} ({r['entity_errors_e00']}) | {r['wer_errors']} ({r['wer_errors_e00']}) | {r['fixed_minus_regressed_spans']} | {r['improved']} | {', '.join(k for k, v in r['gates'].items() if not v) or '-'} |")
    (args.run / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
