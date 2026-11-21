"""Eight report figures from SAVED numbers only (CPU, matplotlib; no model, no cluster).

    python -X utf8 sft_report.py --run <fetched runs/RUN_ID> [--baseline <fetched E00 eval dir>] [--out <figures dir>]

Figures: 1 CE vs steps, 2 weighted CE vs steps, 3 generated WER/entity WER by epoch, 4 entity vs non-entity CE, 5 WER by language, 6 entity WER by language, 7 substitutions/deletions/insertions, 8 entity WER by category.
Figures 5-8 compare THIS run (epoch 2 by default, --epoch) with a REFERENCE run: --reference <run folder>, or by default the run named in <runs folder>/reference.json ({"run": "E01-LR25", "epoch": 2}),
so a weighted run is judged against the previous CE baseline. The reference run itself, a run with no reference.json, or --no-reference compares with E00 instead (--include-base adds E00 as a third bar).

Reads train_metrics.jsonl, epoch_summary.jsonl and eval/epoch-*/metrics.json (written by sft_eval.py). Figures that need generated-text metrics are skipped, with a note, until those exist.
Titles say what is optimised: plain CE is the objective of CE runs and a diagnostic (not optimised) for weighted runs; WCE@w (w = the run's entity weight) is the objective of weighted runs and WCE@2 a fixed diagnostic for CE runs.
Training CE is measured with LoRA dropout on, validation CE with it off. Validation points are the epoch ends plus the mid-epoch points of val_midpoints.jsonl when the run has them.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANGS = ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")


def rows(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []


def load_evals(run):
    paths = sorted(Path(run, "eval").glob("epoch-*/metrics.json"), key=lambda q: int(q.parent.name.split("-")[1]))
    return {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in paths}


def rate(block, kind):
    value = block[kind]["rate"]
    return 100 * value if value is not None else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, help="E00 eval folder containing metrics.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--stage", default="raw", choices=("raw", "percentage", "final"))
    p.add_argument("--reference", type=Path, help="a previous run folder (e.g. the CE baseline E01-LR25): figures 5-7 then compare THIS run with that run instead of with E00")
    p.add_argument("--epoch", type=int, help="the epoch of this run shown in figures 5-7 (default: 2, the provisional checkpoint of the 3-epoch schedule, else the last epoch)")
    p.add_argument("--reference-epoch", type=int, help="the epoch of --reference to show (default: the same as --epoch)")
    p.add_argument("--no-reference", action="store_true", help="ignore the runs-folder reference.json and compare with E00 (when --baseline is given)")
    p.add_argument("--include-base", action="store_true", help="also show E00 (the untouched base model) in figures 5-7 when --baseline is given")
    args = p.parse_args(argv)
    ref_file = args.run.resolve().parent / "reference.json"        # {"run": "E01-LR25", "epoch": 2}: the previous experiment that figures 5-8 compare against by default
    if args.reference is None and not args.no_reference and ref_file.is_file():
        cfg = json.loads(ref_file.read_text(encoding="utf-8"))
        ref_dir = args.run.resolve().parent / cfg["run"]
        if ref_dir.is_dir() and ref_dir.name != args.run.resolve().name:
            args.reference = ref_dir
            args.reference_epoch = args.reference_epoch or cfg.get("epoch")
    out = args.out or args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))["config"]
    rid, weighted = config["run_id"], config["objective"] == "wce"
    weight = float(config.get("entity_weight", 2.0)) if weighted else 2.0
    mids = rows(args.run / "val_midpoints.jsonl")
    train, epochs = rows(args.run / "train_metrics.jsonl"), rows(args.run / "epoch_summary.jsonl")
    evals = load_evals(args.run)
    base = json.loads((args.baseline / "metrics.json").read_text(encoding="utf-8")) if args.baseline else None
    made, skipped = [], []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        made.append(name)

    def wce(m, w):
        """WCE@w from the stored token-pooled CE / EntCE (identical to the logged wce2 for w = 2); None when a part is N/A."""
        if m["ce"] is None or m["ent_ce"] is None or not m["tokens"]:
            return None
        return (m["ce"] * m["tokens"] + (w - 1) * m["ent_ce"] * m["entity_tokens"]) / (m["tokens"] + (w - 1) * m["entity_tokens"])

    validation = sorted([{"update": e["update"], "m": e["validation"]} for e in epochs] + [{"update": r["update"], "m": r["validation"]} for r in mids], key=lambda r: r["update"])
    val_label = "validation (dropout off, epoch ends + mid-epoch points)" if mids else "validation (dropout off, measured at epoch ends)"

    def curve(fn, title, name):
        fig, ax = plt.subplots(figsize=(7, 4))
        tpts = [(r["update"], fn(r)) for r in train if fn(r) is not None]
        ax.plot([u for u, _ in tpts], [v for _, v in tpts], label="train (10-step windows, LoRA dropout on)")
        points = [(v["update"], fn(v["m"])) for v in validation if fn(v["m"]) is not None]
        if points:
            ax.plot(*zip(*points), "o-", label=val_label)
        ax.set_xlabel("optimiser step"), ax.set_ylabel("loss"), ax.set_title(title), ax.legend(), ax.grid(alpha=0.3)
        save(fig, name)

    if train:
        curve(lambda m: m["ce"], f"{rid}: unweighted CE ({'diagnostic, not optimised' if weighted else 'OPTIMISED'})", "fig1_ce_vs_steps.png")
        curve(lambda m: wce(m, weight), f"{rid}: weighted CE WCE@{weight:g} ({'OPTIMISED' if weighted else 'diagnostic, not optimised'})", "fig2_wce2_vs_steps.png")
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in (("ent_ce", "train EntCE"), ("non_ent_ce", "train NonEntCE")):
            pts = [(r["update"], r[key]) for r in train if r[key] is not None]
            if pts:
                ax.plot(*zip(*pts), label=label)
        for key, label, style in (("ent_ce", "validation EntCE", "o--"), ("non_ent_ce", "validation NonEntCE", "s--")):
            pts = [(v["update"], v["m"][key]) for v in validation if v["m"][key] is not None]
            if pts:
                ax.plot(*zip(*pts), style, label=label)
        ax.set_xlabel("optimiser step"), ax.set_ylabel("CE"), ax.set_title(f"{rid}: entity vs non-entity CE"), ax.legend(), ax.grid(alpha=0.3)
        save(fig, "fig4_entity_vs_nonentity_ce.png")
    else:
        skipped.append("fig1,2,4 (no train_metrics.jsonl)")

    def scope(m):
        return m["stages"][args.stage]["by_scope"]

    if evals:
        xs = [int(k.split("-")[1]) for k in evals]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [rate(scope(m)["ALL"], "wer") for m in evals.values()], "o-", label="WER")
        ax.plot(xs, [rate(scope(m)["ALL"], "entity") for m in evals.values()], "s-", label="Entity WER")
        if base:
            ax.axhline(rate(scope(base)["ALL"], "wer"), ls=":", color="C0", label="E00 WER")
            ax.axhline(rate(scope(base)["ALL"], "entity"), ls=":", color="C1", label="E00 Entity WER")
        ax.set_xlabel("epoch"), ax.set_ylabel(f"% ({args.stage}, 35 validation calls)"), ax.set_xticks(xs), ax.legend(), ax.grid(alpha=0.3)
        ax.set_title(f"{rid}: generated validation WER / Entity WER")
        save(fig, "fig3_generated_wer_entity_wer.png")

        want = args.epoch if args.epoch else (2 if "epoch-2" in evals else int(list(evals)[-1].split("-")[1]))
        if f"epoch-{want}" not in evals:
            raise SystemExit(f"{rid} has no evaluation for epoch {want}")
        shown_name, shown = f"epoch-{want}", evals[f"epoch-{want}"]
        series = {}
        if args.reference:
            ref_evals = load_evals(args.reference)
            ref_epoch = args.reference_epoch or want
            if f"epoch-{ref_epoch}" not in ref_evals:
                raise SystemExit(f"{args.reference.name} has no evaluation for epoch {ref_epoch}")
            if base and args.include_base:
                series["E00 (base)"] = base
            series[f"{args.reference.name} epoch-{ref_epoch} (reference)"] = ref_evals[f"epoch-{ref_epoch}"]
        elif base:
            series["E00 (base)"] = base
        series[f"{rid} {shown_name}"] = shown
        vs = " vs ".join(series) if len(series) <= 2 else f"{rid} {shown_name} vs {args.reference.name}"
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (name, m) in enumerate(series.items()):
            ax.bar([j + i * width for j in range(len(LANGS))], [rate(scope(m)[lang], "wer") for lang in LANGS], width, label=name)
        ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(LANGS))])
        ax.set_xticklabels(LANGS)
        ax.set_ylabel(f"WER % ({args.stage}); Mandarin counted per character"), ax.legend(), ax.set_title(f"Error rate by language: {vs}", fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
        save(fig, "fig5_wer_by_language.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (name, m) in enumerate(series.items()):
            ax.bar([j + i * width for j in range(len(LANGS))], [rate(scope(m)[lang], "entity") for lang in LANGS], width, label=name)
        ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(LANGS))])
        ax.set_xticklabels(LANGS)
        ax.set_ylabel(f"Entity WER % ({args.stage})"), ax.legend(), ax.set_title(f"Entity WER by language: {vs}", fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
        save(fig, "fig6_entity_wer_by_language.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        kinds = ("substitutions", "deletions", "insertions")
        for i, (name, m) in enumerate(series.items()):
            a, bottom = scope(m)["ALL"]["wer"], 0.0
            for n, key in enumerate(kinds):
                v = 100 * a[key] / a["reference_words"]
                ax.bar(i, v, bottom=bottom, color=f"C{n}", label=key if i == 0 else None)
                bottom += v
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels(list(series))
        ax.set_ylabel("% of reference words"), ax.legend(), ax.set_title(f"Substitutions / deletions / insertions ({args.stage})", fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
        save(fig, "fig7_sdi_breakdown.png")

        groups = sorted(set().union(*(set(m["stages"][args.stage]["entity_groups"]) for m in series.values())))
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (name, m) in enumerate(series.items()):
            g = m["stages"][args.stage]["entity_groups"]
            ax.bar([j + i * width for j in range(len(groups))], [100 * g[k]["errors"] / g[k]["reference_words"] if k in g and g[k]["reference_words"] else 0 for k in groups], width, label=name)
        ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(groups))])
        ax.set_xticklabels(groups, rotation=60, ha="right")
        ax.set_ylabel("Entity WER %"), ax.legend(), ax.set_title(f"Entity WER by category: {vs}", fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        save(fig, "fig8_entity_wer_by_category.png")
    else:
        skipped.append("fig3,5,6,7,8 (no eval/epoch-*/metrics.json yet)")
    print(json.dumps({"figures": made, "skipped": skipped, "out": str(out)}))


if __name__ == "__main__":
    main()
