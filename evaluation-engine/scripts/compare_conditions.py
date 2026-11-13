"""Paired clean-versus-environment_v2 comparison of two scored runs. Offline; no AWS access; safe to rerun.

    python -B scripts/compare_conditions.py --clean-run dev175-multilang-1 --v2-run <v2 run id>

Both runs must already be scored (score.py). The V2 run may have skipped the clean (C1) calls (collect.py --skip-clean-controls); their
results are then taken from the clean run, but only if the two runs used the same handler, TRANSCRIBE_* settings, Transcribe job
settings and normaliser. If not, this stops with "clean controls must be re-transcribed". Writes runs/<v2 run>/comparison_vs_<clean run>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval import compare as cmp
from transcribe_eval.collect import RUNS_ROOT


def _f(value, spec="+.2f"):
    return "-" if value is None else format(value, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-run", required=True)
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--scores", default="scores", help="scores folder name inside each run (default: scores)")
    parser.add_argument("--dataset-root", help="override the dataset folder used to read each call's language")
    args = parser.parse_args(argv)
    try:
        clean, v2 = cmp.load_run(RUNS_ROOT, args.clean_run, args.scores), cmp.load_run(RUNS_ROOT, args.v2_run, args.scores)
        result = cmp.compare(clean, v2, args.dataset_root)
    except cmp.CompareError as error:
        print(f"Cannot compare: {error}")
        return 2
    print(f"clean run {result['clean_run']} vs environment_v2 run {result['v2_run']} | equivalent settings: {result['equivalent']}")
    for text in result["reasons"]:
        print(f"  NOT equivalent: {text}")
    for text in result["warnings"]:
        print(f"  warning: {text}")
    if result["reused_clean_calls"]:
        print(f"  {result['reused_clean_calls']} clean control call(s) taken from the clean run (not transcribed again)")
    print(f"\n{'condition':<10}{'calls':>6}{'clean WER%':>12}{'V2 WER%':>10}{'delta pp':>10}{'95% CI':>20}{'entity d pp':>13}{'speaker d pp':>14}")
    for key, c in result["paired_by_condition"].items():
        ci = c["wer"]["ci95_pp"]
        print(f"{key:<10}{c['calls']:>6}{_f(c['wer']['clean'], '.2f'):>12}{_f(c['wer']['v2'], '.2f'):>10}{_f(c['wer']['delta_pp']):>10}"
              f"{('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']' if ci else '-'):>20}{_f(c['entity']['delta_pp']):>13}{_f(c['speaker']['delta_pp']):>14}")
    out = RUNS_ROOT / args.v2_run / f"comparison_vs_{args.clean_run}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
