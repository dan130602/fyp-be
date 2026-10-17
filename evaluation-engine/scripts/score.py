"""Score a collected run offline against the dataset reference. No AWS access; safe to rerun.

    python -B scripts/score.py --run 20261016T230539Z-0d52af69

Writes runs/<run id>/scores/{summary.json, calls.jsonl, spans.jsonl, confusions.json, unnormalised.json}.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval.collect import RUNS_ROOT
from transcribe_eval.config import ENGINE_ROOT
from transcribe_eval.reference import load_terms
from transcribe_eval.score import score_run


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run id under runs/")
    args = parser.parse_args(argv)
    run_dir = RUNS_ROOT / args.run
    if not (run_dir / "metadata.json").is_file():
        print(f"No collected run at {run_dir}")
        return 2
    reviewed, terms = load_terms(ENGINE_ROOT / "domain_terms.json")
    summary = score_run(run_dir, terms=terms, terms_reviewed=reviewed)

    wer = summary["normalised_wer"]
    print(f"run {summary['run_id']} ({summary['dataset_id']}, {summary['calls_scored']} calls, normaliser {summary['normalizer_version']})")
    print(f"normalised WER {_pct(wer['wer'])}  N={wer['reference_words']}  S={wer['substitutions']}  D={wer['deletions']}  I={wer['insertions']}")
    entity = summary["entity_wer"]
    if entity.get("available") is False:
        print(f"entity WER     not available: {entity['reason']}")
    else:
        print(f"entity WER     {_pct(entity['wer'])}  N={entity['reference_words']}  S={entity['substitutions']}  "
              f"D={entity['deletions']}  I={entity['insertions']}  (domain terms: {'reviewed' if entity['domain_terms']['reviewed'] else 'none yet'})")
    conf = summary["average_confidence"]
    print(f"confidence     {conf['mean']} over {conf['items']} items (correct items {conf['mean_correct_items']}, others {conf['mean_other_items']})")
    spk = summary["speaker_attribution"]
    print(f"speaker        {_pct(spk['accuracy'])}  ({spk['correct_items']}/{spk['eligible_items']} AWS items)")
    print(f"details in {run_dir / 'scores'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
