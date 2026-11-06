"""List candidate domain terms for YOUR review: the words found right before annotated value mentions in the reference.

    python -B scripts/term_candidates.py --dataset advisory_execution_pilot_v1

Nothing is written. Copy the terms you approve into domain_terms.json and set "reviewed": true; only then does entity WER use them.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval.config import FYP_BE_ROOT
from transcribe_eval.reference import load_turns, selected_spans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="advisory_execution_pilot_v1")
    parser.add_argument("--words-before", type=int, default=3)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    root = FYP_BE_ROOT / "evaluation-dataset" / "datasets" / args.dataset
    found = defaultdict(Counter)
    for call in sorted(p.name for p in (root / "calls").glob("CALL_*")):
        turns = load_turns(root, call)
        spans = selected_spans(root, call, turns)
        if spans is None:
            print(f"{args.dataset} has no value annotations; nothing to suggest")
            return 2
        texts = {t["turn_id"]: t["text"] for t in turns}
        for span in spans:
            if not span["group"].startswith("value:"):
                continue
            before = re.findall(r"[A-Za-z][A-Za-z'/-]*", texts[span["turn_id"]][:span["start"]])[-args.words_before:]
            if before:
                found[span["group"]][" ".join(before)] += 1
    for group in sorted(found):
        print(f"\n{group}")
        for phrase, count in found[group].most_common(args.top):
            print(f"  {count:4d}  ... {phrase} <value>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
