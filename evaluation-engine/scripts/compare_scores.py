"""Explain exactly what a normaliser change did to a run's scores. Offline; no AWS access; safe to rerun.

    python -B scripts/compare_scores.py --run 20261016T230539Z-0d52af69

Needs the previous scores kept as runs/<run id>/scores_1.2.0/ (copy `scores` there BEFORE re-scoring with a new normaliser) and the new
`scores/`. It then:
  1. proves the baseline: scoring with every scoring-only rule switched off must reproduce scores_1.2.0 exactly, so every difference
     below is caused by the rules and nothing else;
  2. attributes each removed error to a rule by switching rules off one at a time (per turn, so an explanation never needs a guess);
  3. writes runs/<run id>/scores/changes_from_1.2.0.json and prints a short report.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_eval.collect import RUNS_ROOT
from transcribe_eval.config import ENGINE_ROOT
from transcribe_eval.normalize import ALL_RULES, RULES, rules_info
from transcribe_eval.reference import load_terms
from transcribe_eval.score import load_call_inputs, score_call

OLD = "scores_1.2.0"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _units(directory, call_id):
    return _read(Path(directory) / "units" / f"{call_id}.json")["units"]


def _error_units(units):
    return [{"turn_id": u["turn_id"], "ref_text": u["ref_text"], "hyp_text": u["hyp_text"], "substitutions": u["substitutions"],
             "deletions": u["deletions"], "insertions": u["insertions"]} for u in units if u["errors"]]


def _turn_errors(units):
    errors = {}
    for u in units:
        errors[u["turn_id"]] = errors.get(u["turn_id"], 0) + u["errors"]
    return errors


def _counts(result):
    wer = result["wer"]
    return {key: wer[key] for key in ("reference_words", "substitutions", "deletions", "insertions", "errors", "wer")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    run_dir = RUNS_ROOT / args.run
    old_dir, new_dir = run_dir / OLD, run_dir / "scores"
    if not (old_dir / "summary.json").is_file() or not (new_dir / "summary.json").is_file():
        print(f"Need {old_dir} (the previous scores) and {new_dir} (the new scores)")
        return 2
    reviewed, terms = load_terms(ENGINE_ROOT / "domain_terms.json")
    inputs = load_call_inputs(run_dir, terms)

    def score_all(rules):
        return {call_id: score_call(rules=rules, **kwargs) for call_id, kwargs, _, _ in inputs}

    baseline = score_all(frozenset())
    full = score_all(ALL_RULES)
    ablated = {rule: score_all(ALL_RULES - {rule}) for rule in RULES}
    alone = {rule: score_all(frozenset({rule})) for rule in RULES}

    old_summary, new_summary = _read(old_dir / "summary.json"), _read(new_dir / "summary.json")
    old_calls = {c["call_id"]: c for c in map(json.loads, (old_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines())}
    reproduced = all(_counts(baseline[call_id]) == {k: old_calls[call_id]["wer"][k] for k in _counts(baseline[call_id])}
                     and _error_units(baseline[call_id]["units"]) == _error_units(_units(old_dir, call_id)) for call_id in baseline)
    consistent = all(_counts(full[call_id]) == {k: c["wer"][k] for k in _counts(full[call_id])}
                     for call_id, c in ((c["call_id"], c) for c in map(json.loads, (new_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines())))

    per_call, removed, remaining, appeared = [], [], [], []
    for call_id in baseline:
        old_u, new_u = _units(old_dir, call_id), _units(new_dir, call_id)
        per_call.append({"call_id": call_id, "before": _counts(baseline[call_id]), "after": _counts(full[call_id])})
        new_errors = _error_units(new_u)
        new_keys = {(e["turn_id"], e["ref_text"], e["hyp_text"]) for e in new_errors}
        old_keys = {(e["turn_id"], e["ref_text"], e["hyp_text"]) for e in _error_units(old_u)}
        full_by_turn = _turn_errors(full[call_id]["units"])
        blame = {rule: _turn_errors(ablated[rule][call_id]["units"]) for rule in RULES}
        for error in _error_units(old_u):
            if (error["turn_id"], error["ref_text"], error["hyp_text"]) in new_keys:
                remaining.append({"call_id": call_id, **error})
                continue
            turn = error["turn_id"]
            rules = [rule for rule in RULES if blame[rule].get(turn, 0) > full_by_turn.get(turn, 0)]
            removed.append({"call_id": call_id, **error, "explained_by": rules or ["combination of rules"]})
        for error in new_errors:
            if (error["turn_id"], error["ref_text"], error["hyp_text"]) not in old_keys:
                appeared.append({"call_id": call_id, **error})

    def total(results):
        return sum(r["wer"]["errors"] for r in results.values())

    by_rule = {rule: {"errors_removed_when_all_rules_on": total(ablated[rule]) - total(full),
                      "errors_removed_when_only_this_rule_on": total(baseline) - total(alone[rule])} for rule in RULES}
    report = {
        "run_id": args.run, "dataset_id": new_summary["dataset_id"],
        "from": {"normalizer_version": old_summary["normalizer_version"], "scores_dir": OLD, "scored_at": old_summary["scored_at"],
                 "normalised_wer": old_summary["normalised_wer"]},
        "to": {"normalizer_version": new_summary["normalizer_version"], "scores_dir": "scores", "scored_at": new_summary["scored_at"],
               "normalised_wer": new_summary["normalised_wer"], "rules": rules_info()},
        "baseline_reproduced": reproduced, "new_scores_match_a_fresh_run": consistent,
        "per_call": per_call, "by_rule": by_rule,
        "removed_errors": removed, "remaining_errors": remaining, "errors_that_appeared": appeared,
    }
    (new_dir / "changes_from_1.2.0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    before, after = old_summary["normalised_wer"], new_summary["normalised_wer"]
    print(f"run {args.run} ({report['dataset_id']}): {old_summary['normalizer_version']} -> {new_summary['normalizer_version']}")
    print(f"baseline reproduced exactly with all new rules off: {reproduced};  new scores equal a fresh scoring: {consistent}")
    print(f"errors {before['errors']} -> {after['errors']}   (S {before['substitutions']}->{after['substitutions']}, D {before['deletions']}->{after['deletions']}, "
          f"I {before['insertions']}->{after['insertions']})   WER {before['wer']} -> {after['wer']}   reference words {before['reference_words']} -> {after['reference_words']}")
    for row in per_call:
        b, a = row["before"], row["after"]
        print(f"  {row['call_id']}: errors {b['errors']} -> {a['errors']}  (WER {b['wer']} -> {a['wer']})")
    for rule, counts in by_rule.items():
        print(f"  rule {rule:<13} removes {counts['errors_removed_when_all_rules_on']} errors with all rules on, "
              f"{counts['errors_removed_when_only_this_rule_on']} on its own")
    print("removed error groups:")
    for item in removed:
        print(f"  {item['call_id']} turn {item['turn_id']}: {item['ref_text']!r} -> {item['hyp_text']!r}  [{', '.join(item['explained_by'])}]")
    print(f"errors that remain: {len(remaining)}; errors that did not exist before: {len(appeared)}")
    for item in appeared:
        print(f"  NEW {item['call_id']} turn {item['turn_id']}: {item['ref_text']!r} -> {item['hyp_text']!r}")
    return 0 if reproduced and consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
