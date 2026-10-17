"""Score phase: Transcribe output against the dataset reference, fully offline (see README "Metric definitions").

One alignment per call (jiwer, on normalised tokens: the whole reference dialogue against all pronunciation items) feeds every metric.
Corpus figures are summed counts, never averaged rates.
"""

import importlib.metadata
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from .config import FYP_BE_ROOT
from .hypothesis import load_items, raw_words as hypothesis_words
from .normalize import NORMALIZER_VERSION, normalize_words
from .reference import load_turns, raw_words as reference_words, selected_spans

ROLES = ("EMPLOYEE", "CLIENT")


def align(ref, hyp):
    """Per reference token: ('equal'|'substitute'|'delete', hyp index or None); per hyp token: its op; insertions as (anchor, hyp index)."""
    ref_ops, hyp_ops, insertions = [None] * len(ref), [None] * len(hyp), []
    if not ref:
        return ref_ops, ["insert"] * len(hyp), [(0, k) for k in range(len(hyp))]
    if not hyp:
        return [("delete", None)] * len(ref), hyp_ops, insertions
    for chunk in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if chunk.type == "insert":
            for k in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                hyp_ops[k] = "insert"
                insertions.append((chunk.ref_start_idx, k))
        elif chunk.type == "delete":
            for r in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ref_ops[r] = ("delete", None)
        else:  # equal / substitute: same length on both sides
            for offset, r in enumerate(range(chunk.ref_start_idx, chunk.ref_end_idx)):
                ref_ops[r] = (chunk.type, chunk.hyp_start_idx + offset)
                hyp_ops[chunk.hyp_start_idx + offset] = chunk.type
    return ref_ops, hyp_ops, insertions


def _counts(ref_ops, insertions, selected=None):
    """S, D, I, N over all reference tokens, or over a selected set (insertions only when anchored strictly inside a selected run)."""
    indices = range(len(ref_ops)) if selected is None else sorted(selected)
    ops = Counter(ref_ops[r][0] for r in indices)
    if selected is None:
        inserted = len(insertions)
    else:
        inserted = sum(1 for anchor, _ in insertions if anchor - 1 in selected and anchor in selected)
    n = len(indices)
    errors = ops["substitute"] + ops["delete"] + inserted
    return {"reference_words": n, "substitutions": ops["substitute"], "deletions": ops["delete"], "insertions": inserted,
            "hits": ops["equal"], "errors": errors, "wer": round(errors / n, 4) if n else None}


def _speaker(items, item_tokens, hyp_ops, ref_ops_by_hyp, ref_speakers):
    """Per original AWS item: eligible when every token it produced is a hit; correct when its mapped label is the reference speaker."""
    eligible = []
    for item in items:
        tokens = item_tokens.get(item["item"], [])
        if not tokens or any(hyp_ops[k] != "equal" for k in tokens):
            continue
        speakers = {ref_speakers[ref_ops_by_hyp[k]] for k in tokens}
        if len(speakers) == 1:
            eligible.append((item["speaker"], speakers.pop()))
    labels = sorted({item["speaker"] for item in items if item["speaker"]})
    best, best_map = -1, {}
    for roles in itertools.permutations(ROLES + (None,) * max(0, len(labels) - len(ROLES)), len(labels)):
        mapping = dict(zip(labels, roles))
        score = sum(1 for label, speaker in eligible if mapping.get(label) == speaker)
        if score > best:
            best, best_map = score, mapping
    return {"eligible_items": len(eligible), "correct_items": max(best, 0), "speakers_detected": len(labels),
            "label_mapping": best_map, "accuracy": round(best / len(eligible), 4) if eligible else None}


def score_call(turns, items, breaks_after_item, spans):
    words, positions, speakers, ref_breaks = reference_words(turns)
    ref, ref_origins, ref_notes = normalize_words(words, ref_breaks)
    hwords, item_of_word, hyp_breaks = hypothesis_words(items, breaks_after_item)
    hyp, hyp_origins, hyp_notes = normalize_words(hwords, hyp_breaks)
    ref_ops, hyp_ops, insertions = align(ref, hyp)
    ref_speakers = [speakers[origin[0]] for origin in ref_origins]
    ref_by_hyp = {op[1]: r for r, op in enumerate(ref_ops) if op[0] != "delete"}
    item_tokens = {}
    for k, origin in enumerate(hyp_origins):
        for item in {item_of_word[w] for w in origin}:
            item_tokens.setdefault(item, []).append(k)

    token_spans = []
    for origin in ref_origins:  # each normalised token's character span in its turn
        token_spans.append((positions[origin[0]][0], min(positions[w][1] for w in origin), max(positions[w][2] for w in origin)))

    def members_of(span):
        return {r for r, (turn, start, end) in enumerate(token_spans) if turn == span["turn_id"] and start < span["end"] and span["start"] < end}

    entity, span_rows = None, []
    if spans is not None:
        selected = set()
        for span in spans:
            members = members_of(span)
            selected |= members
            span_rows.append({**span, "tokens": [ref[r] for r in sorted(members)], **_counts(ref_ops, insertions, members),
                              "hypothesis_tokens": [hyp[ref_ops[r][1]] if ref_ops[r][1] is not None else "-" for r in sorted(members)]})
        entity = _counts(ref_ops, insertions, selected)
        entity["groups"] = {group: _counts(ref_ops, insertions, set().union(*(members_of(s) for s in spans if s["group"] == group)))
                            for group in sorted({s["group"] for s in spans})}

    confidences = [item["confidence"] for item in items if item["confidence"] is not None]
    hit_items = {item["item"] for item in items if item_tokens.get(item["item"]) and all(hyp_ops[k] == "equal" for k in item_tokens[item["item"]])}
    hit_conf = [item["confidence"] for item in items if item["item"] in hit_items and item["confidence"] is not None]
    other_conf = [item["confidence"] for item in items if item["item"] not in hit_items and item["confidence"] is not None]
    substitutions = Counter((ref[r], hyp[op[1]]) for r, op in enumerate(ref_ops) if op[0] == "substitute")
    return {
        "wer": _counts(ref_ops, insertions),
        "entity_wer": entity,
        "confidence": {"items": len(confidences), "sum": sum(confidences), "hit_items": len(hit_conf), "hit_sum": sum(hit_conf),
                       "other_items": len(other_conf), "other_sum": sum(other_conf)},
        "speaker": _speaker(items, item_tokens, hyp_ops, ref_by_hyp, ref_speakers),
        "substitutions": substitutions,
        "deletions": Counter(ref[r] for r, op in enumerate(ref_ops) if op[0] == "delete"),
        "insertions": Counter(hyp[k] for _, k in insertions),
        "notes": {"reference": ref_notes, "hypothesis": hyp_notes},
        "spans": span_rows,
        "reference_tokens": len(ref), "hypothesis_tokens": len(hyp),
    }


def _sum_counts(rows):
    keys = ("reference_words", "substitutions", "deletions", "insertions", "hits")
    total = {key: sum(row[key] for row in rows) for key in keys}
    total["errors"] = total["substitutions"] + total["deletions"] + total["insertions"]
    total["wer"] = round(total["errors"] / total["reference_words"], 4) if total["reference_words"] else None
    return total


def score_run(run_dir, terms=(), terms_reviewed=False, dataset_root=None):
    """Score every completed call of a collect run; writes runs/<id>/scores/ and returns the summary."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset_root = Path(dataset_root) if dataset_root else FYP_BE_ROOT / "evaluation-dataset" / "datasets" / metadata["dataset_id"]
    calls, span_rows, notes = [], [], {"reference": Counter(), "hypothesis": Counter()}
    examples = {"reference": {}, "hypothesis": {}}
    subs, dels, ins = Counter(), Counter(), Counter()
    for entry in metadata["calls"]:
        if entry["status"] != "COMPLETED":
            continue
        call_id = entry["call_id"]
        turns = load_turns(dataset_root, call_id)
        items, breaks, hyp_text = load_items(run_dir / entry["transcript_file"])
        spans = selected_spans(dataset_root, call_id, turns, terms)
        result = score_call(turns, items, breaks, spans)
        subs.update(result["substitutions"]), dels.update(result["deletions"]), ins.update(result["insertions"])
        for side in ("reference", "hypothesis"):
            for category, text in result["notes"][side]:
                notes[side][category] += 1
                examples[side].setdefault(category, Counter())[text] += 1
        span_rows.extend({"call_id": call_id, **row} for row in result["spans"])
        calls.append({"call_id": call_id, "wer": result["wer"], "entity_wer": result["entity_wer"], "speaker": result["speaker"],
                      "confidence": result["confidence"], "raw_reference_text": "\n".join(t["text"] for t in turns),
                      "raw_hypothesis_text": hyp_text})

    entity_rows = [c["entity_wer"] for c in calls if c["entity_wer"] is not None]
    entity_available = bool(calls) and len(entity_rows) == len(calls)
    conf = {key: sum(c["confidence"][key] for c in calls) for key in ("items", "sum", "hit_items", "hit_sum", "other_items", "other_sum")}
    speaker_eligible = sum(c["speaker"]["eligible_items"] for c in calls)
    speaker_correct = sum(c["speaker"]["correct_items"] for c in calls)
    groups = {}
    if entity_available:
        for row in entity_rows:
            for group, counts in row["groups"].items():
                groups.setdefault(group, []).append(counts)
    summary = {
        "run_id": metadata["run_id"], "scored_at": datetime.now(timezone.utc).isoformat(), "dataset_id": metadata["dataset_id"],
        "normalizer_version": NORMALIZER_VERSION, "jiwer_version": importlib.metadata.version("jiwer"), "calls_scored": len(calls),
        "calls_not_completed": [e["call_id"] for e in metadata["calls"] if e["status"] != "COMPLETED"],
        "normalised_wer": _sum_counts([c["wer"] for c in calls]),
        "entity_wer": ({**_sum_counts(entity_rows), "groups": {g: _sum_counts(rows) for g, rows in sorted(groups.items())},
                        "domain_terms": {"reviewed": terms_reviewed, "terms": len(terms)}}
                       if entity_available else {"available": False, "reason": "this dataset has no value/event annotations"}),
        "average_confidence": {"items": conf["items"], "mean": round(conf["sum"] / conf["items"], 4) if conf["items"] else None,
                               "mean_correct_items": round(conf["hit_sum"] / conf["hit_items"], 4) if conf["hit_items"] else None,
                               "mean_other_items": round(conf["other_sum"] / conf["other_items"], 4) if conf["other_items"] else None},
        "speaker_attribution": {"counting_unit": "original AWS pronunciation item", "eligible_items": speaker_eligible,
                                "correct_items": speaker_correct,
                                "accuracy": round(speaker_correct / speaker_eligible, 4) if speaker_eligible else None},
        "worst_calls": [{"call_id": c["call_id"], "wer": c["wer"]["wer"]} for c in sorted(calls, key=lambda c: -(c["wer"]["wer"] or 0))[:5]],
    }
    out = run_dir / "scores"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "calls.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in calls), encoding="utf-8")
    (out / "spans.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in span_rows), encoding="utf-8")
    (out / "confusions.json").write_text(json.dumps({
        "substitutions": [{"reference": r, "hypothesis": h, "count": n} for (r, h), n in subs.most_common(50)],
        "deletions": [{"reference": r, "count": n} for r, n in dels.most_common(30)],
        "insertions": [{"hypothesis": h, "count": n} for h, n in ins.most_common(30)]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "unnormalised.json").write_text(json.dumps({side: {category: {"count": notes[side][category],
                                                                          "examples": dict(examples[side][category].most_common(10))}
                                                              for category in notes[side]} for side in notes},
                                                      indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
