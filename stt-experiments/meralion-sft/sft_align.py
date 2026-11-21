"""Recover the >30 s turns as reliably aligned <=30 s segments (CPU part; the aligner itself runs on the cluster in run_align_long.py).

Pipeline: `inputs` (local; reference text + entity spans + turn times of the long turns, development calls only)
          -> run_align_long.py on Origami (Qwen3-ForcedAligner-0.6B, the project's existing tool, aligns the KNOWN reference text against the turn's audio; also saves a 10 ms RMS track)
          -> `segment` (this module): map units to reference characters, choose cuts, verify, write the segments.

Rules: the reference text is never changed and never split by duration; every cut is a gap between two aligned units, preferably a sentence end, then a clause break, then a word gap, never inside a
number/identifier run, never inside an entity span when avoidable, preferably in real silence (cut time = lowest-energy 10 ms frame in the gap). The pieces partition the turn text and the turn audio exactly
(no overlap, no loss), each is <= 29.5 s. A turn is UNRESOLVED (reported, not guessed) when the aligned units do not reproduce the reference text, are non-monotonic or out of range, do not cover the turn
audio, or no cut plan satisfies the limits with cuts that sit in low-energy audio.
"""
import json
import math
import re
import unicodedata
from pathlib import Path

import sft_data as D

MAX_PIECE_SECONDS = 29.5
SENTENCE_END = set("。！？.!?…")
CLAUSE = set("，,；;：:、—")
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
NUMERAL_CHARS = set("零〇一二三四五六七八九十百千万亿两點点") | set("0123456789")
FRAME = 0.010


def canonical(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def long_turn_inputs(calls, split):
    """The long turns of every development call (same test as build_clips) with text, times and entity spans: input of the aligner and of `segment`."""
    rows = []
    for call_id, call in sorted(calls.items()):
        for t in call["turns"]:
            if round(t["end"] * D.SAMPLE_RATE) - round(t["start"] * D.SAMPLE_RATE) > D.MAX_CLIP_SAMPLES:
                rows.append({"call_id": call_id, "turn_id": t["turn_id"], "split": split[call_id], "speaker": t["speaker"], "language": call["language"],
                             "start": t["start"], "end": t["end"], "start_sample": round(t["start"] * D.SAMPLE_RATE), "end_sample": round(t["end"] * D.SAMPLE_RATE),
                             "text": t["text"], "spans": [{"start": s["start"], "end": s["end"], "group": s["group"], "label": s["label"]} for s in call["spans"] if s["turn_id"] == t["turn_id"]],
                             "evidence": t["turn_id"] in call["evidence_turns"]})
    return rows


def unit_ranges(text, units):
    """[(char_start, char_end)] of every aligned unit inside `text`, or (None, reason) when the units do not reproduce the text (canonical alphanumerics must match exactly)."""
    owner = []
    for index, ch in enumerate(text):
        owner += [index] * len(canonical(ch))
    flat = "".join(canonical(ch) for ch in text)
    ranges, cursor = [], 0
    lexical = [(u, canonical(u["text"])) for u in units]
    if "".join(c for _, c in lexical) != flat:
        return None, "alignment_text_mismatch"
    for unit, canon in lexical:
        if not canon:
            ranges.append(None)
            continue
        ranges.append((owner[cursor], owner[cursor + len(canon) - 1] + 1))
        cursor += len(canon)
    return ranges, None


def check_units(units, duration):
    previous = -1.0
    for u in units:
        s, e = float(u["start_time"]), float(u["end_time"])
        if not (math.isfinite(s) and math.isfinite(e)) or s < 0 or e < s or e > duration + 0.1 or s < previous - 1e-6:
            return "invalid_or_non_monotonic_times"
        previous = s
    return None


def cut_char(text, left_end, right_start):
    """Character where the next piece begins: after the last space or clause/sentence mark of the gap, so attached symbols ($, opening quotes) stay with the word they belong to."""
    gap = text[left_end:right_start]
    last = -1
    for i, ch in enumerate(gap):
        if ch.isspace() or ch in SENTENCE_END or ch in CLAUSE:
            last = i
    return left_end + last + 1 if last >= 0 else right_start


def gap_cost(text, left_range, right_range, spans, gap_seconds):
    """(cost, class, cut character) of cutting between two units (lower is better)."""
    between = text[left_range[1]:right_range[0]]
    cut = cut_char(text, left_range[1], right_range[0])
    inside = any(s["start"] < cut < s["end"] for s in spans)
    if any(ch in SENTENCE_END for ch in between):
        base, kind = 0.0, "sentence"
    elif any(ch in CLAUSE for ch in between):
        base, kind = 1.0, "clause"
    elif any(ch.isspace() for ch in between):
        base, kind = 3.0, "word_gap"
    else:
        last, first = text[left_range[1] - 1:left_range[1]], text[right_range[0]:right_range[0] + 1]
        numeric = (last in NUMERAL_CHARS or last.isdigit()) and (first in NUMERAL_CHARS or first.isdigit())
        latin_run = last.isalnum() and first.isalnum() and not (HAN.match(last) or HAN.match(first))
        base, kind = (40.0, "inside_number_or_identifier") if (numeric or latin_run) else (6.0, "han_boundary")
    cost = base - min(max(gap_seconds, 0.0), 0.4) * 5.0
    if inside:
        cost += 100.0
        kind += "+inside_entity"
    return cost, kind, cut


def refine_cut(rms, left_end, right_start):
    """Lowest-energy 10 ms frame inside the gap (widened by 40 ms each side); returns (time, rms there)."""
    lo, hi = (left_end, right_start) if right_start >= left_end else (right_start, left_end)
    lo, hi = max(0.0, lo - 0.04), hi + 0.04
    a, b = max(0, int(lo / FRAME)), min(len(rms) - 1, int(hi / FRAME))
    if b < a:
        return (left_end + right_start) / 2, None
    best = min(range(a, b + 1), key=lambda i: (rms[i], abs(i * FRAME - (left_end + right_start) / 2)))
    return (best + 0.5) * FRAME, rms[best]


def plan_turn(item, alignment, max_piece=MAX_PIECE_SECONDS):
    """Cut plan for one long turn, or {'status': 'unresolved', 'reason': ...}."""
    text, spans = item["text"], item["spans"]
    duration = item["end"] - item["start"]
    units = [u for u in alignment["units"]]
    reason = check_units(units, duration)
    if reason:
        return {"status": "unresolved", "reason": reason}
    ranges, reason = unit_ranges(text, units)
    if reason:
        return {"status": "unresolved", "reason": reason}
    keep = [i for i, r in enumerate(ranges) if r is not None]
    units, ranges = [units[i] for i in keep], [ranges[i] for i in keep]
    if not units:
        return {"status": "unresolved", "reason": "no_aligned_units"}
    first, last = float(units[0]["start_time"]), float(units[-1]["end_time"])
    if first > 2.0 or duration - last > 3.0:
        return {"status": "unresolved", "reason": f"units_do_not_cover_the_turn_audio (first {first:.2f}s, last ends {duration - last:.2f}s before the end)"}
    rms = alignment["rms"]
    levels = []
    for u in units:
        lo, hi = int(float(u["start_time"]) / FRAME), max(int(float(u["start_time"]) / FRAME) + 1, int(float(u["end_time"]) / FRAME))
        chunk = rms[lo:hi]
        if chunk:
            levels.append(sum(chunk) / len(chunk))
    levels.sort()
    typical = levels[len(levels) // 2] if levels else 0.0
    # candidate cuts: after unit k (k = 0..n-2): time, cost, class, char position
    cuts = []
    for k in range(len(units) - 1):
        gap = float(units[k + 1]["start_time"]) - float(units[k]["end_time"])
        cost, kind, char = gap_cost(text, ranges[k], ranges[k + 1], spans, gap)
        t, level = refine_cut(rms, float(units[k]["end_time"]), float(units[k + 1]["start_time"]))
        ratio = (level / typical) if (level is not None and typical > 0) else None
        if ratio is not None and ratio > 0.5:
            cost += 60.0                                          # a cut in loud speech: avoid, never choose unless nothing else exists (then the turn is unresolved below)
        cuts.append({"after_unit": k, "time": t, "cost": cost, "kind": kind, "char": char, "rms_ratio": ratio})
    # DP over cut positions: piece = (previous cut time, this cut time] <= max_piece
    n = len(cuts)
    INF = float("inf")
    best = [INF] * (n + 1)                                         # best[i]: cost of a plan whose last cut is cut i-1 (i=0: start)
    back = [None] * (n + 1)
    best[0] = 0.0
    times = [0.0] + [c["time"] for c in cuts]
    for i in range(1, n + 1):
        for j in range(i):
            if best[j] == INF or times[i] - times[j] > max_piece:
                continue
            piece = best[j] + cuts[i - 1]["cost"] + 5.0 + (3.0 if times[i] - times[j] < 4.0 else 0.0)
            if piece < best[i]:
                best[i], back[i] = piece, j
    end_choice, end_cost = None, INF
    for j in range(n + 1):
        if best[j] < INF and duration - times[j] <= max_piece and best[j] + 5.0 < end_cost:
            end_choice, end_cost = j, best[j] + 5.0
    if end_choice is None:
        return {"status": "unresolved", "reason": "no_cut_plan_within_the_30s_limit"}
    chosen, j = [], end_choice
    while j:
        chosen.append(cuts[j - 1])
        j = back[j]
    chosen.reverse()
    bad = [c for c in chosen if c["rms_ratio"] is not None and c["rms_ratio"] > 0.5]
    if bad:
        return {"status": "unresolved", "reason": f"only cuts inside loud speech satisfy the limit ({len(bad)})"}
    boundaries = [0.0] + [c["time"] for c in chosen] + [duration]
    char_cuts = [0] + [c["char"] for c in chosen] + [len(text)]
    pieces = []
    for p in range(len(boundaries) - 1):
        pieces.append({"piece": p + 1, "t0": boundaries[p], "t1": boundaries[p + 1], "c0": char_cuts[p], "c1": char_cuts[p + 1]})
    return {"status": "resolved", "pieces": pieces, "cuts": [{k: c[k] for k in ("time", "kind", "rms_ratio", "char")} for c in chosen],
            "inside_entity_cuts": [c for c in chosen if "inside_entity" in c["kind"]], "typical_rms": typical, "aligned_units": len(units)}


def pieces_to_clips(item, plan):
    """Clip dicts for the pieces of a resolved turn: sample bounds are absolute, text/spans are exact substrings/offsets of the turn text."""
    text, out = item["text"], []
    base_start, base_end = item["start_sample"], item["end_sample"]
    for piece in plan["pieces"]:
        a = base_start if piece["piece"] == 1 else base_start + round(piece["t0"] * D.SAMPLE_RATE)
        b = base_end if piece["piece"] == len(plan["pieces"]) else base_start + round(piece["t1"] * D.SAMPLE_RATE)
        raw = text[piece["c0"]:piece["c1"]]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        spans = []
        for s in item["spans"]:
            lo, hi = max(s["start"], piece["c0"]), min(s["end"], piece["c1"])
            if lo < hi:
                lo2, hi2 = lo - piece["c0"] - lead, hi - piece["c0"] - lead
                lo2, hi2 = max(lo2, 0), min(hi2, len(body))
                if lo2 < hi2:
                    spans.append({"start": lo2, "end": hi2, "group": s["group"], "label": s["label"]})
        out.append({"clip_id": f"{item['call_id']}:{item['turn_id']}#{piece['piece']}of{len(plan['pieces'])}", "call_id": item["call_id"], "split": item["split"], "language": item["language"],
                    "turn_ids": [item["turn_id"]], "speakers": [item["speaker"]], "start_seconds": a / D.SAMPLE_RATE, "end_seconds": b / D.SAMPLE_RATE,
                    "duration_seconds": round((b - a) / D.SAMPLE_RATE, 3), "start_sample": a, "end_sample": b, "text": body, "entity_spans": sorted(spans, key=lambda s: (s["start"], s["end"])),
                    "evidence_turns": [item["turn_id"]] if item["evidence"] else [], "parent": {"call_id": item["call_id"], "turn_id": item["turn_id"], "piece": piece["piece"], "pieces": len(plan["pieces"]),
                                                                                                 "turn_char_start": piece["c0"] + lead, "turn_char_end": piece["c0"] + lead + len(body),
                                                                                                 "turn_sample_start": base_start, "turn_sample_end": base_end}})
    return out


def verify_turn(item, clips):
    """Independent invariants: partition of text and audio, limits, entity coverage. Returns a list of problems."""
    problems = []
    text = item["text"]
    covered = "".join(text[c["parent"]["turn_char_start"]:c["parent"]["turn_char_end"]] for c in clips)
    if re.sub(r"\s+", "", covered) != re.sub(r"\s+", "", text):
        problems.append("text pieces do not reproduce the turn text exactly (missing or duplicated text)")
    for c in clips:
        if text[c["parent"]["turn_char_start"]:c["parent"]["turn_char_end"]] != c["text"]:
            problems.append(f"{c['clip_id']}: clip text is not the exact reference substring")
        if c["end_sample"] - c["start_sample"] > D.MAX_CLIP_SAMPLES:
            problems.append(f"{c['clip_id']}: longer than 30 s")
    if clips[0]["start_sample"] != item["start_sample"] or clips[-1]["end_sample"] != item["end_sample"]:
        problems.append("audio does not start/end at the turn bounds")
    for a, b in zip(clips, clips[1:]):
        if a["end_sample"] != b["start_sample"]:
            problems.append("audio pieces overlap or leave a gap")
        if a["parent"]["turn_char_end"] > b["parent"]["turn_char_start"]:
            problems.append("text pieces overlap")
    want = sum(1 for s in item["spans"])
    got = {(c["clip_id"], s["label"], s["group"]) for c in clips for s in c["entity_spans"]}
    if want and not got:
        problems.append("entity spans lost")
    for s in item["spans"]:
        pieces = [c for c in clips if any(sp["label"] == s["label"] and sp["group"] == s["group"] for sp in c["entity_spans"])]
        if not pieces:
            problems.append(f"entity span {s['label']} not present in any piece")
    return problems


def segment_all(items, alignments):
    """({turn key: result}, clips, unresolved) for every long turn."""
    results, clips, unresolved = {}, [], []
    for item in items:
        key = f"{item['call_id']}:{item['turn_id']}"
        alignment = alignments.get(key)
        if alignment is None:
            plan = {"status": "unresolved", "reason": "no_alignment_output"}
        elif alignment.get("error"):
            plan = {"status": "unresolved", "reason": f"aligner_error: {alignment['error']}"}
        else:
            plan = plan_turn(item, alignment)
        if plan["status"] == "resolved":
            made = pieces_to_clips(item, plan)
            problems = verify_turn(item, made)
            if problems:
                plan = {"status": "unresolved", "reason": "verification: " + "; ".join(problems[:3])}
            else:
                clips += made
        if plan["status"] != "resolved":
            unresolved.append({"call_id": item["call_id"], "turn_id": item["turn_id"], "split": item["split"], "language": item["language"], "seconds": round(item["end"] - item["start"], 2),
                               "entity_spans": len(item["spans"]), "reason": plan["reason"]})
        results[key] = {k: v for k, v in plan.items() if k != "pieces"} | {"n_pieces": len(plan.get("pieces", []))}
    return results, clips, unresolved


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("inputs", help="write the aligner input (long turns of the development calls)")
    a.add_argument("--data", type=Path, default=D.DEFAULT_OUT, help="the frozen v1 package (its split.json is reused unchanged)")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("segment", help="turn aligner output into verified <=30 s segments")
    b.add_argument("--inputs", type=Path, required=True)
    b.add_argument("--alignments", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True, help="folder for long_turn_segments.jsonl, long_turn_report.json")
    args = p.parse_args(argv)
    if args.command == "inputs":
        split = json.loads((args.data / "split.json").read_text(encoding="utf-8"))["assignment"]
        calls = D.load_calls()
        items = long_turn_inputs(calls, split)
        args.out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"turns": len(items), "train": sum(i["split"] == "TRAIN" for i in items), "validation": sum(i["split"] == "VALIDATION" for i in items)}))
        return 0
    items = json.loads(args.inputs.read_text(encoding="utf-8"))
    alignments = json.loads(args.alignments.read_text(encoding="utf-8"))
    results, clips, unresolved = segment_all(items, alignments)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "long_turn_segments.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clips), encoding="utf-8")
    report = {"turns": len(items), "resolved": sum(r["status"] == "resolved" for r in results.values()), "unresolved": unresolved, "clips": len(clips), "results": results}
    (args.out / "long_turn_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("turns", "resolved", "clips")} | {"unresolved": len(unresolved)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
