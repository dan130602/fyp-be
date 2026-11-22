"""Build the frozen SFT data package (CPU only, development calls only) and its audit report.

    python -X utf8 prepare_sft_data.py --tokenizer <model dir> [--out <folder>] [--k <extra draws>] [--force]

Writes into --out (default evaluation-engine/runs/sft-data-v1, git-ignored: it holds reference transcripts):
  split.json                 call -> TRAIN|VALIDATION, quotas, features used, near-duplicate audit
  clips_train.jsonl / clips_validation.jsonl      one clip per line (text, audio sample bounds, entity character spans, evidence turns, important flag)
  excluded_long_turns.json   every turn longer than 30 s that cannot be a clip (ids, duration, entity spans lost); never split proportionally
  validation_calls.json      the 35 full-call validation ids (full-call generation keeps ALL their reference content)
  manifest.json              hashes of the above, token fingerprint of every training/validation target, important-clip rule and counts, K table
  PREP_REPORT.md             aggregate numbers only (no reference text)
"""
import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

import sft_data as D


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call_voices(call):
    root = D.DEV_ROOT / "calls" / call["call_id"] / "renders" / call["render_id"]
    try:
        if (root / "generation.json").is_file():
            turns = json.loads((root / "generation.json").read_text(encoding="utf-8"))["turns"]
            out = {}
            for t in turns.values():
                out.setdefault(t["speaker"], set()).add(t["voice"])
            return {k: sorted(v) for k, v in out.items()}
        settings = json.loads((root / "render_manifest.json").read_text(encoding="utf-8"))["settings"]
        return {"EMPLOYEE": [settings["employee_voice"]], "CLIENT": [settings["client_voice"]]}
    except (OSError, KeyError, ValueError):
        return {}


def quantiles(values, points=(0.05, 0.25, 0.5, 0.75, 0.95)):
    ordered = sorted(values)
    return {f"p{int(p * 100)}": round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 2) for p in points}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, required=True, help="the pinned model directory (tokenizer files only are read)")
    parser.add_argument("--out", type=Path, default=D.DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=None, help="extra draws per epoch to record in manifest.json (default: not frozen; only a table is written)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split-from", type=Path, help="reuse the frozen call assignment of this existing package (split.json) instead of recomputing it (v2 packages)")
    parser.add_argument("--long-segments", type=Path, help="folder with long_turn_segments.jsonl + long_turn_report.json from sft_align.py: aligned segments replace the excluded long turns")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        sys.exit(f"{args.out} is not empty: refusing to overwrite a frozen data package (use a new --out, or --force deliberately)")
    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))

    calls = D.load_calls()
    if args.split_from:
        frozen = json.loads((args.split_from / "split.json").read_text(encoding="utf-8"))
        split = frozen["assignment"]
        if sorted(split) != sorted(calls):
            sys.exit("the frozen split does not cover exactly the 175 development calls")
        meta = {k: frozen[k] for k in ("seed", "quotas", "swaps", "objective", "near_duplicate_groups", "top_similarity_pairs")}
    else:
        split, meta = D.freeze_split(calls)
    recovered = {"TRAIN": 0, "VALIDATION": 0}
    unresolved = []
    if args.long_segments:
        report = json.loads((args.long_segments / "long_turn_report.json").read_text(encoding="utf-8"))
        unresolved = report["unresolved"]
        segment_clips = [json.loads(l) for l in (args.long_segments / "long_turn_segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for c in segment_clips:
            if split[c["call_id"]] != c["split"]:
                sys.exit(f"{c['clip_id']}: split differs from the frozen assignment")
        recovered_keys = {(c["parent"]["call_id"], c["parent"]["turn_id"]) for c in segment_clips}
    clips, excluded = {"TRAIN": [], "VALIDATION": []}, {"TRAIN": [], "VALIDATION": []}
    for call_id, call in sorted(calls.items()):
        made, long_turns = D.build_clips(call, split[call_id])
        clips[split[call_id]] += made
        long_turns = [t for t in long_turns if not args.long_segments or (call_id, t["turn_id"]) not in recovered_keys]
        excluded[split[call_id]] += [{"call_id": call_id, "turn_id": t["turn_id"], "speaker": t["speaker"], "language": call["language"],
                                      "start": t["start"], "end": t["end"], "duration_seconds": round(t["end"] - t["start"], 3),
                                      "entity_spans_lost": sum(1 for s in call["spans"] if s["turn_id"] == t["turn_id"]),
                                      "characters": len(t["text"])} for t in long_turns]
    if args.long_segments:
        for c in segment_clips:
            clips[c["split"]].append(c)
            recovered[c["split"]] += 1 if c["parent"]["piece"] == 1 else 0
        for sp in clips:
            clips[sp].sort(key=lambda c: (c["call_id"], c["start_sample"]))
    # audio/reference alignment checks that need no audio: ordered, non-overlapping, inside the recorded duration, every clip within one 30 s chunk
    problems = []
    for call_id, call in calls.items():
        last = 0.0
        for t in call["turns"]:
            if t["start"] < last - 1e-6 or t["end"] <= t["start"] or t["end"] > call["audio_seconds"] + 0.05:
                problems.append(f"{call_id} turn {t['turn_id']}: times out of order or beyond the recording")
            last = t["end"]
    for sp in clips:
        for c in clips[sp]:
            if c["end_sample"] - c["start_sample"] > D.MAX_CLIP_SAMPLES:
                problems.append(f"{c['clip_id']}: longer than 30 s")
    ids = [c["clip_id"] for sp in clips for c in clips[sp]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate clip ids")
    spans_seen = {}
    for sp in clips:
        for c in clips[sp]:
            if split[c["call_id"]] != sp or c["split"] != sp:
                problems.append(f"{c['clip_id']}: not in its assigned split")
            spans_seen.setdefault(c["call_id"], []).append((c["start_sample"], c["end_sample"], c["clip_id"]))
    for call_id, rows in spans_seen.items():
        rows.sort()
        for (a0, a1, aid), (b0, b1, bid) in zip(rows, rows[1:]):
            if b0 < a1:
                problems.append(f"{aid} and {bid} overlap in audio")
    if problems:
        sys.exit("alignment checks failed:\n" + "\n".join(problems[:20]))

    important = {c["clip_id"] for c in clips["TRAIN"] if D.important_clip(c)}
    for sp in clips:
        for c in clips[sp]:
            c["important"] = D.important_clip(c)

    token_stats = {}
    fingerprints = {}
    for sp in clips:
        n_tok = n_ent = zero_overlap = widened = 0
        target_lengths = []
        lists = []
        for c in clips[sp]:
            ids, flags, _ = D.tokenise_clip(tokenizer, c)
            lists.append(ids)
            n_tok += len(ids)
            n_ent += sum(flags)
            target_lengths.append(len(ids))
            enc = tokenizer(c["text"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for s in c["entity_spans"]:
                hit = [(a, b) for a, b in enc if a < s["end"] and s["start"] < b and b > a]
                if not hit:
                    zero_overlap += 1
                elif min(a for a, _ in hit) < s["start"] - 1 or max(b for _, b in hit) > s["end"] + 1:
                    widened += 1
        fingerprints[sp] = D.tokens_fingerprint(lists)
        token_stats[sp] = {"target_tokens_incl_terminators": n_tok, "entity_tokens": n_ent, "entity_token_share": round(n_ent / n_tok, 4),
                           "max_target_tokens": max(target_lengths), "spans_with_no_token": zero_overlap, "spans_widened_by_token_boundaries": widened}
        if zero_overlap:
            sys.exit(f"{zero_overlap} entity spans overlap no token in {sp}: stop and inspect")

    def write_jsonl(name, rows):
        path = args.out / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        return path

    def write_json(name, obj):
        path = args.out / name
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return path

    voices = {c: call_voices(calls[c]) for c in calls}
    voice_overlap = {}
    for role in ("EMPLOYEE", "CLIENT"):
        tr = {v for c in calls if split[c] == "TRAIN" for v in voices[c].get(role, [])}
        va = {v for c in calls if split[c] == "VALIDATION" for v in voices[c].get(role, [])}
        voice_overlap[role] = {"train_voices": len(tr), "validation_voices": len(va), "shared": len(tr & va), "validation_only": sorted(va - tr)}
    split_doc = {"seed": meta["seed"], "rule": "repo allocator (fyp_evaluation.splits.allocate) on the 175 development calls, validation fraction 0.2, exact per-language quotas, no locked calls",
                 "quotas": meta["quotas"], "swaps": meta["swaps"], "objective": meta["objective"], "assignment": split, "near_duplicate_groups": meta["near_duplicate_groups"],
                 "top_similarity_pairs": meta["top_similarity_pairs"], "voice_overlap": voice_overlap,
                 "per_call": {c: {"language": calls[c]["language"], "duration_profile": calls[c]["duration_profile"], "render_id": calls[c]["render_id"], "voices": voices[c]} for c in sorted(calls)}}
    split_path = write_json("split.json", split_doc)
    train_path = write_jsonl("clips_train.jsonl", clips["TRAIN"])
    val_path = write_jsonl("clips_validation.jsonl", clips["VALIDATION"])
    excl_path = write_json("excluded_long_turns.json", excluded)
    validation_ids = sorted(c for c in calls if split[c] == "VALIDATION")
    vc_path = write_json("validation_calls.json", {"call_ids": validation_ids, "note": "development validation: previously used for inference optimisation and error analysis; NOT an unseen test set"})

    # K table (accumulation 8; the final partial group of each epoch is flushed)
    n, m = len(clips["TRAIN"]), len(important)
    k_table = []
    for k in sorted({0, m // 4, m // 2, m}):
        draws = n + k
        steps = len(D.group_boundaries(draws, 8))
        k_table.append({"K": k, "draws_per_epoch": draws, "optimiser_steps_per_epoch": steps, "draws_3_epochs": 3 * draws, "optimiser_steps_3_epochs": 3 * steps})
    manifest = {"schema": f"sft-data-{args.version}", "model_revision": D.MODEL_REVISION, "files": {p.name: sha256_file(p) for p in (split_path, train_path, val_path, excl_path, vc_path)},
                "token_fingerprints": fingerprints, "tokenizer_files_sha256": {n_: sha256_file(args.tokenizer / n_) for n_ in ("tokenizer.json", "tokenizer_config.json")},
                "derived_from": str(args.split_from) if args.split_from else None, "long_turns": {"recovered_turns": recovered, "unresolved": unresolved, "complete": not unresolved},
                "counts": {"train_calls": 140, "validation_calls": 35, "train_clips": n, "validation_clips": len(clips["VALIDATION"]), "important_train_clips": m},
                "important_rule": D.important_clip.__doc__.strip(), "k_table": k_table, "frozen_k": args.k,
                "max_clip_samples": D.MAX_CLIP_SAMPLES, "terminator_ids": list(D.TERMINATOR_IDS), "prompt_content": D.ASR_CONTENT}
    if args.k is not None:
        if not 0 <= args.k <= m:
            sys.exit(f"K must be in 0..{m}")
    write_json("manifest.json", manifest)

    # ---- report (aggregates only)
    lines = [f"# SFT data package {args.version} (development calls only)", "", f"Model revision `{D.MODEL_REVISION}`; split seed `{meta['seed']}`.", ""]
    lines += ["## Calls", "", "| language | train calls | validation calls | render |", "|---|---:|---:|---|"]
    for lang in D.LANGUAGES:
        tr = sum(1 for c in calls if calls[c]["language"] == lang and split[c] == "TRAIN")
        va = sum(1 for c in calls if calls[c]["language"] == lang and split[c] == "VALIDATION")
        renders = sorted({calls[c]["render_id"] for c in calls if calls[c]["language"] == lang})
        lines.append(f"| {lang} | {tr} | {va} | {', '.join(renders)} |")
    lines += ["", f"Render provider is confounded with language (English = Polly, Singlish = OmniVoice, Mandarin/mixed = Gemini TTS): no split can separate them.",
              f"Voices per role (every validation voice also appears in training: voice overlap is unavoidable with this few voices): `{json.dumps(voice_overlap, ensure_ascii=False)}`", ""
              f"Near-duplicate audit (word 4-gram Jaccard, same-language pairs, threshold {D.NEAR_DUPLICATE_JACCARD}): groups = {meta['near_duplicate_groups']}; top pairs {meta['top_similarity_pairs'][:3]}.",
              "Call-level split only: shared product names, currencies, financial terms and TTS voices are NOT leakage and remain on both sides.", ""]
    lines += ["## Clips", "", "| | train | validation |", "|---|---:|---:|"]
    stats = {}
    for sp in ("TRAIN", "VALIDATION"):
        cs = clips[sp]
        d = [c["duration_seconds"] for c in cs]
        stats[sp] = {"clips": len(cs), "hours": round(sum(d) / 3600, 2), "mean": round(statistics.mean(d), 1), "q": quantiles(d), "multi_turn": sum(len(c["turn_ids"]) > 1 for c in cs),
                     "single_turn": sum(len(c["turn_ids"]) == 1 for c in cs), "multi_speaker": sum(len(set(c["speakers"])) > 1 for c in cs),
                     "languages": dict(collections.Counter(c["language"] for c in cs)), "spans": sum(len(c["entity_spans"]) for c in cs),
                     "entity_clips": sum(bool(c["entity_spans"]) for c in cs), "evidence_clips": sum(bool(c["evidence_turns"]) for c in cs), "important": sum(c["important"] for c in cs)}
    for key in ("clips", "hours", "mean", "multi_turn", "single_turn", "multi_speaker", "spans", "entity_clips", "evidence_clips", "important"):
        lines.append(f"| {key} | {stats['TRAIN'][key]} | {stats['VALIDATION'][key]} |")
    lines += ["", f"Duration quantiles (s): train {stats['TRAIN']['q']}; validation {stats['VALIDATION']['q']}.",
              f"Clips by language: train {stats['TRAIN']['languages']}; validation {stats['VALIDATION']['languages']}.", ""]
    lines += ["## Entity masks (exact scorer spans; broader tags are separate)", "",
              f"Tokens: {json.dumps(token_stats)}", "",
              "Entity span groups in TRAIN clips: " + json.dumps(dict(collections.Counter(s["group"] for c in clips["TRAIN"] for s in c["entity_spans"]).most_common()), ensure_ascii=False), ""]
    lines += [("## Long turns (> 30 s) NOT recovered (blockers); full-call validation keeps all content" if args.long_segments else "## Long turns (> 30 s): not clips, EXCLUDED from training and from teacher-forced validation; full-call validation keeps them"), ""]
    for sp in ("TRAIN", "VALIDATION"):
        ex = excluded[sp]
        total_spans = sum(len(calls[c]["spans"]) for c in calls if split[c] == sp)
        lines.append(f"- {sp}: {len(ex)} turns ({sum(1 for e in ex if e['speaker'] == 'EMPLOYEE')} employee, {sum(1 for e in ex if e['speaker'] == 'CLIENT')} client), "
                     f"{round(sum(e['duration_seconds'] for e in ex) / 3600, 2)} h, {sum(e['entity_spans_lost'] for e in ex)} of {total_spans} entity spans, "
                     f"by language {dict(collections.Counter(e['language'] for e in ex))}, per-call top {collections.Counter(e['call_id'] for e in ex).most_common(3)}")
    if args.long_segments:
        res = report["results"]
        kinds = collections.Counter(c["kind"] for r in res.values() for c in r.get("cuts", []))
        ratios = [c["rms_ratio"] for r in res.values() for c in r.get("cuts", []) if c.get("rms_ratio") is not None]
        seg = collections.defaultdict(lambda: {"turns": set(), "clips": 0, "hours": 0.0, "spans": 0})
        for c in segment_clips:
            e = seg[c["split"]]
            e["turns"].add((c["call_id"], c["parent"]["turn_id"]))
            e["clips"] += 1
            e["hours"] += c["duration_seconds"] / 3600
            e["spans"] += len(c["entity_spans"])
        lines += ["", "## Recovered long turns (forced alignment of the KNOWN reference text; segments <= 29.5 s; v1 package untouched)", ""]
        for sp in ("TRAIN", "VALIDATION"):
            e = seg[sp]
            lines.append(f"- {sp}: {len(e['turns'])} turns -> {e['clips']} clips, {e['hours']:.2f} h, {e['spans']} entity spans carried")
        lines.append(f"- cut classes: {dict(kinds)}; cut energy / typical speech energy: median {statistics.median(ratios):.3f}, max {max(ratios):.3f}" if ratios else "- no cuts")
        lines.append(f"- UNRESOLVED (blockers): {len(unresolved)} -> {unresolved if unresolved else 'none'}")
    lines += ["", "All 175 development calls: 92 long turns (32 employee, 60 client) -> matches the pre-registered count.", ""]
    lines += ["## Important-clip rule and K", "", D.important_clip.__doc__.strip(), "",
              f"N = {n} unique training clips, M = {m} important ({100 * m / n:.1f}%). Entity-bearing {stats['TRAIN']['entity_clips']}, evidence-bearing {stats['TRAIN']['evidence_clips']}.",
              "", "| K | draws/epoch | optimiser steps/epoch | draws, 3 epochs | optimiser steps, 3 epochs |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['K']} | {r['draws_per_epoch']} | {r['optimiser_steps_per_epoch']} | {r['draws_3_epochs']} | {r['optimiser_steps_3_epochs']} |" for r in k_table]
    (args.out / "PREP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: train {n} clips ({stats['TRAIN']['hours']} h), validation {len(clips['VALIDATION'])} clips ({stats['VALIDATION']['hours']} h), important {m}")


if __name__ == "__main__":
    main()
