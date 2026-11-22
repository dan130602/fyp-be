"""SFT data preparation for MERaLiON-3-3B-ASR: development-only call split, turn-aware clips, entity spans and sampling plans.

Everything here is CPU-only and reads the DEVELOPMENT dataset folder only (`datasets/advisory_execution_multilingual_v1_development`; its manifest
lists the 175 development call ids). The 75 held-out calls are never listed, opened or counted: the call-spec features used for the split are read
for the 175 development ids one file at a time (the repository's `load_features` would read every spec, so it is not used).

Outputs (reference transcripts!) go to a git-ignored run folder, never into stt-experiments (AGENTS.md: no copied reference transcripts there).

Frozen rules (see PREP_REPORT.md written next to the manifests):
  clips      consecutive COMPLETE turns, chronological, packed greedily while (end of last turn - start of first turn) <= 30 s (pauses included);
             a turn longer than 30 s cannot be a clip and is EXCLUDED from training (never split proportionally); full-call validation is unaffected
  target     reference turn texts joined by one space (no space between two CJK characters), no speaker labels; the model's own output form is
             <text> <end_of_turn> "\\n" <eos> (verified against stored generations), so the supervised target is text tokens + [107, 108, 1]
  entities   exactly the scorer's spans (`transcribe_eval.reference.selected_spans`: value mentions + critical events + linked events + reviewed terms)
  important  the ORIGINAL (broader) targeted-sampling tag, `important_clip` below (entity spans OR compliance-evidence turns); kept in the package for reproducibility of the first E03. The corrected targeted rule is the ENTITY pool: clips with entity spans (sft_train --target-pool entity)
"""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FYP_BE = HERE.parents[1]
DEV_ROOT = FYP_BE / "evaluation-dataset" / "datasets" / "advisory_execution_multilingual_v1_development"
SPEC_ROOT = FYP_BE / "evaluation-dataset" / "datasets" / "advisory_execution_multilingual_v1" / "authoring" / "call_specs"
ENGINE_ROOT = FYP_BE / "evaluation-engine"
DEFAULT_OUT = ENGINE_ROOT / "runs" / "sft-data-v1"
MODEL_REVISION = "9f226d738dee62c60a2f43b0ab7b9a2585a132c3"

SAMPLE_RATE = 16000
MAX_CLIP_SECONDS = 30.0
MAX_CLIP_SAMPLES = 30 * SAMPLE_RATE          # the processor makes exactly one 30 s chunk (300 speech tokens) up to this length
END_OF_TURN_ID, NEWLINE_ID, EOS_ID = 107, 108, 1
TERMINATOR_IDS = (END_OF_TURN_ID, NEWLINE_ID, EOS_ID)
SPLIT_SEED = "sft-split-v1"
VALIDATION_FRACTION = 0.2
NEAR_DUPLICATE_JACCARD = 0.5                  # word 4-gram shingles; pairs at or above this stay on one side of the split
LANGUAGES = ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")

ASR_CONTENT = ("Instruction: Please transcribe this speech. \n"
               "Follow the text instruction based on the following audio: <SpeechHere>")   # meralion_3_asr.prompts.ASR_CONTENT, verified


def _engine():
    if str(ENGINE_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINE_ROOT))
    from transcribe_eval import reference
    return reference


def is_cjk(ch):
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF
            or 0x2026 == code)


def join_turn_texts(texts):
    """(clip text, [offset of each turn's text inside it]): one space between turns unless a CJK character sits on BOTH sides of the join
    (the model writes Chinese-to-Chinese without a space and Latin/Chinese with one)."""
    out, offsets = "", []
    for text in texts:
        if out and not (is_cjk(out[-1]) and is_cjk(text[:1])):
            out += " "
        offsets.append(len(out))
        out += text
    return out, offsets


# ---------------------------------------------------------------- calls
def development_ids(dev_root=DEV_ROOT):
    manifest = json.loads((Path(dev_root) / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_id"] != "advisory_execution_multilingual_v1_development":
        raise ValueError("not the development dataset")
    ids = sorted(manifest["call_ids"])
    if len(ids) != 175:
        raise ValueError(f"expected 175 development calls, found {len(ids)}")
    return ids


def load_call(call_id, dev_root=DEV_ROOT, terms=()):
    """One development call: turns, audio-derived turn times, entity spans, compliance evidence turns, language, render."""
    reference = _engine()
    root = Path(dev_root)
    gt = json.loads((root / "calls" / call_id / "ground_truth.json").read_text(encoding="utf-8"))
    turns = reference.load_turns(root, call_id)
    times = reference.load_turn_times(root, call_id)
    if gt["status"]["timings"] != "AUDIO_DERIVED" or sorted(times) != [t["turn_id"] for t in turns]:
        raise ValueError(f"{call_id}: turn times are not audio-derived for every turn")
    spans = reference.selected_spans(root, call_id, turns, terms)
    if spans is None:
        raise ValueError(f"{call_id}: no entity annotations")
    evidence = sorted({t for c in gt["checks"] if c["kind"] == "COMPLIANCE" for t in c["gold_evidence_turns"]})
    render = gt["timing"]["render_id"]
    return {
        "call_id": call_id, "language": gt["metadata"]["language_profile"], "duration_profile": gt["metadata"]["duration_profile"],
        "render_id": render, "audio_seconds": float(gt["timing"]["duration_seconds"]),
        "turns": [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"],
                   "start": times[t["turn_id"]][0], "end": times[t["turn_id"]][1]} for t in turns],
        "spans": spans, "evidence_turns": evidence,
    }


def load_calls(ids=None, dev_root=DEV_ROOT, terms=()):
    ids = ids or development_ids(dev_root)
    return {cid: load_call(cid, dev_root, terms) for cid in ids}


# ---------------------------------------------------------------- split
def _spec_features(call_id):
    """The stratification features of fyp_evaluation.splits.features.load_features, for ONE development call."""
    sys.path.insert(0, str(FYP_BE / "evaluation-dataset" / "src"))
    from fyp_evaluation.call_specs.compile import parse_json_text
    from fyp_evaluation.splits.features import COMPLIANCE, SECONDARY
    spec = parse_json_text((SPEC_ROOT / f"{call_id}.json").read_text(encoding="utf-8"))
    meta, gold, edge = spec["metadata"], {g["question_id"]: g["expected"] for g in spec["gold"]}, spec["edge_cases"]
    return {"call_id": call_id, "language": meta["language_profile"], "duration": meta["duration_profile"], "mixing_style": meta["mixing_style"],
            "edge_case_ids": sorted({e["edge_case_id"] for e in edge}), "families": sorted({e["family"] for e in edge}),
            "variants": sorted({f"{e['edge_case_id']}|{e['language_variant']}" for e in edge if e["language_variant"] != "NONE"}),
            "failing": sorted(q for q in COMPLIANCE if gold[q]["expected_verdict"] == "FAIL"),
            "secondary": {SECONDARY[q]: str(gold[q]["expected_value"]) for q in SECONDARY}, "assignments": len(edge)}


def _shingles(text, n=4):
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def near_duplicate_groups(calls, threshold=NEAR_DUPLICATE_JACCARD):
    """Connected groups of same-language development calls whose dialogue word-4-gram Jaccard is >= threshold, plus the sorted top pairs."""
    ids = sorted(calls)
    shingles = {c: _shingles(" ".join(t["text"] for t in calls[c]["turns"])) for c in ids}
    parent = {c: c for c in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if calls[a]["language"] != calls[b]["language"]:
                continue
            union = len(shingles[a] | shingles[b])
            score = len(shingles[a] & shingles[b]) / union if union else 0.0
            pairs.append((score, a, b))
            if score >= threshold:
                parent[find(a)] = find(b)
    groups = {}
    for c in ids:
        groups.setdefault(find(c), []).append(c)
    return [sorted(g) for g in groups.values() if len(g) > 1], sorted(pairs, reverse=True)[:10]


def freeze_split(calls, seed=SPLIT_SEED):
    """{call_id: 'TRAIN'|'VALIDATION'} using the repository's own constrained allocator on the development calls only (validation = its 'held-out' side).
    The allocator's 30% constant is replaced by 20% for the duration of the call; quotas are exact per language."""
    sys.path.insert(0, str(FYP_BE / "evaluation-dataset" / "src"))
    from fyp_evaluation.splits import allocate as allocator
    features = {c: _spec_features(c) for c in sorted(calls)}
    groups, top_pairs = near_duplicate_groups(calls)
    quotas = {lang: round(VALIDATION_FRACTION * sum(1 for f in features.values() if f["language"] == lang)) for lang in LANGUAGES}
    previous = allocator.HELDOUT_FRACTION
    allocator.HELDOUT_FRACTION = VALIDATION_FRACTION
    try:
        assignment, meta = allocator.allocate(features, [], quotas, seed=seed, groups=groups, long_target=None)
    finally:
        allocator.HELDOUT_FRACTION = previous
    errors = allocator.validate_assignment(features, assignment, [], quotas, groups)
    if errors:
        raise ValueError(f"split validation failed: {errors}")
    split = {c: ("VALIDATION" if v == "HELDOUT" else "TRAIN") for c, v in assignment.items()}
    return split, {"seed": seed, "quotas": quotas, "swaps": meta["swaps"], "objective": meta["objective"], "near_duplicate_groups": groups,
                   "top_similarity_pairs": [(round(s, 3), a, b) for s, a, b in top_pairs], "features": features}


# ---------------------------------------------------------------- clips
def build_clips(call, split):
    """Greedy chronological packing of complete turns into <= 30 s clips; returns (clips, excluded long turns)."""
    clips, excluded, current = [], [], []

    def flush():
        if current:
            clips.append(make_clip(call, split, list(current)))
            current.clear()
    for turn in call["turns"]:
        duration = turn["end"] - turn["start"]
        if round(turn["end"] * SAMPLE_RATE) - round(turn["start"] * SAMPLE_RATE) > MAX_CLIP_SAMPLES:
            flush()
            excluded.append(turn)
            continue
        if current and round(turn["end"] * SAMPLE_RATE) - round(current[0]["start"] * SAMPLE_RATE) > MAX_CLIP_SAMPLES:
            flush()
        current.append(turn)
    flush()
    return clips, excluded


def make_clip(call, split, turns):
    text, offsets = join_turn_texts([t["text"] for t in turns])
    by_turn = {t["turn_id"]: offsets[i] for i, t in enumerate(turns)}
    spans = []
    for span in call["spans"]:
        if span["turn_id"] in by_turn:
            base = by_turn[span["turn_id"]]
            spans.append({"start": base + span["start"], "end": base + span["end"], "group": span["group"], "label": span["label"]})
    spans.sort(key=lambda s: (s["start"], s["end"], s["group"]))
    for span in spans:
        if not text[span["start"]:span["end"]].strip():
            raise ValueError(f"{call['call_id']}: empty entity span")
    start, end = turns[0]["start"], turns[-1]["end"]
    return {
        "clip_id": f"{call['call_id']}:{turns[0]['turn_id']}-{turns[-1]['turn_id']}", "call_id": call["call_id"], "split": split,
        "language": call["language"], "turn_ids": [t["turn_id"] for t in turns], "speakers": [t["speaker"] for t in turns],
        "start_seconds": start, "end_seconds": end, "duration_seconds": round(end - start, 3),
        "start_sample": round(start * SAMPLE_RATE), "end_sample": round(end * SAMPLE_RATE),
        "text": text, "entity_spans": spans, "evidence_turns": sorted(set(t["turn_id"] for t in turns) & set(call["evidence_turns"])),
    }


def important_clip(clip):
    """FROZEN targeted-sampling rule (broader than the entity masks): the clip contains any scored value mention or critical-event span, OR a turn that is
    gold evidence of a COMPLIANCE check.  Reported with its coverage in PREP_REPORT.md; the entity masks never read this."""
    return bool(clip["entity_spans"]) or bool(clip["evidence_turns"])


# ---------------------------------------------------------------- tokenisation (tokenizer passed in; fast tokenizer with offsets)
def tokenise_clip(tokenizer, clip):
    """(target ids incl. terminators, entity flag per target id, per-id group list). Entity = the token's character range overlaps an entity span."""
    encoding = tokenizer(clip["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoding["input_ids"]), encoding["offset_mapping"]
    flags, groups = [], []
    for (a, b) in offsets:
        hit = [s["group"] for s in clip["entity_spans"] if a < s["end"] and s["start"] < b and b > a]
        flags.append(bool(hit))
        groups.append(sorted(set(hit)))
    ids += list(TERMINATOR_IDS)
    flags += [False] * len(TERMINATOR_IDS)
    groups += [[] for _ in TERMINATOR_IDS]
    return ids, flags, groups


def tokens_fingerprint(token_lists):
    digest = hashlib.sha256()
    for ids in token_lists:
        digest.update((",".join(map(str, ids)) + "\n").encode())
    return digest.hexdigest()


# ---------------------------------------------------------------- sampling plan (matched budgets)
def plan_epochs(train_clip_ids, target_ids, k, epochs, seed, mode):
    """Per-epoch draw lists. Every unique clip once per epoch + K extra draws: `uniform` = K clips without replacement from ALL clips,
    `targeted` = K clips drawn from the TARGET pool (`target_ids`: for the entity rule, the clips that contain entity spans of the eval engine's entity scoring); then shuffled with a per-epoch seed.
    Draws are without replacement within a round. When K does not exceed the pool this is a single round (exactly the original plan, so earlier runs' plans are reproduced);
    when K exceeds the pool, further rounds each draw without replacement again, so no clip gets a second extra copy before every pool clip has had one. Pure function of its arguments."""
    ids = sorted(train_clip_ids)
    target = sorted(target_ids)
    pool = {"uniform": ids, "targeted": target}[mode]
    if k < 0 or (k > 0 and not pool):
        raise ValueError(f"K={k} cannot be drawn from an empty or invalid {mode} pool of {len(pool)} clips")
    plans = []
    for epoch in range(epochs):
        rng = random.Random(f"{seed}:{mode}:{epoch}")
        extra, left = [], k
        while left > 0:
            take = min(left, len(pool))
            extra += rng.sample(pool, take)
            left -= take
        draws = ids + extra
        random.Random(f"{seed}:shuffle:{mode}:{epoch}").shuffle(draws)
        plans.append(draws)
    return plans


def group_boundaries(plan_length, accumulation):
    """Sizes of the accumulation groups of one epoch: full groups then a flushed partial one."""
    full, rest = divmod(plan_length, accumulation)
    return [accumulation] * full + ([rest] if rest else [])
