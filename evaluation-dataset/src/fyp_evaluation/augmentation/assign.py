"""Deterministic condition + severity assignment for environment_v2.

Inputs are ONLY the frozen split (splits/split.v1.json) and each call G9 metadata (language, SHORT/LONG, duration, pinned render).
The result never depends on levels, noise assets or config status, so freezing the config cannot change an assignment.
Randomness uses random.Random.random() only (the one method Python documents as reproducible across versions).
"""

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from . import ALGORITHM, CLEAN, CONDITION_ORDER, CONDITIONS, SEED, VARIANT_ID, AugmentationError

SPLITS = ("DEVELOPMENT", "HELDOUT")
LANGUAGES = ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")
# Calls per condition within each frozen split, as proposed by the owner.
TARGETS = {"DEVELOPMENT": {"C1": 29, "N1": 29, "N2": 29, "N3": 29, "D1": 30, "D2": 29},
           "HELDOUT": {"C1": 13, "N1": 13, "N2": 13, "N3": 12, "D1": 12, "D2": 12}}
CELLS = [CLEAN] + [c if not CONDITIONS[c][1] else c + ":" + s for c in CONDITION_ORDER[1:] for s in (CONDITIONS[c][1] or ("",))]
PENALTY_ITERATIONS, RESTARTS = 60000, 10


def split_cell(cell):
    condition, _, severity = cell.partition(":")
    return condition, severity or None


def _below(rng, n):
    return int(rng.random() * n)


def client_voices(dataset_root):
    """{call_id: provider:voice} for the CLIENT of every call, from the committed G8 voice assignment files."""
    voices = {}
    for provider in ("polly", "gemini", "omnivoice"):
        document = json.loads((Path(dataset_root) / "authoring" / (provider + "_voice_assignments.json")).read_text(encoding="utf-8"))
        for call_id, entry in document["assignments"].items():
            voices[call_id] = provider + ":" + entry["client_voice"]
    return voices


def load_inputs(dataset_root):
    root = Path(dataset_root)
    voices = client_voices(root)
    split_bytes = (root / "splits" / "split.v1.json").read_bytes()
    assignments = json.loads(split_bytes)["assignments"]
    calls = []
    for call_id, entry in sorted(assignments.items()):
        truth = json.loads((root / "calls" / call_id / "ground_truth.json").read_text(encoding="utf-8"))
        render_id = truth["timing"]["render_id"]
        folder = root / "calls" / call_id / "renders" / render_id
        audio_file = json.loads((folder / "timings.json").read_text(encoding="utf-8"))["audio_file"]
        if truth["metadata"]["language_profile"] != entry["language"]:
            raise AugmentationError(f"{call_id}: split language and ground-truth language disagree")
        calls.append({"call_id": call_id, "split": entry["split"], "language": entry["language"],
                      "duration_profile": truth["metadata"]["duration_profile"], "seconds": truth["timing"]["duration_seconds"],
                      "render_id": render_id, "audio_file": audio_file, "audio_sha256": truth["timing"]["audio_sha256"],
                      "voice": voices[call_id]})
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    return {"dataset_id": manifest["dataset_id"], "split_sha256": hashlib.sha256(split_bytes).hexdigest(), "calls": calls}


def _tables(by):
    """Aggregate the (split, language, cell, duration) counts into the tables the objective and the validator use."""
    cond, sev, long_cond, long_sev = Counter(), Counter(), Counter(), Counter()
    lang_cond, lang_sev, lang_long = Counter(), Counter(), Counter()
    for (split, lang, cell, dur), n in by.items():
        if not n:
            continue
        c, s = split_cell(cell)
        cond[(split, c)] += n
        sev[(split, c, s)] += n
        lang_cond[(split, lang, c)] += n
        lang_sev[(split, lang, c, s)] += n
        if dur == "LONG":
            long_cond[(split, c)] += n
            long_sev[(split, c, s)] += n
            lang_long[(split, lang, c)] += n
    return cond, sev, long_cond, long_sev, lang_cond, lang_sev, lang_long


def objective(by, minutes, sizes, voice_cells=None, voice_sizes=None):
    cond, sev, long_cond, long_sev, lang_cond, lang_sev, lang_long = _tables(by)
    score = 0.0
    overall = Counter()
    for split in SPLITS:
        n_long = sum(long_cond[(split, c)] for c in CONDITION_ORDER)
        mean_minutes = sum(minutes[(split, c)] for c in CONDITION_ORDER) / len(CONDITION_ORDER)
        for c in CONDITION_ORDER:
            score += 1000 * (cond[(split, c)] - TARGETS[split][c]) ** 2
            score += 200 * (long_cond[(split, c)] - n_long / 6) ** 2
            score += (minutes[(split, c)] - mean_minutes) ** 2
            if CONDITIONS[c][1]:
                first, second = CONDITIONS[c][1]
                score += 400 * max(0, abs(sev[(split, c, first)] - sev[(split, c, second)]) - 1) ** 2
                score += 60 * max(0, abs(long_sev[(split, c, first)] - long_sev[(split, c, second)]) - 1) ** 2
                overall[(c, first)] += sev[(split, c, first)]
                overall[(c, second)] += sev[(split, c, second)]
            for lang in LANGUAGES:
                score += 2500 * max(0, abs(lang_cond[(split, lang, c)] - sizes[(split, lang)] / 6) - 0.5) ** 2
                score += 50 * max(0, lang_long[(split, lang, c)] - 1) ** 2
                if CONDITIONS[c][1]:
                    score += 80 * max(0, abs(lang_sev[(split, lang, c, first)] - lang_sev[(split, lang, c, second)]) - 1) ** 2
    for c in CONDITION_ORDER[1:]:
        if not CONDITIONS[c][1]:
            continue
        first, second = CONDITIONS[c][1]
        score += 300 * max(0, abs(overall[(c, first)] - overall[(c, second)]) - 1) ** 2
    if voice_cells is not None:                       # no speaker should be piled into, or missing from, a condition
        for voice, n in voice_sizes.items():
            cap = math.ceil(n / 6)
            for c in CONDITION_ORDER:
                count = voice_cells[(voice, c)]
                score += 150 * max(0, count - cap) ** 2
                if n >= 6 and count == 0:
                    score += 60
    return score


def _initial(calls, rng):
    cells = {}
    for split in SPLITS:
        for lang in LANGUAGES:
            group = sorted((c for c in calls if c["split"] == split and c["language"] == lang),
                           key=lambda c: (c["duration_profile"] != "LONG", -c["seconds"], c["call_id"]))
            offset = _below(rng, 6)
            for i, call in enumerate(group):
                condition = CONDITION_ORDER[(i + offset) % 6]
                severities = CONDITIONS[condition][1]
                cells[call["call_id"]] = condition if not severities else f"{condition}:{severities[(i // 6) % 2]}"
    return cells


def _accept(rng, current, trial, temperature):
    return trial <= current or rng.random() < 2.718281828 ** ((current - trial) / temperature)


def allocate(calls, seed=SEED):
    for split in SPLITS:
        members = [c for c in calls if c["split"] == split]
        if sum(TARGETS[split].values()) != len(members):
            raise AugmentationError(f"{split}: targets sum to {sum(TARGETS[split].values())} but the split has {len(members)} calls")
    rng = random.Random(seed)
    sizes = Counter((c["split"], c["language"]) for c in calls)
    voice_sizes = Counter(c["voice"] for c in calls)
    best = None
    for _ in range(RESTARTS):
        cells = _initial(calls, rng)
        by, minutes, voice_cells = Counter(), Counter(), Counter()
        for c in calls:
            by[(c["split"], c["language"], cells[c["call_id"]], c["duration_profile"])] += 1
            minutes[(c["split"], split_cell(cells[c["call_id"]])[0])] += c["seconds"] / 60
            voice_cells[(c["voice"], split_cell(cells[c["call_id"]])[0])] += 1
        current = objective(by, minutes, sizes, voice_cells, voice_sizes)
        temperature = 50.0

        def move(call, new_cell):
            old = cells[call["call_id"]]
            for cell, sign in ((old, -1), (new_cell, 1)):
                by[(call["split"], call["language"], cell, call["duration_profile"])] += sign
                minutes[(call["split"], split_cell(cell)[0])] += sign * call["seconds"] / 60
                voice_cells[(call["voice"], split_cell(cell)[0])] += sign
            cells[call["call_id"]] = new_cell

        for _ in range(PENALTY_ITERATIONS):
            a = calls[_below(rng, len(calls))]
            if rng.random() < 0.15:
                old, new_cell = cells[a["call_id"]], CELLS[_below(rng, len(CELLS))]
                if new_cell == old:
                    continue
                move(a, new_cell)
                trial = objective(by, minutes, sizes, voice_cells, voice_sizes)
                if _accept(rng, current, trial, temperature):
                    current = trial
                else:
                    move(a, old)
            else:
                b = calls[_below(rng, len(calls))]
                if (a["split"], a["language"]) != (b["split"], b["language"]) or cells[a["call_id"]] == cells[b["call_id"]]:
                    continue
                cell_a, cell_b = cells[a["call_id"]], cells[b["call_id"]]
                move(a, cell_b)
                move(b, cell_a)
                trial = objective(by, minutes, sizes, voice_cells, voice_sizes)
                if _accept(rng, current, trial, temperature):
                    current = trial
                else:
                    move(a, cell_a)
                    move(b, cell_b)
            temperature = max(0.01, temperature * 0.9998)
        if best is None or current < best[0]:
            best = (current, dict(cells))
    return best[1], round(best[0], 3)


def count_tables(calls, cells):
    table = {}
    for c in calls:
        row = table.setdefault(c["split"], {}).setdefault(c["language"], {}).setdefault(cells[c["call_id"]], {"SHORT": 0, "LONG": 0})
        row[c["duration_profile"]] += 1
    return table


def validate_assignment(calls, cells):
    """Hard rules; returns a list of problems (empty = valid)."""
    if set(cells) != {c["call_id"] for c in calls}:
        return ["assignment does not cover exactly the 250 calls"]
    problems = []
    by = Counter((c["split"], c["language"], cells[c["call_id"]], c["duration_profile"]) for c in calls)
    cond, sev, _long_cond, _long_sev, lang_cond, _lang_sev, _lang_long = _tables(by)
    sizes = Counter((c["split"], c["language"]) for c in calls)
    for split in SPLITS:
        for c in CONDITION_ORDER:
            if cond[(split, c)] != TARGETS[split][c]:
                problems.append(f"{split} {c}: {cond[(split, c)]} calls, expected {TARGETS[split][c]}")
            if CONDITIONS[c][1]:
                first, second = CONDITIONS[c][1]
                if abs(sev[(split, c, first)] - sev[(split, c, second)]) > 1:
                    problems.append(f"{split} {c}: severity split {sev[(split, c, first)]}/{sev[(split, c, second)]}")
            for lang in LANGUAGES:
                if abs(lang_cond[(split, lang, c)] - sizes[(split, lang)] / 6) > 1.0:
                    problems.append(f"{split} {lang} {c}: {lang_cond[(split, lang, c)]} of {sizes[(split, lang)]} calls")
                if split == "HELDOUT" and lang_cond[(split, lang, c)] < 2:
                    problems.append(f"HELDOUT {lang} {c}: only {lang_cond[(split, lang, c)]} call(s)")
    for c in CONDITION_ORDER[1:]:
        if not CONDITIONS[c][1]:
            continue
        first, second = CONDITIONS[c][1]
        total = [sum(sev[(s, c, v)] for s in SPLITS) for v in (first, second)]
        if abs(total[0] - total[1]) > 1:
            problems.append(f"{c}: overall severity split {total}")
    return problems


def build_assignments(dataset_root, seed=SEED):
    inputs = load_inputs(dataset_root)
    calls = inputs["calls"]
    cells, score = allocate(calls, seed)
    problems = validate_assignment(calls, cells)
    if problems:
        raise AugmentationError("allocation violates hard rules: " + "; ".join(problems))
    rows = {}
    for c in calls:
        condition, severity = split_cell(cells[c["call_id"]])
        call_id = c["call_id"]
        rows[call_id] = {
            "split": c["split"], "language": c["language"], "duration_profile": c["duration_profile"], "duration_seconds": c["seconds"],
            "client_voice": c["voice"], "condition": condition, "severity": severity,
            "source": {"render_id": c["render_id"], "path": "calls/" + call_id + "/renders/" + c["render_id"] + "/" + c["audio_file"],
                       "audio_sha256": c["audio_sha256"]},
            "variant_path": None if condition == CLEAN else "calls/" + call_id + "/audio_variants/environment_v2/audio.wav"}
    return {"schema_version": "1.0", "variant_id": VARIANT_ID, "dataset_id": inputs["dataset_id"], "seed": seed, "algorithm": ALGORITHM,
            "split_sha256": inputs["split_sha256"], "objective": score, "targets": TARGETS,
            "severities": {c: list(CONDITIONS[c][1]) for c in CONDITION_ORDER},
            "counts": count_tables(calls, cells), "calls": rows}


def dump(document):
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + chr(10)).encode("utf-8")
