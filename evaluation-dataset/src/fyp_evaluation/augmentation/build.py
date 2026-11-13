"""Config, per-call augmentation, previews, freeze and verification for environment_v2 (one call at a time, offline)."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from fyp_evaluation.planning.outputs import sha256 as normalised_sha256

from . import AUDIO_ASSETS, CALL_VARIANT_DIR, CLEAN, CONDITIONS, SEED, VARIANT_ID, VARIANT_ROOT, AugmentationError
from . import assign as assign_module
from . import audio

DRAFT, FROZEN = "DRAFT", "FROZEN"
CODE_VERSION = "1.0"
SNR_TOLERANCE_DB = 0.2
AI_TOLERANCE = 0.02
PEAK_TOLERANCE = 0.9801   # 0.98 plus int16 rounding
UNKNOWN_LICENCE = "UNKNOWN - fill in before freezing"


# ----------------------------------------------------------------------------------------------------------------- paths + json
def evaluation_root(dataset_root):
    """datasets/<dataset id> -> the evaluation-dataset folder that holds audio_assets/."""
    return Path(dataset_root).resolve().parents[1]


def variant_dir(dataset_root):
    return Path(dataset_root) / VARIANT_ROOT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, document):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(assign_module.dump(document))


def plain_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------------------------------- assets + config
def scan_assets(dataset_root):
    """Describe every noise WAV under audio_assets/noise/<category>/; keeps any provenance already filled in assets.json."""
    folder = evaluation_root(dataset_root) / AUDIO_ASSETS
    existing = {a["file"]: a for a in read_json(folder / "assets.json")["assets"]} if (folder / "assets.json").is_file() else {}
    assets = []
    for category in sorted(p for p in folder.iterdir() if p.is_dir()):
        for index, path in enumerate(sorted(category.glob("*.wav")), start=1):
            file = f"{category.name}/{path.name}"
            info = sf.info(str(path))
            entry = {"id": f"{category.name}_{index:02d}", "category": category.name, "file": file, "sha256": plain_sha256(path),
                     "sample_rate": info.samplerate, "channels": info.channels, "subtype": info.subtype,
                     "duration_s": round(info.duration, 1), "source_url": None, "author": None, "license": UNKNOWN_LICENCE,
                     "obtained_on": "2026-11-12", "notes": "", "listening_ok": None}
            if file in existing:
                entry.update({k: existing[file][k] for k in ("source_url", "author", "license", "obtained_on", "notes", "listening_ok")})
            assets.append(entry)
    return {"schema_version": "1.0", "note": "audio files are local only (gitignored); this provenance file is committed", "assets": assets}


def default_config(assets):
    by_category = {}
    for a in assets["assets"]:
        by_category.setdefault(a["category"], []).append({"id": a["id"], "path": a["file"], "sha256": a["sha256"]})
    return {
        "schema_version": "1.0", "variant_id": VARIANT_ID, "status": DRAFT, "seed": SEED, "code_version": CODE_VERSION,
        "frozen_at": None, "frozen_assignments_sha256": None,
        "cutoff_definition": "Butterworth design frequency; sosfiltfilt (zero phase) squares the response, so about -6 dB at the nominal edge",
        "noise": {"highpass_hz": audio.HP_HZ, "crest_limit_db": 12, "pad_ms": 200, "fade_ms": 150,
                  "level_mode": "target_ai", "max_snr_shift_db": 3, "excerpt_calm_fraction": 0.4,
                  "level_mode_note": "target_ai: each call gets the SNR (nominal +-max_snr_shift_db) at which the noise masks THAT call client speech "
                                     "to the level target_ai; snr_db is the nominal centre. level_mode snr: use snr_db as is.",
                  "snr_definition": "active client speech power (20 ms frames within 40 dB of the loudest, after 100 Hz high-pass) over "
                                    "mean noise power on the client turn interiors; one gain per call"},
        "headroom_peak": 0.98,
        "output": {"format": "WAV", "subtype": "PCM_16", "sample_rate": "source", "channels": 1},
        "conditions": {
            "N1": {"kind": "noise", "asset_category": "office", "level": {"snr_db": 3, "target_ai": 0.57}},
            "N2": {"kind": "noise", "asset_category": "cafe", "level": {"snr_db": 3, "target_ai": 0.65}},
            "N3": {"kind": "noise", "asset_category": "train", "level": {"snr_db": 2, "target_ai": 0.59}},
            "D1": {"kind": "telephone", "codec": "mulaw_8k", "order": 4, "level": {"band_hz": [600, 2100]}},
            "D2": {"kind": "muffle", "gain_db": -6.0, "order": 2, "level": {"cutoff_hz": 1200}}},
        "notch_definition": "one notch = 0.03 on the masking target (about 1 dB of SNR on a typical call); for D1/D2 one notch = both filter edges "
                            "20% further in (D1 500-2500 -> 600-2100 Hz, D2 1500 -> 1200 Hz)",
        "assets": by_category}


def condition_params(config, condition, severity):
    spec = config["conditions"][condition]
    level = spec["level"] if "level" in spec else spec["levels"][severity]
    kind = spec["kind"]
    if kind == "noise":
        params = {"snr_db": level["snr_db"], "calm_fraction": config["noise"].get("excerpt_calm_fraction")}
        if config["noise"].get("level_mode") == "target_ai":
            shift = config["noise"]["max_snr_shift_db"]
            params.update(target_ai=level["target_ai"], snr_bounds=[level["snr_db"] - shift, level["snr_db"] + shift])
        return kind, params
    if kind == "telephone":
        return kind, {"band_hz": level["band_hz"], "order": spec.get("order", 4), "codec": spec.get("codec") == "mulaw_8k"}
    if kind == "muffle":
        return kind, {"cutoff_hz": level["cutoff_hz"], "gain_db": spec["gain_db"], "order": spec.get("order", 2)}
    raise AugmentationError(f"{condition}: unknown kind {kind}")


def pick_asset(config, category, call_id):
    """Deterministic choice between the assets of a category (a hash of the call id)."""
    options = config["assets"][category]
    return options[int(hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:8], 16) % len(options)]


def load_config(dataset_root):
    path = variant_dir(dataset_root) / "config.json"
    if not path.is_file():
        raise AugmentationError(f"no config at {path}; run init first")
    return read_json(path)


# ----------------------------------------------------------------------------------------------------------------- one call
def load_call(dataset_root, call_id, verify_hash=True):
    root = Path(dataset_root)
    truth = read_json(root / "calls" / call_id / "ground_truth.json")
    render_id = truth["timing"]["render_id"]
    folder = root / "calls" / call_id / "renders" / render_id
    timings = read_json(folder / "timings.json")
    path = folder / timings["audio_file"]
    if verify_hash and normalised_sha256(path.read_bytes()) != truth["timing"]["audio_sha256"]:
        raise AugmentationError(f"{call_id}: {path.name} is not the G9-pinned audio (hash differs from ground_truth.timing.audio_sha256)")
    pcm, rate = audio.load_audio(path)
    if [t["turn_id"] for t in timings["turns"]] != [d["turn_id"] for d in truth["dialogue"]]:
        raise AugmentationError(f"{call_id}: timings and dialogue turn ids differ")
    return {"call_id": call_id, "path": path, "render_id": render_id, "format": sf.info(str(path)).format, "pcm": pcm, "rate": rate,
            "turns": timings["turns"], "roles": {d["turn_id"]: d["speaker"] for d in truth["dialogue"]},
            "timings_seconds": timings["duration_seconds"], "g9_audio_sha256": truth["timing"]["audio_sha256"]}


_ASSET_OK = {}


def _noise_path(dataset_root, asset):
    path = evaluation_root(dataset_root) / AUDIO_ASSETS / asset["path"]
    asset_id = asset["id"]
    if str(path) not in _ASSET_OK:
        if not path.is_file() or plain_sha256(path) != asset["sha256"]:
            raise AugmentationError(f"noise asset {asset_id} is missing or its sha256 differs from the config")
        _ASSET_OK[str(path)] = True
    return path


def augment_loaded(call, condition, severity, config, dataset_root):
    """Run one condition on a loaded call; validates the result and returns (output pcm, info, regions, asset)."""
    kind, params = condition_params(config, condition, severity)
    rate, pcm, call_id = call["rate"], call["pcm"], call["call_id"]
    regions = audio.client_regions(call["turns"], call["roles"], len(pcm), rate, config["noise"]["pad_ms"])
    noise, rng_for_turn, asset = None, None, None
    if kind == "noise":
        asset = pick_asset(config, config["conditions"][condition]["asset_category"], call_id)
        noise = audio.prepare_noise(_noise_path(dataset_root, asset), rate, config["noise"].get("crest_limit_db"))
        number = int(call_id.split("_")[1])
        seed = config["seed"]

        def rng_for_turn(turn_id):
            return np.random.default_rng([seed, number, turn_id])

    output, info = audio.augment_pcm(pcm, rate, regions, kind, params, noise=noise, rng_for_turn=rng_for_turn,
                                     fade_ms=config["noise"]["fade_ms"], headroom=config["headroom_peak"])
    if len(output) != len(pcm):
        raise AugmentationError(f"{call_id}: output has {len(output)} frames, source {len(pcm)}")
    if not info["outside_identical"]:
        raise AugmentationError(f"{call_id}: samples outside the client regions changed")
    peak = info["client_peak_after"]
    if peak > PEAK_TOLERANCE:
        raise AugmentationError(f"{call_id}: client peak {peak:.4f} above the headroom limit")
    if kind == "noise":
        snr = audio.measure_snr(pcm, output, regions, rate, info["client_gain_db"])
        info["snr"] = snr
        achieved, target = snr["fullband_db"], info["resolved_snr_db"]
        if abs(achieved - target) > SNR_TOLERANCE_DB:
            raise AugmentationError(f"{call_id}: achieved SNR {achieved:.2f} dB, target {target:.2f} dB")
        info["asset_gain_flag"] = info["asset_gain_db"] > 30
        info["ai_index"] = audio.articulation_index(pcm, output, regions, rate, info["client_gain_db"])
        goal, got = info["ai_target"], info["ai_index"]
        if goal is not None and not info["snr_clamped"] and abs(got - goal) > AI_TOLERANCE:
            raise AugmentationError(f"{call_id}: achieved masking {got:.3f}, target {goal}")
    else:
        info["client_level_change_db"] = audio.level_change_db(pcm, output, regions, rate)
    return output, info, regions, asset


def _metadata(call, row, config, assignment, kind, params, output, info, regions, asset, wav, config_sha256, assignments_sha256):
    from datetime import datetime, timezone
    rate = call["rate"]
    noise = None
    if asset is not None:
        noise = {"asset_id": asset["id"], "asset_sha256": asset["sha256"], "asset_gain_db": info["asset_gain_db"],
                 "asset_gain_flag": info["asset_gain_flag"], "noise_level_dbfs": info["noise_level_dbfs"],
                 "speech_level_dbfs": info["speech_level_dbfs"], "ai_index": info["ai_index"], "ai_target": info["ai_target"],
                 "resolved_snr_db": info["resolved_snr_db"], "snr_clamped": info["snr_clamped"], "offsets_s": info["offsets_s"]}
    return {
        "schema_version": "1.0", "variant_id": VARIANT_ID, "call_id": call["call_id"], "assignment": assignment, "kind": kind,
        "params": params, "config_sha256": config_sha256, "assignments_sha256": assignments_sha256, "code_version": CODE_VERSION,
        "seed": config["seed"],
        "source": {"path": row["source"]["path"], "render_id": call["render_id"], "format": call["format"], "sample_rate": rate,
                   "g9_audio_sha256": call["g9_audio_sha256"], "decoded_frames": len(call["pcm"]),
                   "timings_frames": int(round(call["timings_seconds"] * rate)),
                   "delta_ms": round((len(call["pcm"]) / rate - call["timings_seconds"]) * 1000, 3)},
        "output": {"path": "audio.wav", "sha256": plain_sha256(wav), "sample_rate": rate, "subtype": "PCM_16", "frames": len(output),
                   "duration_seconds": round(len(output) / rate, 3)},
        "client_regions": {"count": len(regions), "seconds": round(sum(r["hi"] - r["lo"] for r in regions) / rate, 3)},
        "noise": noise, "snr": info.get("snr"),
        "channel": None if asset is not None else {"client_level_change_db": info["client_level_change_db"]},
        "client_gain": {"db": info["client_gain_db"], "regions_triggering": info["client_gain_regions_triggering"],
                        "pre_gain_peak": info["client_pre_gain_peak"], "peak_after": info["client_peak_after"]},
        "employee": {"fullscale_samples_preserved": info["employee_fullscale_samples"], "peak": info["employee_peak"]},
        "rm_samples_identical": info["outside_identical"], "created_at": datetime.now(timezone.utc).isoformat()}


def augment_call(dataset_root, call_id, config, assignments, *, config_sha256, assignments_sha256, rebuild=False):
    """Write calls/<id>/audio_variants/environment_v2/{audio.wav, augmentation.json}. Returns the metadata (None for C1)."""
    row = assignments["calls"][call_id]
    if row["condition"] == CLEAN:
        return None
    root = Path(dataset_root)
    folder = root / "calls" / call_id / CALL_VARIANT_DIR
    meta_path = folder / "augmentation.json"
    assignment = {"condition": row["condition"], "severity": row["severity"]}
    if meta_path.is_file():
        previous = read_json(meta_path)
        if previous["config_sha256"] == config_sha256 and previous["assignment"] == assignment and (folder / "audio.wav").is_file():
            return previous
        if not rebuild:
            raise AugmentationError(f"{call_id}: a variant built with other settings exists; pass --rebuild to replace it")
    call = load_call(dataset_root, call_id)
    if call["g9_audio_sha256"] != row["source"]["audio_sha256"]:
        raise AugmentationError(f"{call_id}: ground truth audio hash differs from the assignment manifest")
    output, info, regions, asset = augment_loaded(call, row["condition"], row["severity"], config, root)
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / "audio.wav"
    audio.write_wav(wav, output, call["rate"])
    back, back_rate = audio.load_audio(wav)
    if back_rate != call["rate"] or not np.array_equal(back, output):
        raise AugmentationError(f"{call_id}: the written WAV does not read back identically")
    kind, params = condition_params(config, row["condition"], row["severity"])
    meta = _metadata(call, row, config, assignment, kind, params, output, info, regions, asset, wav, config_sha256, assignments_sha256)
    write_json(meta_path, meta)
    return meta


# ----------------------------------------------------------------------------------------------------------------- previews (listening gate)
PREVIEW_PLAN = (("N1", "ENGLISH"), ("N2", "MANDARIN"), ("N3", "SINGLISH"), ("D1", "EN_ZH_MIXED"), ("D2", "MANDARIN"))


def pick_window(call, target_seconds=40.0):
    """Frames of an excerpt that starts on an RM turn and contains client turns: RM, client, RM ... from a mid-call client turn."""
    turns, roles, rate = call["turns"], call["roles"], call["rate"]
    first = next((i for i, t in enumerate(turns) if i >= 2 and roles[t["turn_id"]] == "CLIENT" and t["end_seconds"] - t["start_seconds"] >= 6), 1)
    start = turns[first - 1]["start_seconds"]
    end = turns[first]["end_seconds"]
    for t in turns[first + 1:]:
        if end - start >= target_seconds:
            break
        end = t["end_seconds"]
    return int(round(start * rate)), min(len(call["pcm"]), int(round(end * rate)))


def preview(dataset_root, only=None, calls=None, cases=None, tag=None, progress=print):
    """Short listening clips (not part of the dataset): one call per condition, every severity, plus the clean excerpt of each call."""
    root = Path(dataset_root)
    config = load_config(root)
    assignments = read_json(variant_dir(root) / "assignments.json")
    out_dir = variant_dir(root) / "previews" / (tag or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = calls or {}
    rows, written_refs = [], set()
    if cases:
        plan = [(condition, call_id) for condition, call_id in cases]
    else:
        plan = [(condition, calls.get(condition) or next(c for c, r in sorted(assignments["calls"].items()) if r["split"] == "DEVELOPMENT"
                                                          and r["language"] == language and r["duration_profile"] == "SHORT"))
                for condition, language in PREVIEW_PLAN]
    for condition, call_id in plan:
        if only and condition not in only:
            continue
        if assignments["calls"][call_id]["split"] != "DEVELOPMENT":
            raise AugmentationError(f"{call_id}: previews are limited to DEVELOPMENT calls")
        call = load_call(root, call_id)
        lo, hi = pick_window(call)
        if call_id not in written_refs:
            audio.write_wav(out_dir / (call_id + "_C1.wav"), call["pcm"][lo:hi], call["rate"])
            written_refs.add(call_id)
        for severity in (CONDITIONS[condition][1] or (None,)):
            output, info, regions, asset = augment_loaded(call, condition, severity, config, root)
            name = call_id + "_" + condition + ("" if severity is None else "_" + severity) + ".wav"
            audio.write_wav(out_dir / name, output[lo:hi], call["rate"])
            snr = info.get("snr") or {}
            rows.append({"file": name, "call_id": call_id, "language": assignments["calls"][call_id]["language"], "condition": condition,
                         "severity": severity, "format": call["format"], "sample_rate": call["rate"],
                         "window_s": [round(lo / call["rate"], 1), round(hi / call["rate"], 1)],
                         "asset": None if asset is None else asset["id"],
                         "snr_target_db": info.get("resolved_snr_db"), "ai_target": info.get("ai_target"), "snr_clamped": info.get("snr_clamped"),
                         "ai_index": info.get("ai_index"), "snr_fullband_db": snr.get("fullband_db"), "snr_speechband_db": snr.get("speechband_db"),
                         "asset_gain_db": info.get("asset_gain_db"), "noise_dbfs": info.get("noise_level_dbfs"),
                         "client_level_change_db": info.get("client_level_change_db"), "client_gain_db": info["client_gain_db"],
                         "client_peak_after": info["client_peak_after"], "employee_fullscale_preserved": info["employee_fullscale_samples"]})
            progress(name)
    report_path = out_dir / "preview_report.json"
    kept = [r for r in read_json(report_path)["clips"] if r["file"] not in {n["file"] for n in rows}] if report_path.is_file() else []
    report = {"config_sha256": plain_sha256(variant_dir(root) / "config.json"), "status": config["status"], "clips": kept + rows}
    write_json(report_path, report)    # a re-run into the same folder adds to the report instead of replacing it
    return rows


# ----------------------------------------------------------------------------------------------------------------- freeze
def asset_problems(dataset_root, config):
    folder = evaluation_root(dataset_root) / AUDIO_ASSETS
    listed = {a["id"]: a for a in read_json(folder / "assets.json")["assets"]}
    problems = []
    for options in config["assets"].values():
        for option in options:
            entry = listed.get(option["id"])
            if entry is None:
                problems.append(option["id"] + ": not in assets.json")
                continue
            if str(entry["license"]).startswith("UNKNOWN"):
                problems.append(option["id"] + ": licence not recorded")
            if entry["listening_ok"] is not True:
                problems.append(option["id"] + ": listening_ok is not true")
            if entry["sha256"] != option["sha256"]:
                problems.append(option["id"] + ": config and assets.json hashes differ")
    return problems


def freeze(dataset_root):
    """DRAFT -> FROZEN. Pins the sha256 of the existing assignments.json and never rewrites it."""
    from datetime import datetime, timezone
    root = Path(dataset_root)
    config = load_config(root)
    if config["status"] == FROZEN:
        raise AugmentationError("config is already FROZEN")
    path = variant_dir(root) / "assignments.json"
    if assign_module.dump(assign_module.build_assignments(root, config["seed"])) != path.read_bytes():
        raise AugmentationError("assignments.json does not match a fresh deterministic recomputation; refusing to freeze")
    problems = asset_problems(root, config)
    if problems:
        raise AugmentationError("cannot freeze: " + "; ".join(problems))
    config.update({"status": FROZEN, "frozen_at": datetime.now(timezone.utc).isoformat(), "frozen_assignments_sha256": plain_sha256(path)})
    write_json(variant_dir(root) / "config.json", config)
    return config


# ----------------------------------------------------------------------------------------------------------------- build + verify
PROTECTED_NAMES = ("ground_truth.json", "transcript.json", "speaker_roles.json")


def protected_hashes(dataset_root, call_ids):
    """sha256 of every file the build must never change: gold, transcripts, roles, timings, render manifests and the split."""
    root = Path(dataset_root)
    out = {"splits/split.v1.json": plain_sha256(root / "splits" / "split.v1.json")}
    for call_id in call_ids:
        base = root / "calls" / call_id
        for name in PROTECTED_NAMES:
            out["calls/" + call_id + "/" + name] = plain_sha256(base / name)
        for render in sorted((base / "renders").iterdir()):
            for name in ("timings.json", "render_manifest.json"):
                out["calls/" + call_id + "/renders/" + render.name + "/" + name] = plain_sha256(render / name)
    return out


def _frozen_state(root):
    config = load_config(root)
    if config["status"] != FROZEN:
        raise AugmentationError("config.json is DRAFT; listen to the previews, then freeze before building")
    path = variant_dir(root) / "assignments.json"
    assignments_sha = plain_sha256(path)
    if config["frozen_assignments_sha256"] != assignments_sha:
        raise AugmentationError("assignments.json differs from the hash pinned at freeze")
    assignments = read_json(path)
    if assignments["split_sha256"] != plain_sha256(Path(root) / "splits" / "split.v1.json"):
        raise AugmentationError("the split changed since the assignments were written")
    return config, assignments, assignments_sha, plain_sha256(variant_dir(root) / "config.json")


def build(dataset_root, split=None, calls=None, rebuild=False, progress=print):
    root = Path(dataset_root)
    config, assignments, assignments_sha, config_sha = _frozen_state(root)
    ids = sorted(c for c, r in assignments["calls"].items()
                 if r["condition"] != CLEAN and (not split or r["split"] == split) and (not calls or c in calls))
    report_path = variant_dir(root) / "build_report.json"
    baseline = read_json(report_path)["protected_files_sha256"] if report_path.is_file() else None
    current = protected_hashes(root, sorted(assignments["calls"]))
    if baseline is not None and baseline != current:
        raise AugmentationError("a protected file (gold, transcript, roles, timings, manifest or split) changed since the first build")
    for index, call_id in enumerate(ids, start=1):
        meta = augment_call(root, call_id, config, assignments, config_sha256=config_sha, assignments_sha256=assignments_sha, rebuild=rebuild)
        progress(str(index) + "/" + str(len(ids)) + " " + call_id + " " + meta["assignment"]["condition"] + " " + str(meta["assignment"]["severity"]))
    summary = {"config_sha256": config_sha, "assignments_sha256": assignments_sha, "protected_files_sha256": current, "built": {}}
    for call_id, row in sorted(assignments["calls"].items()):
        meta_path = root / "calls" / call_id / CALL_VARIANT_DIR / "augmentation.json"
        if row["condition"] != CLEAN and meta_path.is_file():
            meta = read_json(meta_path)
            summary["built"][call_id] = {"condition": row["condition"], "severity": row["severity"], "output_sha256": meta["output"]["sha256"],
                                         "client_gain_db": meta["client_gain"]["db"],
                                         "snr_fullband_db": (meta["snr"] or {}).get("fullband_db")}
    write_json(report_path, summary)
    return summary


def verify(dataset_root, calls=None, progress=print):
    """Offline re-check of every built variant against the frozen config and assignments; returns a list of problems."""
    root = Path(dataset_root)
    config, assignments, assignments_sha, config_sha = _frozen_state(root)
    problems = []
    for call_id, row in sorted(assignments["calls"].items()):
        if calls and call_id not in calls:
            continue
        folder = root / "calls" / call_id / CALL_VARIANT_DIR
        if row["condition"] == CLEAN:
            if folder.exists():
                problems.append(call_id + ": C1 must not have a variant folder")
            continue
        try:
            meta = read_json(folder / "augmentation.json")
            if meta["assignment"] != {"condition": row["condition"], "severity": row["severity"]}:
                problems.append(call_id + ": augmentation.json disagrees with the assignment")
            if meta["config_sha256"] != config_sha or meta["assignments_sha256"] != assignments_sha:
                problems.append(call_id + ": built with a different config or assignments")
            if plain_sha256(folder / "audio.wav") != meta["output"]["sha256"]:
                problems.append(call_id + ": audio.wav hash differs from augmentation.json")
            call = load_call(root, call_id)
            output, rate = audio.load_audio(folder / "audio.wav")
            if rate != call["rate"] or len(output) != len(call["pcm"]):
                problems.append(call_id + ": sample rate or frame count differs from the source")
                continue
            regions = audio.client_regions(call["turns"], call["roles"], len(output), rate, config["noise"]["pad_ms"])
            outside = audio._outside_mask(len(output), regions)
            if not np.array_equal(output[outside], call["pcm"][outside]):
                problems.append(call_id + ": samples outside the client regions differ from the source")
            peak = max(int(np.abs(output[r["lo"]:r["hi"]].astype(np.int32)).max()) for r in regions) / audio.FULL_SCALE
            if peak > PEAK_TOLERANCE:
                problems.append(call_id + ": client peak above the limit")
            if meta["kind"] == "noise":
                achieved = audio.measure_snr(call["pcm"], output, regions, rate, meta["client_gain"]["db"])["fullband_db"]
                if abs(achieved - meta["noise"]["resolved_snr_db"]) > SNR_TOLERANCE_DB:
                    problems.append(call_id + ": achieved SNR off target")
        except (OSError, KeyError, AugmentationError) as error:
            problems.append(call_id + ": " + str(error))
        progress("verified " + call_id)
    report_path = variant_dir(root) / "build_report.json"
    if report_path.is_file() and not calls:
        if read_json(report_path)["protected_files_sha256"] != protected_hashes(root, sorted(assignments["calls"])):
            problems.append("protected files changed since the build")
    return problems


# ----------------------------------------------------------------------------------------------------------------- calibration (no audio written)
def calibrate(dataset_root, split="DEVELOPMENT", progress=print):
    """Run the real processing for every assigned call of one split WITHOUT writing audio, and report how each cell behaves.

    Needs only a DRAFT or FROZEN config. DEVELOPMENT by default: levels are tuned on development audio only.
    """
    root = Path(dataset_root)
    config = load_config(root)
    assignments = read_json(variant_dir(root) / "assignments.json")
    rows = []
    for call_id, row in sorted(assignments["calls"].items()):
        if row["condition"] == CLEAN or row["split"] != split:
            continue
        entry = {"call_id": call_id, "language": row["language"], "duration_profile": row["duration_profile"],
                 "condition": row["condition"], "severity": row["severity"], "ok": True, "error": None}
        try:
            call = load_call(root, call_id, verify_hash=False)
            output, info, regions, asset = augment_loaded(call, row["condition"], row["severity"], config, root)
            entry.update({"ai_index": info.get("ai_index"), "resolved_snr_db": info.get("resolved_snr_db"), "snr_clamped": info.get("snr_clamped"),
                          "snr_fullband_db": (info.get("snr") or {}).get("fullband_db"),
                          "snr_speechband_db": (info.get("snr") or {}).get("speechband_db"), "asset_gain_db": info.get("asset_gain_db"),
                          "client_gain_db": info["client_gain_db"], "client_level_change_db": info.get("client_level_change_db"),
                          "employee_fullscale": info["employee_fullscale_samples"]})
        except AugmentationError as error:
            entry.update(ok=False, error=str(error))
        rows.append(entry)
        progress(call_id + " " + row["condition"] + " " + str(row["severity"]) + (" ok" if entry["ok"] else " FAILED"))
    write_json(variant_dir(root) / "calibration_report.json", {"config_sha256": plain_sha256(variant_dir(root) / "config.json"),
                                                               "split": split, "rows": rows})
    return rows


def summarise_calibration(rows):
    """Group by condition, severity and language: calls, AI min/mean, worst client gain, largest asset gain, failures."""
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["severity"], r["language"]), []).append(r)
    lines = []
    for key in sorted(groups):
        g = groups[key]
        good = [r for r in g if r["ok"]]
        ai = [r["ai_index"] for r in good if r.get("ai_index") is not None]
        gains = [r["client_gain_db"] for r in good]
        asset = [r["asset_gain_db"] for r in good if r.get("asset_gain_db") is not None]
        change = [r["client_level_change_db"] for r in good if r.get("client_level_change_db") is not None]
        snrs = [r["resolved_snr_db"] for r in good if r.get("resolved_snr_db") is not None]
        lines.append({"condition": key[0], "severity": key[1], "language": key[2], "calls": len(g), "failed": len(g) - len(good),
                      "ai_min": min(ai) if ai else None, "ai_max": max(ai) if ai else None, "ai_mean": sum(ai) / len(ai) if ai else None,
                      "snr_min": min(snrs) if snrs else None, "snr_max": max(snrs) if snrs else None,
                      "clamped": sum(1 for r in good if r.get("snr_clamped")),
                      "client_gain_min_db": min(gains) if gains else None, "asset_gain_max_db": max(asset) if asset else None,
                      "level_change_min_db": min(change) if change else None})
    return lines


# ----------------------------------------------------------------------------------------------------------------- sharing copy
def share(dataset_root, target_root, split="DEVELOPMENT", progress=print):
    """Copy one split's environment_v2 variants into a derived dataset folder (the _development copy), file by file, verified by hash.

    The target already holds whole-call copies (calls/CALL_x/...). Only audio_variants/ is added, plus a subset assignments.json
    (this split's calls only, with the hash of the full manifest it was cut from) and the frozen config, so the evaluation engine can
    resolve environment_v2 against the copy. Nothing from another split is copied. Written directly (no temp-rename: OneDrive locks).
    """
    import shutil
    root, target = Path(dataset_root), Path(target_root)
    config, assignments, assignments_sha, _ = _frozen_state(root)
    derived = read_json(target / "dataset_manifest.json").get("derived_from") or {}
    if derived.get("split") != split:
        raise AugmentationError("target " + str(target) + " is not a " + split + " copy (dataset_manifest derived_from.split differs)")
    rows = {c: r for c, r in sorted(assignments["calls"].items()) if r["split"] == split}
    copied = 0
    for call_id, row in rows.items():
        if not (target / "calls" / call_id).is_dir():
            raise AugmentationError(call_id + ": the target has no such call folder")
        source = target / row["source"]["path"]
        if not source.is_file() or normalised_sha256(source.read_bytes()) != row["source"]["audio_sha256"]:
            raise AugmentationError(call_id + ": the target clean audio is not the G9-pinned recording")
        if row["condition"] == CLEAN:
            continue
        origin = root / "calls" / call_id / CALL_VARIANT_DIR
        destination = target / "calls" / call_id / CALL_VARIANT_DIR
        destination.mkdir(parents=True, exist_ok=True)
        meta = read_json(origin / "augmentation.json")
        for name in ("audio.wav", "augmentation.json"):
            shutil.copyfile(origin / name, destination / name)
        if plain_sha256(destination / "audio.wav") != meta["output"]["sha256"]:
            raise AugmentationError(call_id + ": the copied audio does not match its recorded sha256")
        copied += 1
        progress(str(copied) + " " + call_id + " " + row["condition"])
    folder = target / VARIANT_ROOT
    subset = {key: value for key, value in assignments.items() if key not in ("calls", "counts", "targets", "objective")}
    subset.update(calls=rows, subset={"split": split, "calls": len(rows), "source_assignments_sha256": assignments_sha,
                                      "note": "cut from the full assignments.json; the frozen config pins the full file hash"})
    write_json(folder / "assignments.json", subset)
    write_json(folder / "config.json", config)
    return {"split": split, "calls": len(rows), "variants_copied": copied, "clean_calls": len(rows) - copied}
