"""environment_v2: client-only audio environments. Synthetic audio only; real-dataset checks skip when the release is not on disk."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fyp_evaluation.augmentation import AugmentationError, assign, audio, build

RATE = 16000
DATASET = Path(__file__).resolve().parents[1] / "datasets" / "advisory_execution_multilingual_v1"
needs_dataset = pytest.mark.skipif(not (DATASET / "splits" / "split.v1.json").is_file(), reason="release not on disk")


def speechlike(seconds, amplitude, seed, rate=RATE):
    """Band-limited noise with a syllable-rate envelope: a stand-in for speech with real silences inside."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = audio.sosfiltfilt(audio.butter(4, [200, 3000], "bandpass", fs=rate, output="sos"), rng.standard_normal(n))
    envelope = np.clip(np.sin(2 * np.pi * 4 * np.arange(n) / rate + rng.uniform(0, 6)), 0, None) ** 2
    x = x * envelope
    return amplitude * x / np.abs(x).max()


def make_call(client_amplitude=0.2, employee_amplitude=0.5, turns=6, rate=RATE):
    """EMPLOYEE / CLIENT alternation with exact 400 ms digital-silence gaps. Returns (int16 pcm, turns, roles)."""
    pieces, timings, roles, cursor = [], [], {}, 0.0
    for turn_id in range(turns):
        who = "EMPLOYEE" if turn_id % 2 == 0 else "CLIENT"
        seconds = 2.0 + 0.5 * (turn_id % 3)
        x = speechlike(seconds, employee_amplitude if who == "EMPLOYEE" else client_amplitude, turn_id, rate)
        if turn_id:
            pieces.append(np.zeros(int(0.4 * rate)))
            cursor += 0.4
        timings.append({"turn_id": turn_id, "start_seconds": round(cursor, 3), "end_seconds": round(cursor + len(x) / rate, 3)})
        roles[turn_id] = who
        pieces.append(x)
        cursor += len(x) / rate
    return audio.to_pcm(np.concatenate(pieces)), timings, roles


def noise_asset(path, seconds=30, rate=44100, seed=7):
    rng = np.random.default_rng(seed)
    x = audio.sosfiltfilt(audio.butter(2, [80, 4000], "bandpass", fs=rate, output="sos"), rng.standard_normal(seconds * rate))
    sf.write(str(path), (0.1 * x / np.abs(x).max()).astype(np.float32), rate, subtype="PCM_24")


def rng_for(seed=1):
    return lambda turn_id: np.random.default_rng([seed, 1, turn_id])


@pytest.fixture(scope="module")
def noise(tmp_path_factory):
    path = tmp_path_factory.mktemp("noise") / "office.wav"
    noise_asset(path)
    return audio.prepare_noise(path, RATE)


# ------------------------------------------------------------------------------------------------------------ regions + client-only
def test_client_regions_are_padded_inside_the_gaps_and_never_overlap_other_turns():
    pcm, turns, roles = make_call()
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    assert [r["turn_id"] for r in regions] == [1, 3, 5]
    for r in regions:
        assert r["lo"] < r["start"] and r["end"] <= r["hi"]
        assert r["start"] - r["lo"] <= int(0.2 * RATE) and r["hi"] - r["end"] <= int(0.2 * RATE)
        assert not pcm[r["lo"]:r["start"]].any() and not pcm[r["end"]:r["hi"]].any()   # padding is digital silence


@pytest.mark.parametrize("kind,params", [("noise", {"snr_db": 10}), ("telephone", {"band_hz": [300, 3400]}), ("muffle", {"cutoff_hz": 1500})])
def test_only_client_regions_change_and_the_timeline_is_identical(kind, params, noise):
    pcm, turns, roles = make_call()
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    out, info = audio.augment_pcm(pcm, RATE, regions, kind, params, noise=noise, rng_for_turn=rng_for())
    assert out.dtype == np.int16 and len(out) == len(pcm)
    assert info["outside_identical"]
    outside = audio._outside_mask(len(pcm), regions)
    assert np.array_equal(out[outside], pcm[outside])
    assert all(not np.array_equal(out[r["lo"]:r["hi"]], pcm[r["lo"]:r["hi"]]) for r in regions)


def test_noise_reaches_the_target_snr_against_client_speech_only(noise):
    pcm, turns, roles = make_call()
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    for target in (20, 10, 5):
        out, info = audio.augment_pcm(pcm, RATE, regions, "noise", {"snr_db": target}, noise=noise, rng_for_turn=rng_for())
        achieved = audio.measure_snr(pcm, out, regions, RATE, info["client_gain_db"])
        assert abs(achieved["fullband_db"] - target) <= 0.2
        assert achieved["speechband_db"] > achieved["fullband_db"] - 6   # speech-band SNR is recorded and plausible


def test_headroom_gain_is_client_only_and_keeps_the_snr(noise):
    pcm, turns, roles = make_call(client_amplitude=0.99, employee_amplitude=1.0)
    pcm[100] = 32767                                           # a pre-existing full-scale RM sample
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    out, info = audio.augment_pcm(pcm, RATE, regions, "noise", {"snr_db": 5}, noise=noise, rng_for_turn=rng_for())
    assert info["client_gain_db"] < 0 and info["client_gain_regions_triggering"] > 0
    assert info["client_peak_after"] <= build.PEAK_TOLERANCE
    assert info["outside_identical"] and info["employee_fullscale_samples"] >= 1   # preserved and reported, not a failure
    assert out[100] == 32767
    assert abs(audio.measure_snr(pcm, out, regions, RATE, info["client_gain_db"])["fullband_db"] - 5) <= 0.2


def test_a_same_seed_gives_identical_audio_and_other_seeds_other_excerpts(noise):
    pcm, turns, roles = make_call()
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    a, info_a = audio.augment_pcm(pcm, RATE, regions, "noise", {"snr_db": 10}, noise=noise, rng_for_turn=rng_for(1))
    b, info_b = audio.augment_pcm(pcm, RATE, regions, "noise", {"snr_db": 10}, noise=noise, rng_for_turn=rng_for(1))
    c, info_c = audio.augment_pcm(pcm, RATE, regions, "noise", {"snr_db": 10}, noise=noise, rng_for_turn=rng_for(2))
    assert np.array_equal(a, b) and info_a["offsets_s"] == info_b["offsets_s"]
    assert info_a["offsets_s"] != info_c["offsets_s"]
    assert len({offset for _, offset in info_a["offsets_s"]}) == len(info_a["offsets_s"])   # a different excerpt for every turn


# ------------------------------------------------------------------------------------------------------------ filters (actual response)
def measured_response(function, freqs, rate=48000):
    impulse = np.zeros(2 * rate)
    impulse[rate] = 1.0
    h = function(impulse)
    spectrum = np.abs(np.fft.rfft(h))
    axis = np.fft.rfftfreq(len(h), 1 / rate)
    return np.array([spectrum[np.argmin(np.abs(axis - f))] for f in freqs]), h


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


@pytest.mark.parametrize("cutoff", [2500, 1500])
def test_muffling_follows_the_squared_butterworth_response_and_is_zero_phase(cutoff):
    freqs = [200, 700, cutoff * 0.5, cutoff, cutoff * 1.5, cutoff * 2, cutoff * 3]
    got, h = measured_response(lambda x: audio.simulate_muffling(x, 48000, cutoff, gain_db=0.0), freqs)
    expected = audio.zero_phase_response(audio.muffle_sos(48000, cutoff), 48000, freqs)
    assert np.all(np.abs(db(got) - db(expected)) < 1.0)
    assert -7.0 < db(got[3]) < -5.0                    # about -6 dB at the nominal edge, not -3 dB
    centre = len(h) // 2
    assert np.allclose(h[centre + 1:centre + 200], h[centre - 1:centre - 200:-1], atol=1e-9)   # symmetric: zero phase, no delay


def test_muffling_attenuation_is_a_fixed_gain():
    x = speechlike(2.0, 0.3, 11)
    plain = audio.simulate_muffling(x, RATE, 2500, gain_db=0.0)
    assert np.allclose(audio.simulate_muffling(x, RATE, 2500, gain_db=-6.0), plain * 10 ** (-6 / 20))


@pytest.mark.parametrize("band", [(300, 3400), (500, 2500)])
def test_telephone_bandpass_follows_the_squared_response_and_rejects_outside_the_band(band):
    low, high = band
    freqs = [low / 2, low * 0.8, low, 1000, high, high * 1.25, min(high * 2, 20000)]
    got, _ = measured_response(lambda x: audio.simulate_telephone(x, 48000, band, codec=False), freqs)
    sos = audio.telephone_bandpass_sos(48000, low, high)
    expected = audio.zero_phase_response(sos, 48000, freqs)
    assert np.all(np.abs(db(got) - db(expected)) < 1.0)
    assert db(got[0]) < db(got[3]) - 30 and db(got[-1]) < db(got[3]) - 30     # at least 30 dB down an octave outside the band
    assert -7.0 < db(got[2]) - db(got[3]) < -4.5                                # low edge about -6 dB relative to the passband


def test_the_mulaw_codec_adds_quantisation_noise_but_no_delay_and_keeps_the_length():
    t = np.arange(int(1.0 * RATE)) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t)
    banded = audio.simulate_telephone(tone, RATE, (300, 3400), codec=False)
    coded = audio.simulate_telephone(tone, RATE, (300, 3400), codec=True)
    assert len(coded) == len(tone)
    core = slice(RATE // 4, -RATE // 4)
    sqnr = 10 * np.log10((banded[core] ** 2).mean() / ((coded[core] - banded[core]) ** 2).mean())
    assert 30 < sqnr < 50                                  # an 8-bit companded codec, not a transparent one
    noise_in = np.random.default_rng(3).standard_normal(RATE * 2)
    without = audio.simulate_telephone(noise_in, RATE, (300, 3400), codec=False)[2000:-2000]
    with_codec = audio.simulate_telephone(noise_in, RATE, (300, 3400), codec=True)[2000:-2000]
    assert np.argmax(np.correlate(without, with_codec, "full")) == len(without) - 1    # lag 0: the 8 kHz round trip shifts nothing in time


def test_mulaw_keeps_digital_silence_silent():
    assert not audio.mulaw_roundtrip(np.zeros(100)).any()


def test_the_filters_change_frequency_content_not_the_sample_grid():
    pcm, turns, roles = make_call(rate=24000)
    regions = audio.client_regions(turns, roles, len(pcm), 24000)
    for kind, params in (("telephone", {"band_hz": [500, 2500]}), ("muffle", {"cutoff_hz": 1500})):
        out, info = audio.augment_pcm(pcm, 24000, regions, kind, params)
        assert len(out) == len(pcm) and info["outside_identical"]
        for r in regions:
            a, b = audio.to_float(pcm[r["lo"]:r["hi"]]), audio.to_float(out[r["lo"]:r["hi"]])
            lag = np.argmax(np.correlate(b, a, "full")) - (len(a) - 1)
            assert lag == 0, kind


# ------------------------------------------------------------------------------------------------------------ pipeline on a synthetic dataset
def make_dataset(tmp_path, conditions=(("N1", None), ("D1", None), ("D2", None))):
    """A tiny release: evaluation-dataset/audio_assets and evaluation-dataset/datasets/ds. Returns (dataset root, eval root)."""
    eval_root = tmp_path / "evaluation-dataset"
    root = eval_root / "datasets" / "ds"
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "split.v1.json").write_text(json.dumps({"assignments": {}}), encoding="utf-8")
    (eval_root / "audio_assets" / "noise" / "office").mkdir(parents=True)
    noise_asset(eval_root / "audio_assets" / "noise" / "office" / "office.wav")
    assets = build.scan_assets(root)
    for a in assets["assets"]:
        a.update(license="CC0 (test)", listening_ok=True)
    build.write_json(eval_root / "audio_assets" / "noise" / "assets.json", assets)
    rows = {}
    for number, (condition, severity) in enumerate(conditions, start=1):
        call_id = "CALL_%03d" % number
        pcm, turns, roles = make_call(client_amplitude=0.6, employee_amplitude=1.0)
        folder = root / "calls" / call_id / "renders" / "synthetic_A"
        folder.mkdir(parents=True)
        audio.write_wav(folder / "audio.wav", pcm, RATE)
        dialogue = [{"turn_id": t["turn_id"], "speaker": roles[t["turn_id"]], "text": "x"} for t in turns]
        sha = build.normalised_sha256((folder / "audio.wav").read_bytes())
        truth = {"metadata": {"language_profile": "ENGLISH", "duration_profile": "SHORT"}, "dialogue": dialogue,
                 "timing": {"render_id": "synthetic_A", "audio_sha256": sha, "duration_seconds": len(pcm) / RATE}}
        (root / "calls" / call_id / "ground_truth.json").write_text(json.dumps(truth), encoding="utf-8")
        timings = {"audio_file": "audio.wav", "duration_seconds": len(pcm) / RATE, "turns": turns}
        (folder / "timings.json").write_text(json.dumps(timings), encoding="utf-8")
        (folder / "render_manifest.json").write_text("{}", encoding="utf-8")
        for name in build.PROTECTED_NAMES[1:]:
            (root / "calls" / call_id / name).write_text("{}", encoding="utf-8")
        rows[call_id] = {"split": "DEVELOPMENT", "language": "ENGLISH", "duration_profile": "SHORT", "duration_seconds": len(pcm) / RATE,
                         "condition": condition, "severity": severity,
                         "source": {"render_id": "synthetic_A", "path": "calls/" + call_id + "/renders/synthetic_A/audio.wav", "audio_sha256": sha},
                         "variant_path": "calls/" + call_id + "/audio_variants/environment_v2/audio.wav"}
    folder = build.variant_dir(root)
    folder.mkdir(parents=True)
    split_hash = build.plain_sha256(root / "splits" / "split.v1.json")
    (folder / "assignments.json").write_bytes(assign.dump({"split_sha256": split_hash, "calls": rows}))
    build.write_json(folder / "config.json", build.default_config(assets))
    return root, eval_root


def freeze_without_recompute(root, monkeypatch):
    """freeze() recomputes the assignment from the real release; here the tiny assignments file is the reference."""
    document = json.loads((build.variant_dir(root) / "assignments.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(assign, "build_assignments", lambda *args, **kwargs: document)
    return build.freeze(root)


def test_build_refuses_while_the_config_is_draft(tmp_path):
    root, _ = make_dataset(tmp_path)
    with pytest.raises(AugmentationError, match="DRAFT"):
        build.build(root)


def test_freezing_never_touches_the_assignments_and_pins_their_hash(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path)
    path = build.variant_dir(root) / "assignments.json"
    before = path.read_bytes()
    config = freeze_without_recompute(root, monkeypatch)
    assert path.read_bytes() == before                        # same assigned calls, conditions and severities, byte for byte
    assert config["status"] == "FROZEN" and config["frozen_assignments_sha256"] == build.plain_sha256(path)
    text = path.read_text(encoding="utf-8")
    assert "config_sha256" not in text                        # the assignment manifest never references the config
    assert "snr_db" not in text and "cutoff_hz" not in text   # no numeric levels, so level edits cannot invalidate it


def test_editing_levels_before_freeze_leaves_the_assignments_valid(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path)
    path = build.variant_dir(root) / "assignments.json"
    before = path.read_bytes()
    config = build.load_config(root)
    config["conditions"]["N1"]["level"]["snr_db"] = 8
    config["conditions"]["D2"]["level"]["cutoff_hz"] = 1800
    build.write_json(build.variant_dir(root) / "config.json", config)
    frozen = freeze_without_recompute(root, monkeypatch)
    assert path.read_bytes() == before
    assert frozen["conditions"]["N1"]["level"]["snr_db"] == 8
    summary = build.build(root, progress=lambda m: None)
    assert len(summary["built"]) == 3 and build.verify(root, progress=lambda m: None) == []


def test_freeze_needs_recorded_licences_and_listening_approval(tmp_path, monkeypatch):
    root, eval_root = make_dataset(tmp_path)
    path = eval_root / "audio_assets" / "noise" / "assets.json"
    listing = json.loads(path.read_text(encoding="utf-8"))
    listing["assets"][0].update(license=build.UNKNOWN_LICENCE, listening_ok=None)
    build.write_json(path, listing)
    with pytest.raises(AugmentationError, match="licence not recorded"):
        freeze_without_recompute(root, monkeypatch)


def test_build_writes_only_the_variants_and_is_reproducible(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path)
    freeze_without_recompute(root, monkeypatch)
    ids = ["CALL_001", "CALL_002", "CALL_003"]
    protected_before = build.protected_hashes(root, ids)
    summary = build.build(root, progress=lambda m: None)
    assert sorted(summary["built"]) == ids
    assert build.protected_hashes(root, ids) == protected_before               # gold, timings, manifests untouched
    assert build.verify(root, progress=lambda m: None) == []
    assert not list((root / "calls" / "CALL_001" / "renders").glob("*/audio_variants"))   # never written inside renders/
    meta_file = root / "calls" / "CALL_001" / "audio_variants" / "environment_v2" / "augmentation.json"
    first = json.loads(meta_file.read_text(encoding="utf-8"))
    assert first["rm_samples_identical"] and first["source"]["decoded_frames"] == first["output"]["frames"]
    assert first["noise"]["asset_id"] == "office_01" and abs(first["snr"]["fullband_db"] - first["noise"]["resolved_snr_db"]) <= 0.2
    again = build.build(root, calls=["CALL_001"], rebuild=True, progress=lambda m: None)
    assert again["built"]["CALL_001"]["output_sha256"] == first["output"]["sha256"]   # same settings give identical bytes
    assert first["client_gain"]["db"] <= 0 and first["client_gain"]["peak_after"] <= build.PEAK_TOLERANCE   # headroom is client-only
    assert first["employee"]["fullscale_samples_preserved"] > 0                    # RM full-scale samples preserved and reported


def test_a_changed_source_audio_is_refused(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path)
    freeze_without_recompute(root, monkeypatch)
    wav = root / "calls" / "CALL_001" / "renders" / "synthetic_A" / "audio.wav"
    pcm, rate = audio.load_audio(wav)
    audio.write_wav(wav, np.roll(pcm, 1), rate)
    with pytest.raises(AugmentationError, match="G9-pinned"):
        build.build(root, calls=["CALL_001"], progress=lambda m: None)


def test_verify_catches_a_tampered_variant(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path)
    freeze_without_recompute(root, monkeypatch)
    build.build(root, progress=lambda m: None)
    wav = root / "calls" / "CALL_002" / "audio_variants" / "environment_v2" / "audio.wav"
    pcm, rate = audio.load_audio(wav)
    pcm[5] = 1234                                              # a sample in the first RM turn
    audio.write_wav(wav, pcm, rate)
    assert any("CALL_002" in problem for problem in build.verify(root, progress=lambda m: None))


# ------------------------------------------------------------------------------------------------------------ the real release
@pytest.fixture(scope="module")
def real_inputs():
    return assign.load_inputs(DATASET)


@needs_dataset
@pytest.mark.skipif(not (build.variant_dir(DATASET) / "assignments.json").is_file(), reason="assignments not written yet")
def test_real_assignments_are_deterministic_and_satisfy_the_hard_rules(real_inputs):
    path = build.variant_dir(DATASET) / "assignments.json"
    assert assign.dump(assign.build_assignments(DATASET)) == path.read_bytes()        # recomputation is byte-identical
    document = json.loads(path.read_text(encoding="utf-8"))
    cells = {c: (r["condition"] if r["severity"] is None else r["condition"] + ":" + r["severity"]) for c, r in document["calls"].items()}
    assert assign.validate_assignment(real_inputs["calls"], cells) == []
    split = json.loads((DATASET / "splits" / "split.v1.json").read_text(encoding="utf-8"))["assignments"]
    assert {c: r["split"] for c, r in document["calls"].items()} == {c: v["split"] for c, v in split.items()}    # frozen split untouched
    counts = {}
    for row in document["calls"].values():
        counts[row["condition"]] = counts.get(row["condition"], 0) + 1
    assert counts == {"C1": 42, "N1": 42, "N2": 42, "N3": 41, "D1": 42, "D2": 41}
    assert sum(1 for r in document["calls"].values() if r["variant_path"]) == 208
    assert all(r["variant_path"] is None for r in document["calls"].values() if r["condition"] == "C1")
    assert document["split_sha256"] == build.plain_sha256(DATASET / "splits" / "split.v1.json")


@needs_dataset
def test_real_wav_calls_match_their_g9_timings_exactly(real_inputs):
    for call in real_inputs["calls"]:
        if call["audio_file"] != "audio.wav":
            continue
        folder = DATASET / "calls" / call["call_id"] / "renders" / call["render_id"]
        frames = sf.info(str(folder / "audio.wav")).frames
        assert frames == round(call["seconds"] * 24000), call["call_id"]


@needs_dataset
def test_real_mp3_decode_agrees_with_ffmpeg_and_with_the_g9_duration(real_inputs):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    import subprocess
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    english = [c for c in real_inputs["calls"] if c["audio_file"] == "audio.mp3"]
    assert len(english) == 75
    for call in english:
        path = DATASET / "calls" / call["call_id"] / "renders" / call["render_id"] / "audio.mp3"
        pcm, rate = audio.load_audio(path)
        raw = subprocess.run([exe, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
        reference = np.frombuffer(raw, dtype=np.int16)
        assert rate == 16000 and len(pcm) == len(reference), call["call_id"]
        assert np.abs(pcm.astype(np.int32) - reference.astype(np.int32)).max() <= 2, call["call_id"]   # same samples: lag 0, no inserted or dropped frames
        assert abs(len(pcm) / rate - call["seconds"]) <= 0.005, call["call_id"]                          # tail-only difference of a few ms


@needs_dataset
def test_real_g9_pinned_audio_hashes_hold_for_a_sample(real_inputs):
    by_file = {}
    for call in real_inputs["calls"]:
        by_file.setdefault((call["audio_file"], call["render_id"]), []).append(call)
    for calls in by_file.values():
        for call in calls[:2]:
            assert build.load_call(DATASET, call["call_id"])["g9_audio_sha256"] == call["audio_sha256"]


@needs_dataset
@pytest.mark.skipif(not (build.variant_dir(DATASET) / "config.json").is_file(), reason="config not written yet")
def test_previews_leave_gold_timings_manifests_and_the_split_untouched(real_inputs):
    ids = sorted(c["call_id"] for c in real_inputs["calls"])[:40]
    before = build.protected_hashes(DATASET, ids)
    variants = {p: build.plain_sha256(p) for p in DATASET.glob("calls/*/audio_variants/environment_v2/audio.wav")}
    rows = build.preview(DATASET, only=["N1", "D1"], tag="_pytest_scratch", progress=lambda m: None)
    try:
        assert {r["condition"] for r in rows} == {"N1", "D1"}
        assert build.protected_hashes(DATASET, ids) == before
        assert {p: build.plain_sha256(p) for p in DATASET.glob("calls/*/audio_variants/environment_v2/audio.wav")} == variants   # built variants untouched
    finally:
        import shutil
        shutil.rmtree(build.variant_dir(DATASET) / "previews" / "_pytest_scratch", ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------ noise limiter + masking proxy
def test_the_spike_limiter_caps_the_crest_factor_but_keeps_the_sustained_level():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(200000).astype(np.float32) * 0.05
    x[1000::20000] = 3.0                                       # rare spikes, about 36 dB over the average
    limited = audio.soft_limit(x, 12)
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    assert np.abs(limited).max() <= rms * 10 ** (12 / 20) * 1.001
    quiet = np.abs(x) < 0.7 * rms * 10 ** (12 / 20)
    assert np.array_equal(limited[quiet], x[quiet])            # everything below the knee is untouched
    assert abs(10 * math.log10(float((limited.astype(np.float64) ** 2).mean() / (x.astype(np.float64) ** 2).mean()))) < 1.0


def test_the_articulation_proxy_is_one_when_clean_and_falls_with_the_snr(noise):
    pcm, turns, roles = make_call()
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    scores = []
    for target in (30, 10, 0, -10):
        out, info = audio.augment_pcm(pcm, RATE, regions, "noise", {"snr_db": target}, noise=noise, rng_for_turn=rng_for())
        scores.append(audio.articulation_index(pcm, out, regions, RATE, info["client_gain_db"]))
    assert scores[0] > 0.95 and scores == sorted(scores, reverse=True) and scores[-1] < 0.5


def test_calibrate_reports_every_cell_and_writes_no_audio(tmp_path):
    root, _ = make_dataset(tmp_path)
    rows = build.calibrate(root, split="DEVELOPMENT", progress=lambda m: None)
    assert [r["call_id"] for r in rows] == ["CALL_001", "CALL_002", "CALL_003"] and all(r["ok"] for r in rows)
    assert not list(root.glob("calls/*/audio_variants"))                           # calibration never creates variants
    by_condition = {g["condition"]: g for g in build.summarise_calibration(rows)}
    assert by_condition["N1"]["ai_mean"] is not None and by_condition["D1"]["ai_mean"] is None
    assert by_condition["D2"]["level_change_min_db"] < 0
    assert (build.variant_dir(root) / "calibration_report.json").is_file()


# ------------------------------------------------------------------------------------------------------------ masking-target levels
def tilted_call(tilt_hz, seed=21):
    """Two stand-in voices: the same call, but speech energy centred on a different band."""
    rng = np.random.default_rng(seed)
    pieces, timings, roles, cursor = [], [], {}, 0.0
    for turn_id in range(6):
        who = "EMPLOYEE" if turn_id % 2 == 0 else "CLIENT"
        n = int(2.5 * RATE)
        broad = audio.butter(4, [150, 7000], "bandpass", fs=RATE, output="sos")
        narrow = audio.butter(4, [tilt_hz * 0.6, tilt_hz * 1.6], "bandpass", fs=RATE, output="sos")
        voice = audio.sosfiltfilt(broad, rng.standard_normal(n)) + 5 * audio.sosfiltfilt(narrow, rng.standard_normal(n))   # energy everywhere, emphasis at tilt_hz
        x = voice * np.clip(np.sin(2 * np.pi * 4 * np.arange(n) / RATE), 0, None) ** 2
        x = 0.3 * x / np.abs(x).max()
        if turn_id:
            pieces.append(np.zeros(int(0.4 * RATE)))
            cursor += 0.4
        timings.append({"turn_id": turn_id, "start_seconds": round(cursor, 3), "end_seconds": round(cursor + 2.5, 3)})
        roles[turn_id] = who
        pieces.append(x)
        cursor += 2.5
    return audio.to_pcm(np.concatenate(pieces)), timings, roles


def run_target(pcm, turns, roles, noise, target, bounds):
    regions = audio.client_regions(turns, roles, len(pcm), RATE)
    params = {"snr_db": 5, "target_ai": target, "snr_bounds": bounds}
    out, info = audio.augment_pcm(pcm, RATE, regions, "noise", params, noise=noise, rng_for_turn=rng_for())
    return out, info, regions


def test_the_masking_target_is_reached_and_the_achieved_snr_matches_the_solved_one(noise):
    pcm, turns, roles = make_call()
    for target in (0.85, 0.6):
        out, info, regions = run_target(pcm, turns, roles, noise, target, [-15, 25])
        assert not info["snr_clamped"]
        assert abs(audio.articulation_index(pcm, out, regions, RATE, info["client_gain_db"]) - target) <= 0.02
        assert abs(audio.measure_snr(pcm, out, regions, RATE, info["client_gain_db"])["fullband_db"] - info["resolved_snr_db"]) <= 0.2


def test_voices_that_the_noise_masks_differently_get_different_snrs_for_the_same_masking(noise):
    low = tilted_call(400)
    high = tilted_call(3000)
    results = {}
    for name, (pcm, turns, roles) in (("low", low), ("high", high)):
        out, info, regions = run_target(pcm, turns, roles, noise, 0.6, [-15, 25])
        results[name] = (info["resolved_snr_db"], audio.articulation_index(pcm, out, regions, RATE, info["client_gain_db"]), info["snr_clamped"])
    assert not results["low"][2] and not results["high"][2]
    assert abs(results["low"][1] - 0.6) <= 0.02 and abs(results["high"][1] - 0.6) <= 0.02    # same masking for both voices
    assert abs(results["low"][0] - results["high"][0]) > 1.0                                  # reached with clearly different SNRs


def test_the_snr_shift_is_clamped_and_reported(noise):
    pcm, turns, roles = make_call()
    out, info, regions = run_target(pcm, turns, roles, noise, 0.30, [4, 6])     # needs far more noise than the bounds allow
    assert info["snr_clamped"] and info["resolved_snr_db"] == 4
    out, info, regions = run_target(pcm, turns, roles, noise, 0.99, [4, 6])
    assert info["snr_clamped"] and info["resolved_snr_db"] == 6


def test_an_unreachable_masking_target_is_clamped_and_flagged_not_silently_missed(noise):
    pcm, turns, roles = make_call()
    _, info, _ = run_target(pcm, turns, roles, noise, 0.05, [-3, 3])        # far more masking than the allowed SNR range can give
    assert info["snr_clamped"] and info["resolved_snr_db"] == -3


# ------------------------------------------------------------------------------------------------------------ loud events + voice balance
def test_excerpts_avoid_loud_events_in_the_noise_asset():
    rng = np.random.default_rng(9)
    seconds = 120
    asset = (0.02 * rng.standard_normal(seconds * RATE)).astype(np.float32)
    t = np.arange(RATE // 2) / RATE
    for start in (20, 55, 90):                                 # three loud half-second beeps
        asset[start * RATE:start * RATE + len(t)] += (0.5 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
    levels = audio._frame_levels_db(asset, RATE)
    ceiling = float(np.median(levels)) + audio.CALM_MARGIN_DB
    for seed in range(60):
        excerpt, _ = audio.noise_excerpt(asset, 6 * RATE, RATE, np.random.default_rng([4, seed]), calm_fraction=0.4)
        hop = RATE // 10
        frames = np.asarray(excerpt[: len(excerpt) // hop * hop], dtype=np.float64).reshape(-1, hop)
        assert (10 * np.log10((frames ** 2).mean(axis=1) + 1e-20)).max() <= ceiling + 0.5
    plain = [audio.noise_excerpt(asset, 6 * RATE, RATE, np.random.default_rng([4, seed]))[0] for seed in range(60)]
    assert any(np.abs(e).max() > 0.4 for e in plain)           # without the rule the beeps do get in


@needs_dataset
@pytest.mark.skipif(not (build.variant_dir(DATASET) / "assignments.json").is_file(), reason="assignments not written yet")
def test_real_assignments_keep_every_client_voice_spread_over_the_conditions():
    document = json.loads((build.variant_dir(DATASET) / "assignments.json").read_text(encoding="utf-8"))
    cells = {}
    for row in document["calls"].values():
        cells.setdefault(row["client_voice"], {}).setdefault(row["condition"], 0)
        cells[row["client_voice"]][row["condition"]] += 1
    for voice, counts in cells.items():
        total = sum(counts.values())
        assert max(counts.values()) <= math.ceil(total / 6) + 1, voice          # nobody is piled into one condition
        if total >= 6:
            assert len(counts) == 6, voice                                       # and every voice meets every condition


def test_every_condition_has_a_single_level_and_plain_ids():
    config = build.default_config({"assets": []})
    assert all("level" in spec and "levels" not in spec for spec in config["conditions"].values())
    assert all(severities == () for _, severities in assign.CONDITIONS.values())
    assert assign.CELLS == ["C1", "N1", "N2", "N3", "D1", "D2"]


@needs_dataset
@pytest.mark.skipif(not (build.variant_dir(DATASET) / "assignments.json").is_file(), reason="assignments not written yet")
def test_real_assignments_carry_no_severity():
    document = json.loads((build.variant_dir(DATASET) / "assignments.json").read_text(encoding="utf-8"))
    assert all(row["severity"] is None for row in document["calls"].values())


# ------------------------------------------------------------------------------------------------------------ sharing copy
def make_dev_copy(tmp_path, root, calls):
    import shutil
    target = tmp_path / "dev_copy"
    target.mkdir()
    (target / "dataset_manifest.json").write_text(json.dumps({"derived_from": {"split": "DEVELOPMENT"}}), encoding="utf-8")
    for call in calls:
        shutil.copytree(root / "calls" / call, target / "calls" / call, ignore=shutil.ignore_patterns("audio_variants"))
    return target


def test_share_copies_only_the_split_variants_and_a_matching_subset_manifest(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path, conditions=(("C1", None), ("N1", None), ("D1", None)))
    path = build.variant_dir(root) / "assignments.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["calls"]["CALL_003"]["split"] = "HELDOUT"                       # one held-out call that must never be copied
    path.write_bytes(assign.dump(document))
    freeze_without_recompute(root, monkeypatch)
    build.build(root, progress=lambda m: None)
    target = make_dev_copy(tmp_path, root, ("CALL_001", "CALL_002"))
    summary = build.share(root, target, progress=lambda m: None)
    assert summary == {"split": "DEVELOPMENT", "calls": 2, "variants_copied": 1, "clean_calls": 1}
    copied = target / "calls" / "CALL_002" / "audio_variants" / "environment_v2"
    source = root / "calls" / "CALL_002" / "audio_variants" / "environment_v2"
    assert build.plain_sha256(copied / "audio.wav") == build.plain_sha256(source / "audio.wav")
    assert not (target / "calls" / "CALL_001" / "audio_variants").exists()      # a clean call needs no file
    assert not (target / "calls" / "CALL_003").exists()                          # the held-out call is absent
    subset = json.loads((target / "audio_augmentations" / "environment_v2" / "assignments.json").read_text(encoding="utf-8"))
    assert sorted(subset["calls"]) == ["CALL_001", "CALL_002"]
    assert subset["subset"]["source_assignments_sha256"] == build.plain_sha256(path)
    shared_config = json.loads((target / "audio_augmentations" / "environment_v2" / "config.json").read_text(encoding="utf-8"))
    assert shared_config["status"] == "FROZEN" and shared_config["frozen_assignments_sha256"] == subset["subset"]["source_assignments_sha256"]


def test_share_refuses_a_target_that_is_not_a_development_copy_or_has_other_audio(tmp_path, monkeypatch):
    root, _ = make_dataset(tmp_path)
    freeze_without_recompute(root, monkeypatch)
    build.build(root, progress=lambda m: None)
    target = make_dev_copy(tmp_path, root, ("CALL_001", "CALL_002", "CALL_003"))
    (target / "dataset_manifest.json").write_text(json.dumps({"derived_from": {"split": "HELDOUT"}}), encoding="utf-8")
    with pytest.raises(AugmentationError, match="not a DEVELOPMENT copy"):
        build.share(root, target)
    (target / "dataset_manifest.json").write_text(json.dumps({"derived_from": {"split": "DEVELOPMENT"}}), encoding="utf-8")
    wav = target / "calls" / "CALL_001" / "renders" / "synthetic_A" / "audio.wav"
    pcm, rate = audio.load_audio(wav)
    audio.write_wav(wav, np.roll(pcm, 1), rate)
    with pytest.raises(AugmentationError, match="G9-pinned"):
        build.share(root, target)
