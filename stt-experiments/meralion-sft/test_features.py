"""The per-clip feature cache: bit-identical to the live processor, bound to the exact audio, and training with it equals training without it."""
import hashlib
import json

import numpy as np
import pytest
import torch

import sft_core as C
import sft_features as F
import sft_train as T
import test_sft as H
import test_sft_run as R


@pytest.fixture(autouse=True)
def small_speech_token(monkeypatch):
    monkeypatch.setattr(C, "SPEECH_TOKEN_ID", H.SPEECH)


def clips_of(data):
    return [json.loads(l) for name in ("clips_train.jsonl", "clips_validation.jsonl") for l in (data / name).read_text().splitlines()]


def make_cache(tmp_path, data, batch, verified=True):
    clips = clips_of(data)
    store = T.WaveStore(batch, {c["call_id"] for c in clips}, R.fake_load_audio)
    proc, prompt = R.FakeProcessor(), "prompt"
    msha = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    out = tmp_path / "cache"
    F.build_from(clips, store, proc, prompt, R.fake_load_audio, out, msha, log=lambda *a: None)
    report = F.verify_from(clips, store, proc, prompt, R.fake_load_audio, out, msha, log=lambda *a: None) if verified else None
    return out, clips, store, msha, report


def test_cached_output_is_bit_identical_to_the_live_processor(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    out, clips, store, msha, report = make_cache(tmp_path, data, batch)
    assert report["bit_identical"] and report["clips"] == len(clips) and report["mismatches"] == 0
    cache = F.FeatureCache.open(out, msha, "prompt")
    for clip in clips:
        audio = store.clip(clip)
        live, cached = F.compute(R.FakeProcessor(), "prompt", audio, R.fake_load_audio), cache.get(clip, audio)
        for key in F.KEYS:
            assert live[key].dtype == cached[key].dtype and torch.equal(live[key], cached[key])
    # the cache hands out copies: a consumer changing them cannot corrupt later steps
    a = cache.get(clips[0], store.clip(clips[0]))
    a["input_features"].zero_()
    assert torch.equal(cache.get(clips[0], store.clip(clips[0]))["input_features"], F.compute(R.FakeProcessor(), "prompt", store.clip(clips[0]), R.fake_load_audio)["input_features"])


def test_verify_detects_a_difference_and_an_unverified_cache_is_refused(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    out, clips, store, msha, _ = make_cache(tmp_path, data, batch, verified=False)
    with pytest.raises(ValueError, match="not marked verified"):
        F.FeatureCache.open(out, msha, "prompt")

    class Drifting(R.FakeProcessor):                                          # a processor that no longer reproduces the cached values
        def __call__(self, text, audios, return_tensors, padding):
            o = super().__call__(text, audios, return_tensors, padding)
            o["input_features"] = o["input_features"] + 1e-6
            return o
    report = F.verify_from(clips, store, Drifting(), "prompt", R.fake_load_audio, out, msha, log=lambda *a: None)
    assert not report["bit_identical"] and report["mismatches"] == len(clips)
    with pytest.raises(ValueError, match="not marked verified"):
        F.FeatureCache.open(out, msha, "prompt")


def test_cache_is_bound_to_audio_manifest_prompt_and_shard_bytes(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    out, clips, store, msha, _ = make_cache(tmp_path, data, batch)
    cache = F.FeatureCache.open(out, msha, "prompt")
    clip = clips[0]
    other = np.array(store.clip(clip)) + np.float32(1e-3)
    with pytest.raises(ValueError, match="audio slice"):
        cache.get(clip, other)
    shifted = dict(clip, clip_id="CALL_000:nope")
    with pytest.raises(ValueError, match="no entry"):
        cache.get(shifted, store.clip(clip))
    with pytest.raises(ValueError, match="different data package"):
        F.FeatureCache.open(out, "0" * 64, "prompt")
    with pytest.raises(ValueError, match="different chat prompt"):
        F.FeatureCache.open(out, msha, "another prompt")
    shard = out / "shard-0000.pt"
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(ValueError, match="damaged or edited"):
        F.FeatureCache.open(out, msha, "prompt").get(clip, store.clip(clip))


def test_training_with_the_cache_reaches_exactly_the_same_weights(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    out, clips, store, msha, report = make_cache(tmp_path, data, batch)
    assert report["bit_identical"]
    T.run(R.args_for(tmp_path, data, batch, "plain", k=m, extra=("--skip-baseline-validation",)), R.stack())
    T.run(R.args_for(tmp_path, data, batch, "cached", k=m, extra=("--skip-baseline-validation", "--feature-cache", str(out))), R.stack())
    for epoch in (1, 2):
        assert R.digest_of(tmp_path, "plain", epoch) == R.digest_of(tmp_path, "cached", epoch)
    plain = [json.loads(l) for l in (tmp_path / "runs" / "plain" / "train_metrics.jsonl").read_text().splitlines()]
    cached = [json.loads(l) for l in (tmp_path / "runs" / "cached" / "train_metrics.jsonl").read_text().splitlines()]
    assert [r["ce"] for r in plain] == [r["ce"] for r in cached] and [r["grad_norm"] for r in plain] == [r["grad_norm"] for r in cached]
    plain_val = [json.loads(l) for l in (tmp_path / "runs" / "plain" / "epoch_summary.jsonl").read_text().splitlines()]
    cached_val = [json.loads(l) for l in (tmp_path / "runs" / "cached" / "epoch_summary.jsonl").read_text().splitlines()]
    assert [e["validation"]["ce"] for e in plain_val[1:]] == [e["validation"]["ce"] for e in cached_val[1:]]


def test_a_cache_built_for_a_changed_data_package_stops_the_run(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    out, *_ = make_cache(tmp_path, data, batch)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["note"] = "edited"
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different data package"):
        T.run(R.args_for(tmp_path, data, batch, "stale", k=m, extra=("--skip-baseline-validation", "--feature-cache", str(out))), R.stack())
