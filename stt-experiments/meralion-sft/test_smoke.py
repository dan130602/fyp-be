"""The whole smoke script (every check A-E, the production trainer inside it) on a CPU stand-in with MERaLiON's module names, so Python-level mistakes cannot first appear inside a GPU allocation."""
import hashlib
import json
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

import sft_core as C
import sft_data as D
import sft_smoke as S
import sft_train as T
import test_sft as H
import test_sft_run as R


class Config(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool, self.proj = nn.AvgPool1d(10), nn.Linear(128, 32)
        self.layers = nn.ModuleList([H.ToyEncoderBlock()])                                  # Whisper-encoder-shaped layer: q/k/v/out_proj + fc1/fc2, used in the forward

    def forward(self, input_features, attention_mask=None):
        h = self.proj(self.pool(input_features).transpose(1, 2))
        for block in self.layers:
            h = block(h)
        return types.SimpleNamespace(last_hidden_state=h)


def stand_in(seed=0):
    base = H.tiny_model(seed=seed, dropout=0.0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.speech_encoder = Enc()
            self.ln_speech = base.ln_speech
            self.speech_audio_adapter = base.speech_audio_adapter
            self.text_decoder = base.text_decoder
            self.config = Config(speech_config=types.SimpleNamespace(apply_spec_augment=True, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, encoder_layerdrop=0.0))

        def forward(self, input_ids, attention_mask, input_features, feature_attention_mask, use_cache=False, labels=None):
            s = self.speech_audio_adapter(self.ln_speech(self.speech_encoder(input_features, attention_mask=feature_attention_mask).last_hidden_state))
            emb = self.text_decoder.base_model.embed_tokens(input_ids)
            emb = emb.masked_scatter((input_ids == H.SPEECH).unsqueeze(-1).expand_as(emb), s.reshape(-1, 32))
            return self.text_decoder(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False, labels=labels)
    m = M()
    m.requires_grad_(False)
    return m


class Tok(R.FakeTokenizer):
    def decode(self, ids):
        return "".join(chr(32 + int(i) % 90) for i in ids)


class Proc(R.FakeProcessor):
    tokenizer = Tok()

    def __call__(self, text, audios, return_tensors, padding):
        out = super().__call__(text, audios, return_tensors, padding)
        out["input_ids"] = torch.cat([out["input_ids"], torch.full((1, 22), 251)], dim=1)       # 330 prompt tokens like the real one
        return out


def package(tmp_path):
    import soundfile
    data, batch = tmp_path / "data", tmp_path / "batch"
    (batch / "audio").mkdir(parents=True), data.mkdir()
    rng = np.random.default_rng(0)
    langs = ["ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH", "ENGLISH", "MANDARIN", "SINGLISH"]
    calls, train, val = [], [], []
    for i, lang in enumerate(langs):
        cid = f"CALL_{i:03d}"
        path = batch / "audio" / f"{cid}.wav"
        soundfile.write(path, (rng.standard_normal(16000 * 130) * 0.05).astype("float32"), 16000, subtype="FLOAT")
        calls.append({"call_id": cid, "audio": f"audio/{cid}.wav", "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for j in range(6):
            text = f"clip {i} {j} buy 500 shares of Apple"
            start = j * 21.0
            clip = {"clip_id": f"{cid}:{j}-{j}", "call_id": cid, "split": "TRAIN" if i < 5 else "VALIDATION", "language": lang, "turn_ids": [j], "speakers": ["CLIENT"], "start_seconds": start,
                    "end_seconds": start + 20.0, "duration_seconds": 20.0, "start_sample": int(start * 16000), "end_sample": int((start + 20) * 16000), "text": text,
                    "entity_spans": [{"start": text.index("500"), "end": text.index("500") + 3, "group": "value:T03", "label": "e"}] if j % 2 == 0 else [], "evidence_turns": [], "important": j % 2 == 0}
            (train if i < 5 else val).append(clip)
    (batch / "calls.json").write_text(json.dumps({"revision": D.MODEL_REVISION, "calls": calls}))
    (data / "clips_train.jsonl").write_text("".join(json.dumps(c) + "\n" for c in train))
    (data / "clips_validation.jsonl").write_text("".join(json.dumps(c) + "\n" for c in val))
    fp = {n: D.tokens_fingerprint([D.tokenise_clip(R.FakeTokenizer(), c)[0] for c in cs]) for n, cs in (("TRAIN", train), ("VALIDATION", val))}
    (data / "manifest.json").write_text(json.dumps({"files": {n: hashlib.sha256((data / n).read_bytes()).hexdigest() for n in ("clips_train.jsonl", "clips_validation.jsonl")}, "token_fingerprints": fp}))
    return data, batch


@pytest.fixture(autouse=True)
def small_speech_token(monkeypatch):
    monkeypatch.setattr(C, "SPEECH_TOKEN_ID", H.SPEECH)


def test_every_smoke_check_passes_on_the_stand_in(tmp_path, monkeypatch):
    data, batch = package(tmp_path)
    monkeypatch.setattr(T, "load_stack", lambda model_dir, device="cuda", **k: (stand_in(0), Proc(), "prompt", R.fake_load_audio))
    monkeypatch.setattr(S, "REPORT", {"checks": {}, "numbers": {}})
    monkeypatch.setattr("sys.argv", ["sft_smoke.py", "--data", str(data), "--model", "unused", "--batch", str(batch), "--out", str(tmp_path / "out"), "--stand-in-test"])
    code = S.main()
    report = json.loads((tmp_path / "out" / "smoke_report.json").read_text())
    failed = {k: v["detail"] for k, v in report["checks"].items() if not v["ok"]}
    assert not failed, failed
    assert code == 0 and report["summary"]["overall"] == "PASS" and set(report["checks"]) >= {"A0_cluster_tokenizer_reproduces_the_frozen_target_token_ids", "B_production_trainer_few_steps", "C2_wce1_equals_ce_and_accumulation_equivalence",
                                                   "D_throughput_vram_and_validation_speed", "E1_repeated_batch_learning_and_reload", "E2_merge_arithmetic_matches_the_lora_forward"}


def test_peak_gb_and_gpu_name_do_not_recurse(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "GPU X")
    assert T.peak_gb() == 3.0 and T.gpu_name() == "GPU X"


def run_with(tmp_path, monkeypatch, patch=None, extra=()):
    data, batch = package(tmp_path)
    monkeypatch.setattr(T, "load_stack", lambda model_dir, device="cuda", **k: (stand_in(0), Proc(), "prompt", R.fake_load_audio))
    monkeypatch.setattr(S, "REPORT", {"checks": {}, "numbers": {}})
    if patch:
        patch()
    monkeypatch.setattr("sys.argv", ["sft_smoke.py", "--data", str(data), "--model", "unused", "--batch", str(batch), "--out", str(tmp_path / "out"), "--stand-in-test", *extra])
    return S.run_main()


def test_a_failed_required_check_is_an_overall_fail(tmp_path, monkeypatch):
    def break_trainer():
        monkeypatch.setattr(T, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    code = run_with(tmp_path, monkeypatch, break_trainer)
    text = (tmp_path / "out" / "SMOKE_RESULT.txt").read_text()
    assert code == 1 and text.startswith("FAIL") and "B_production_trainer_few_steps" in text


def test_a_crash_outside_any_check_is_an_overall_fail(tmp_path, monkeypatch):
    def break_baseline():
        monkeypatch.setattr(S, "fingerprint", lambda model: (_ for _ in ()).throw(MemoryError("oom")))
    with pytest.raises(MemoryError):
        run_with(tmp_path, monkeypatch, break_baseline)
    result = json.loads((tmp_path / "out" / "smoke_report.json").read_text())
    assert result["summary"]["overall"] == "FAIL" and "MemoryError" in result["summary"]["crash"]


def test_skipped_or_missing_required_checks_never_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "REPORT", {"checks": {n: {"status": "pass", "ok": True, "detail": None} for n in S.REQUIRED_FULL}, "numbers": {}})
    (tmp_path / "o").mkdir()
    S.REPORT["checks"]["D_throughput_vram_and_validation_speed"]["status"] = "skipped"
    assert S.finish(tmp_path / "o", False) == 1
    del S.REPORT["checks"]["E1_repeated_batch_learning_and_reload"]
    S.REPORT["checks"]["D_throughput_vram_and_validation_speed"]["status"] = "pass"
    assert S.finish(tmp_path / "o", False) == 1 and "E1_repeated" in (tmp_path / "o" / "SMOKE_RESULT.txt").read_text()
    S.REPORT["checks"]["E1_repeated_batch_learning_and_reload"] = {"status": "pass", "ok": True, "detail": None}
    assert S.finish(tmp_path / "o", False) == 0
    assert S.finish(tmp_path / "o", True) == 1       # a full-run check list is not a pre-flight list: missing A5
