"""End-to-end CPU test of sft_train.run() with a stand-in model/processor: logs, sampling summaries, epoch checkpoints, interrupted-and-resumed == continuous."""
import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import sft_core as C
import sft_data as D
import sft_train as T
import test_sft as H


@pytest.fixture(autouse=True)
def small_speech_token(monkeypatch):
    monkeypatch.setattr(C, "SPEECH_TOKEN_ID", H.SPEECH)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [10 + (ord(c) % 200) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, text, audios, return_tensors, padding):
        ids = torch.cat([torch.tensor([2, 106, 5]), torch.full((300,), H.SPEECH), torch.tensor([107, 108, 106, 251, 108])]).unsqueeze(0)
        energy = float(np.abs(audios[0]).mean())
        feats = torch.full((1, 128, 3000), energy)
        return {"input_ids": ids, "input_features": feats, "feature_attention_mask": torch.ones(1, 3000, dtype=torch.long)}


def stand_in_model(seed=0):
    base = H.tiny_model(seed=seed, dropout=0.0)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.speech_encoder = nn.ModuleList([nn.Sequential(nn.AvgPool1d(10), )])      # [1,128,3000] -> [1,128,300]
            self.speech_encoder.add_module("layers", base.speech_encoder.layers)            # encoder-shaped layers (the encoder-lora scope's targets)
            self.proj = nn.Linear(128, 4)
            self.ln_speech = base.ln_speech
            self.speech_audio_adapter = base.speech_audio_adapter
            self.text_decoder = base.text_decoder
            self.inner_encoder = base.speech_encoder[0]

        def forward(self, input_ids, attention_mask, input_features, feature_attention_mask, use_cache=False, labels=None):
            x = self.speech_encoder[0](input_features).transpose(1, 2)               # [1,300,128]
            h = self.inner_encoder(self.proj(x))
            for block in self.speech_encoder.layers:
                h = block(h)
            s = self.speech_audio_adapter(self.ln_speech(h))
            emb = self.text_decoder.base_model.embed_tokens(input_ids)
            mask = (input_ids == H.SPEECH).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(mask, s.reshape(-1, 32))
            return self.text_decoder(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False, labels=labels)
    m = Wrap()
    m.requires_grad_(False)
    return m


def make_package(tmp_path, n_calls=4):
    import soundfile
    data, batch = tmp_path / "data", tmp_path / "batch"
    (batch / "audio").mkdir(parents=True), data.mkdir()
    calls, train, val = [], [], []
    rng = np.random.default_rng(0)
    for i in range(n_calls):
        cid = f"CALL_{i:03d}"
        wav = (rng.standard_normal(16000 * 70) * 0.05).astype("float32")
        path = batch / "audio" / f"{cid}.wav"
        soundfile.write(path, wav, 16000, subtype="FLOAT")
        calls.append({"call_id": cid, "audio": f"audio/{cid}.wav", "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for j in range(5):
            text = f"clip {i} {j} buy 500 shares of Apple"
            start = j * 12.0
            clip = {"clip_id": f"{cid}:{j}-{j}", "call_id": cid, "split": "TRAIN" if i < n_calls - 1 else "VALIDATION", "language": "ENGLISH", "turn_ids": [j], "speakers": ["CLIENT"],
                    "start_seconds": start, "end_seconds": start + 10.0, "duration_seconds": 10.0, "start_sample": int(start * 16000), "end_sample": int((start + 10) * 16000), "text": text,
                    "entity_spans": [{"start": text.index("500"), "end": text.index("500") + 3, "group": "value:T03", "label": "e"}] if j % 2 == 0 else [], "evidence_turns": [], "important": j % 2 == 0}
            (train if i < n_calls - 1 else val).append(clip)
    (batch / "calls.json").write_text(json.dumps({"revision": D.MODEL_REVISION, "calls": calls}))
    (data / "clips_train.jsonl").write_text("".join(json.dumps(c) + "\n" for c in train))
    (data / "clips_validation.jsonl").write_text("".join(json.dumps(c) + "\n" for c in val))
    fp = {name: D.tokens_fingerprint([D.tokenise_clip(FakeTokenizer(), c)[0] for c in clips]) for name, clips in (("TRAIN", train), ("VALIDATION", val))}
    files = {n: hashlib.sha256((data / n).read_bytes()).hexdigest() for n in ("clips_train.jsonl", "clips_validation.jsonl")}
    (data / "manifest.json").write_text(json.dumps({"files": files, "token_fingerprints": fp}))
    return data, batch, len(train), sum(c["important"] for c in train)


def args_for(tmp_path, data, batch, run_id, k, objective="wce", sampling="targeted", extra=()):
    return T.parse(["--run-id", run_id, "--out-root", str(tmp_path / "runs"), "--data", str(data), "--model", "unused", "--batch", str(batch), "--objective", objective,
                    "--sampling", sampling, "--k", str(k), "--epochs", "2", "--accumulation", "4", "--log-every", "1", "--save-every", "1", "--dropout", "0.05", "--lr", "1e-2", *extra])


def fake_load_audio(x):
    return x[0] if isinstance(x, tuple) else x


def stack():
    return stand_in_model(0), FakeProcessor(), "prompt", fake_load_audio


def digest_of(tmp_path, run_id, epoch):
    state, meta = C.load_lora_safetensors(tmp_path / "runs" / run_id / "checkpoints" / f"epoch-{epoch}.safetensors")
    return C.state_digest(state)


@pytest.mark.parametrize("objective,sampling", [("ce", "uniform"), ("wce", "targeted")])
def test_run_writes_structured_outputs_and_matched_budgets(tmp_path, capsys, objective, sampling):
    data, batch, n, m = make_package(tmp_path)
    T.run(args_for(tmp_path, data, batch, "R", k=m, objective=objective, sampling=sampling), stack())
    out = tmp_path / "runs" / "R"
    rows = [json.loads(l) for l in (out / "train_metrics.jsonl").read_text().splitlines()]
    expected_steps = len(D.group_boundaries(n + m, 4))
    assert [r["step_in_epoch"] for r in rows if r["epoch"] == 1] == list(range(1, expected_steps + 1))
    assert rows[0]["objective_type"] == ("ce" if objective == "ce" else "wce@2") and set(["ce", "wce2", "ent_ce", "non_ent_ce", "objective_loss", "lr", "grad_norm"]) <= set(rows[0])
    if objective == "ce":
        assert rows[0]["objective_loss"] == rows[0]["ce"]
    summaries = [json.loads(l) for l in (out / "sampling_summary.jsonl").read_text().splitlines()]
    assert [s["draws"] for s in summaries] == [n + m, n + m] and all(s["unique_clips"] == n for s in summaries)
    if sampling == "targeted":
        assert all(s["important_draw_share"] > m / n for s in summaries)
    epoch_rows = [json.loads(l) for l in (out / "epoch_summary.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in epoch_rows] == [0, 1, 2] and epoch_rows[0]["update"] == 0 and epoch_rows[0]["train"] is None
    epoch_rows = epoch_rows[1:]
    assert len(epoch_rows) == 2 and epoch_rows[0]["validation"]["ce"] is not None and (out / "checkpoints" / "epoch-2.safetensors").exists()
    log = capsys.readouterr().out
    assert "CE (Loss):" in log if objective == "ce" else "WCE@2 (Loss):" in log
    assert ("WCE@2" not in log) if objective == "ce" else True
    assert rows[-1]["update"] == 2 * expected_steps and rows[-1]["lr"] > 0


def test_interrupted_run_resumes_to_the_same_weights_as_a_continuous_run(tmp_path, monkeypatch):
    data, batch, n, m = make_package(tmp_path)
    T.run(args_for(tmp_path, data, batch, "A", k=m), stack())
    calls = {"n": 0}
    original = T.train_group

    def interrupting(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:                                              # mid-epoch: after 3 completed optimiser steps (state saved after each)
            raise KeyboardInterrupt
        return original(*a, **k)
    monkeypatch.setattr(T, "train_group", interrupting)
    with pytest.raises(KeyboardInterrupt):
        T.run(args_for(tmp_path, data, batch, "B", k=m), stack())
    monkeypatch.setattr(T, "train_group", original)
    T.run(args_for(tmp_path, data, batch, "B", k=m, extra=("--resume",)), stack())
    assert digest_of(tmp_path, "A", 2) == digest_of(tmp_path, "B", 2) and digest_of(tmp_path, "A", 1) == digest_of(tmp_path, "B", 1)


def test_stop_after_epoch_then_resume_equals_continuous(tmp_path):
    data, batch, n, m = make_package(tmp_path)
    T.run(args_for(tmp_path, data, batch, "A", k=m), stack())
    T.run(args_for(tmp_path, data, batch, "C", k=m, extra=("--stop-after-epoch", "1")), stack())
    assert not (tmp_path / "runs" / "C" / "checkpoints" / "epoch-2.safetensors").exists()
    T.run(args_for(tmp_path, data, batch, "C", k=m, extra=("--resume",)), stack())
    assert digest_of(tmp_path, "A", 2) == digest_of(tmp_path, "C", 2)


def test_frozen_parameters_never_change_and_a_second_run_in_the_same_folder_is_refused(tmp_path):
    data, batch, n, m = make_package(tmp_path)
    model = stand_in_model(0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    T.run(args_for(tmp_path, data, batch, "F", k=0), (model, FakeProcessor(), "prompt", fake_load_audio))
    for k, v in model.state_dict().items():
        if "lora_" not in k and "base_layer" not in k:
            assert torch.equal(before[k], v), k
    with pytest.raises(SystemExit):
        T.run(args_for(tmp_path, data, batch, "F", k=0), stack())


def test_validation_pass_never_updates_parameters_or_leaves_gradients(tmp_path):
    data, batch, n, m = make_package(tmp_path)
    model = stand_in_model(0)
    targets = C.inject_decoder_lora(model, 8, 16, 0.05, 42)
    for p in C.lora_parameters(model).values():
        p.data.normal_(0, 0.05)
    before = C.state_digest(C.lora_state(model))
    clips = [json.loads(l) for l in (data / "clips_validation.jsonl").read_text().splitlines()]
    store = T.WaveStore(batch, {c["call_id"] for c in clips}, fake_load_audio)
    valset = T.ClipSet(clips, FakeTokenizer(), store, FakeProcessor(), "prompt", fake_load_audio)
    T.evaluate_ce(model, valset, "cpu", torch.float32)
    assert C.state_digest(C.lora_state(model)) == before and all(p.grad is None for p in model.parameters())
    assert not model.training
