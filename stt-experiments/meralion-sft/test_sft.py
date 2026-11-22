"""CPU tests for the SFT pipeline (tiny Gemma2 inside a stand-in with MERaLiON's module names and forward structure). Run:
    python -X utf8 -m pytest -q test_sft.py --basetemp=<scratch dir>
The real MERaLiON classes are exercised by the GPU smoke (sft_smoke.py), not here.
"""
import json
import math
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sft_core as C
import sft_data as D

MODEL_DIR = Path(__file__).resolve().parents[2] / "stt-service" / "meralion" / "model"
needs_model = pytest.mark.skipif(not (MODEL_DIR / "tokenizer.json").is_file(), reason="the base model directory (stt-service/meralion/model, git-ignored) is not present; download it first")
SPEECH = 299
N_SPEECH = 3


# ---------------------------------------------------------------- stand-in model
class ToyAdapter(nn.Module):
    """The pinned MERaLiON speech adapter's layout (module names and structure of modeling_meralion3.MERaLiON3ASRSpeechAudioAdaperLarge with speech_mlp_use_projection=false):
    mlp_adapter = [Linear, SiLU, Dropout], out_proj; no gate_proj / pool_proj."""

    def __init__(self, first, dropout):
        super().__init__()
        self.mlp_adapter = nn.Sequential(first, nn.SiLU(), nn.Dropout(dropout))
        self.out_proj = nn.Linear(32, 32)

    def forward(self, x):
        return self.out_proj(self.mlp_adapter(x))


class ToyEncoderBlock(nn.Module):
    """A Whisper-encoder-shaped layer: self_attn.{q,k,v,out}_proj + fc1 / fc2 (the names the encoder LoRA scope targets)."""

    def __init__(self, d=32, ffn=64):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj, self.self_attn.out_proj = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, ffn), nn.Linear(ffn, d)

    def forward(self, x):
        a = self.self_attn
        w = torch.softmax(a.q_proj(x) @ a.k_proj(x).transpose(-1, -2) / math.sqrt(x.shape[-1]), dim=-1)
        x = x + a.out_proj(w @ a.v_proj(x))
        return x + self.fc2(F.gelu(self.fc1(x)))


def tiny_model(seed=0, dropout=0.1):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(seed)

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = Gemma2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                               max_position_embeddings=256, sliding_window=64, query_pre_attn_scalar=16, attn_implementation="eager")
            self.speech_encoder = nn.Sequential(nn.Linear(4, 32), nn.Dropout(dropout))
            self.ln_speech = nn.LayerNorm(32)
            first = nn.Linear(32, 32)                                   # drawn before the decoder, as the old single adapter Linear was: the decoder's random initialisation is unchanged
            self.text_decoder = Gemma2ForCausalLM(cfg)
            self.speech_audio_adapter = ToyAdapter(first, dropout)
            # the real encoder has q_proj/v_proj too (and the decoder has o_proj where the encoder has out_proj): make sure an unscoped pattern would have hit them
            self.speech_encoder.add_module("layers", nn.ModuleList([ToyEncoderBlock()]))

        def forward(self, input_ids, attention_mask, input_features, feature_attention_mask, use_cache=False, labels=None):
            h = self.speech_encoder[0](input_features)
            for block in self.speech_encoder.layers:
                h = block(h)
            s = self.speech_audio_adapter(self.ln_speech(h))
            emb = self.text_decoder.base_model.embed_tokens(input_ids)
            mask = (input_ids == SPEECH).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(mask, s.reshape(-1, 32))
            return self.text_decoder(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False, labels=labels)

    model = Fake()
    with torch.no_grad():                                                # tied embeddings at their random-init scale cap the logits near +-0.6: scale up so a repeated batch is learnable
        model.text_decoder.model.embed_tokens.weight.mul_(12.0)
    return model


def make_example(n_target, entity_positions=(), seed=0, clip_id="x"):
    g = torch.Generator().manual_seed(seed)
    prefix_ids = torch.cat([torch.tensor([2, 106, 5]), torch.full((N_SPEECH,), SPEECH), torch.tensor([107, 108])])
    target = torch.randint(10, 250, (n_target,), generator=g)
    ex = C.Example()
    ex.clip_id, ex.language = clip_id, "ENGLISH"
    ex.input_ids = torch.cat([prefix_ids, target]).unsqueeze(0)
    ex.attention_mask = torch.ones_like(ex.input_ids)
    ex.input_features = torch.randn(1, N_SPEECH, 4, generator=g)
    ex.feature_attention_mask = torch.ones(1, 10, dtype=torch.long)
    ex.prefix = prefix_ids.numel()
    ent = torch.zeros(n_target, dtype=torch.bool)
    ent[list(entity_positions)] = True
    ex.entity, ex.n, ex.n_entity, ex.audio_seconds = ent, n_target, int(ent.sum()), 1.0
    return ex


# ---------------------------------------------------------------- data rules
def test_clip_packing_respects_30_seconds_and_excludes_long_turns():
    def turn(i, a, b, text="word"):
        return {"turn_id": i, "speaker": "EMPLOYEE" if i % 2 == 0 else "CLIENT", "text": text, "start": a, "end": b}
    call = {"call_id": "CALL_X", "language": "ENGLISH", "evidence_turns": [], "spans": [],
            "turns": [turn(0, 0, 10), turn(1, 10.5, 20), turn(2, 20.5, 29.9), turn(3, 30, 70), turn(4, 71, 80), turn(5, 80.5, 84)]}
    clips, excluded = D.build_clips(call, "TRAIN")
    assert [c["turn_ids"] for c in clips] == [[0, 1, 2], [4, 5]]        # 0..29.9 s fits; turn 3 (40 s) is excluded and splits the packing
    assert [t["turn_id"] for t in excluded] == [3]
    call["turns"][2]["end"] = 30.5                                       # 30.5 s from the first start no longer fits one chunk
    clips, _ = D.build_clips(call, "TRAIN")
    assert [c["turn_ids"] for c in clips] == [[0, 1], [2], [4, 5]]
    assert all(c["end_sample"] - c["start_sample"] <= D.MAX_CLIP_SAMPLES for c in clips)
    assert all(c["end_sample"] - c["start_sample"] == round(c["end_seconds"] * 16000) - round(c["start_seconds"] * 16000) for c in clips)


def test_join_rule_and_entity_span_offsets():
    text, offsets = D.join_turn_texts(["Hello there.", "买入五百股", "Yes please", "好的"])
    assert text == "Hello there. 买入五百股 Yes please 好的"
    assert text[offsets[2]:].startswith("Yes please") and text[offsets[3]:] == "好的"
    assert D.join_turn_texts(["您好。", "好的"])[0] == "您好。好的"
    call = {"call_id": "C", "language": "ENGLISH", "evidence_turns": [1], "turns": [], "spans": [{"turn_id": 1, "start": 4, "end": 10, "group": "value:T01", "label": "EV"}]}
    turns = [{"turn_id": 0, "speaker": "EMPLOYEE", "text": "Good morning.", "start": 0.0, "end": 2.0}, {"turn_id": 1, "speaker": "CLIENT", "text": "Buy $500 of Apple", "start": 2.5, "end": 5.0}]
    clip = D.make_clip(call, "TRAIN", turns)
    span = clip["entity_spans"][0]
    assert clip["text"][span["start"]:span["end"]] == "Buy $500 of Apple"[4:10] == "$500 o"
    assert clip["evidence_turns"] == [1] and D.important_clip(clip)


@needs_model
def test_target_ids_match_a_stored_model_generation_and_terminators():
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    text = "And selling below your original purchase price would crystallize a loss; the transaction fee is zero point two five percent of the transaction amount. Are you willing to accept an investment loss from this sale as part of reducing your exposure?"
    stored = [2079, 13026, 3582, 861, 3464, 8210, 3712, 1134, 16162, 20992, 476, 4783, 235289, 573, 13678, 10848, 603, 8199, 2377, 1378, 4105, 5243, 576, 573, 13678, 3619, 235265, 5881,
              692, 11465, 577, 6502, 671, 9245, 4783, 774, 736, 5934, 685, 1702, 576, 18442, 861, 12793, 235336, 107, 108, 1]       # diag-217-v2 P001: what the model itself generated
    clip = {"text": text, "entity_spans": []}
    ids, flags, _ = D.tokenise_clip(tok, clip)
    assert ids == stored and ids[-3:] == [107, 108, 1] and not any(flags)


@needs_model
def test_entity_mask_marks_the_occurrence_not_the_vocabulary():
    pytest.importorskip("tokenizers")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    text = "Buy 500 shares now, then sell 500 shares later, 买入五百股 好的"
    first = text.index("500")
    zh = text.index("五百")
    clip = {"text": text, "entity_spans": [{"start": first, "end": first + 3, "group": "value:T03", "label": "a"}, {"start": zh, "end": zh + 2, "group": "value:T03", "label": "b"}]}
    ids, flags, groups = D.tokenise_clip(tok, clip)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    flagged = [(a, b) for (a, b), f in zip(enc["offset_mapping"], flags) if f]
    assert len(flagged) >= 2                                            # the second '500' is NOT flagged although it has the same token ids
    assert all(a < first + 3 and first < b or a < zh + 2 and zh < b for a, b in flagged)
    second = text.index("500", first + 3)
    assert not any(a < second + 3 and second < b for a, b in flagged)
    assert flags[-3:] == [False] * 3 and len(flags) == len(ids)


def test_sampling_plan_is_matched_deterministic_and_covers_every_clip():
    train = [f"c{i}" for i in range(40)]
    important = train[:15]
    uni = D.plan_epochs(train, important, 15, 3, 42, "uniform")
    tar = D.plan_epochs(train, important, 15, 3, 42, "targeted")
    assert uni == D.plan_epochs(train, important, 15, 3, 42, "uniform")
    for u, t in zip(uni, tar):
        assert len(u) == len(t) == 55 and set(u) == set(t) == set(train)
        from collections import Counter
        assert all(Counter(t)[c] == 2 for c in important) and all(Counter(t)[c] == 1 for c in train[15:])      # K = M: every important clip exactly twice
        assert max(Counter(u).values()) <= 2
    assert uni[0] != uni[1]                                             # reshuffled every epoch
    with pytest.raises(ValueError):
        D.plan_epochs(train, [], 5, 1, 42, "targeted")                  # nothing to draw from
    assert D.group_boundaries(55, 8) == [8] * 6 + [7]


def test_lr_schedule_advances_per_update_and_ends_positive():
    total = 100
    f = [C.lr_factor(u, total) for u in range(1, total + 1)]
    assert f[0] == pytest.approx(1 / 5) and f[4] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[4:], f[5:])) and 0 < f[-1] < 0.05


# ---------------------------------------------------------------- LoRA scope, modes, losses
def test_lora_is_scoped_to_decoder_projections_and_fp32():
    model = tiny_model()
    targets = C.inject_decoder_lora(model, 8, 16, 0.05, seed=42)
    assert len(targets) == 2 * 7 and all(t.startswith("text_decoder.model.layers.") for t in targets)
    counts = C.check_trainable(model, targets)
    assert counts["trainable"] == sum(p.numel() for p in C.lora_parameters(model).values())
    assert not any("speech_encoder" in n or "speech_audio_adapter" in n or "lm_head" in n or "embed" in n for n, p in model.named_parameters() if p.requires_grad)
    assert all(p.dtype == torch.float32 for p in C.lora_parameters(model).values())
    assert all(float(p.detach().abs().sum()) == 0 for n, p in C.lora_parameters(model).items() if ".lora_B." in n)       # B starts at zero: the adapted model equals the base
    # identical initialisation for every run with the same seed
    other = tiny_model()
    C.inject_decoder_lora(other, 8, 16, 0.05, seed=42)
    assert C.state_digest(C.lora_state(model)) == C.state_digest(C.lora_state(other))


def test_mode_policy_keeps_frozen_dropout_off_even_after_recursive_train():
    model = tiny_model()
    C.inject_decoder_lora(model, 8, 16, 0.05)
    model.train()                                                       # what a careless trainer does: this switches the adapter dropout ON
    with pytest.raises(AssertionError):
        C.check_mode_policy(model, True)
    C.set_train_mode(model)
    C.check_mode_policy(model, True)
    assert not model.speech_audio_adapter.mlp_adapter[2].training and model.text_decoder.model.layers[0].self_attn.q_proj.lora_dropout["default"].training
    C.set_eval_mode(model)
    C.check_mode_policy(model, False)


def test_shift_applied_once_and_matches_the_models_own_loss():
    model = tiny_model(dropout=0.0)
    ex = make_example(9, entity_positions=[2, 3])
    model.eval()
    with torch.no_grad():
        mine = C.token_losses(model, ex)
        labels = ex.input_ids.clone()
        labels[:, :ex.prefix] = -100
        theirs = model(input_ids=ex.input_ids, attention_mask=ex.attention_mask, input_features=ex.input_features, feature_attention_mask=ex.feature_attention_mask, labels=labels).loss
    assert mine.numel() == 9                                            # every supervised token incl. the last one is scored exactly once
    assert float(mine.mean()) == pytest.approx(float(theirs), rel=1e-5)


def test_wce_weight_one_equals_ce_and_wce2_is_the_pooled_weighted_mean():
    loss = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ent = torch.tensor([False, True, True, False, False])
    ce = C.micro_objective(loss, ent, "ce", 2.0, 5.0)
    assert float(ce) == pytest.approx(3.0)
    assert float(C.micro_objective(loss, ent, "wce", 1.0, 5.0)) == pytest.approx(float(ce))
    d2 = C.group_denominator([(5, 2)], "wce", 2.0)
    assert d2 == 7.0 and float(C.micro_objective(loss, ent, "wce", 2.0, d2)) == pytest.approx((1 + 2 * 2 + 2 * 3 + 4 + 5) / 7)
    s = C.Sums()
    s.add(loss, ent)
    m = s.metrics()
    assert m["ce"] == pytest.approx(3.0) and m["wce2"] == pytest.approx(20 / 7)
    assert m["ent_ce"] == pytest.approx(2.5) and m["non_ent_ce"] == pytest.approx(10 / 3)
    empty = C.Sums()
    empty.add(torch.tensor([1.0]), torch.tensor([False]))
    assert empty.metrics()["ent_ce"] is None                            # N/A when the group is absent


@pytest.mark.parametrize("objective,weight", [("ce", 2.0), ("wce", 2.0)])
def test_accumulated_gradients_equal_the_combined_batch_with_variable_lengths(objective, weight):
    exs = [make_example(n, entity_positions=pos, seed=i) for i, (n, pos) in enumerate([(5, [1]), (11, [0, 1, 7]), (8, [])])]
    accumulated = tiny_model(seed=3, dropout=0.0)
    C.inject_decoder_lora(accumulated, 8, 16, 0.0, seed=1)
    # give B non-zero values so every LoRA gradient is non-trivial
    for n, p in C.lora_parameters(accumulated).items():
        if ".lora_B." in n:
            p.data.normal_(0, 0.02)
    combined = tiny_model(seed=3, dropout=0.0)
    C.inject_decoder_lora(combined, 8, 16, 0.0, seed=1)
    combined.load_state_dict(accumulated.state_dict())
    accumulated.eval(), combined.eval()
    denominator = C.group_denominator([(e.n, e.n_entity) for e in exs], objective, weight)
    for e in exs:                                                       # what train_group does: one backward per micro-batch, shared group denominator
        C.micro_objective(C.token_losses(accumulated, e), e.entity, objective, weight, denominator).backward()
    # reference: ONE padded batch, one loss over all supervised tokens
    maxlen = max(e.input_ids.shape[1] for e in exs)
    ids = torch.stack([F.pad(e.input_ids[0], (0, maxlen - e.input_ids.shape[1])) for e in exs])
    mask = torch.stack([F.pad(e.attention_mask[0], (0, maxlen - e.attention_mask.shape[1])) for e in exs])
    feats = torch.cat([e.input_features for e in exs])
    logits = combined(input_ids=ids, attention_mask=mask, input_features=feats, feature_attention_mask=None).logits
    total, weights = 0.0, 0.0
    w = 1.0 if objective == "ce" else weight
    for i, e in enumerate(exs):
        tok = F.cross_entropy(logits[i, e.prefix - 1:e.input_ids.shape[1] - 1].float(), e.input_ids[0, e.prefix:], reduction="none")
        wt = torch.where(e.entity, torch.full_like(tok, w), torch.ones_like(tok))
        total = total + (wt * tok).sum()
        weights += float(wt.sum())
    (total / weights).backward()
    ga, gc = C.lora_parameters(accumulated), C.lora_parameters(combined)
    assert any(float(p.grad.abs().sum()) > 0 for p in ga.values())
    for name in ga:
        assert torch.allclose(ga[name].grad, gc[name].grad, atol=1e-6, rtol=1e-4), name
    # the naive mean of per-clip means would be a different objective
    naive = sum(float(C.token_losses(accumulated, e).detach().mean()) for e in exs) / 3
    pooled = float(total / weights) if objective == "ce" else None
    if objective == "ce":
        assert abs(naive - pooled) > 1e-4


def test_training_changes_only_lora_and_a_repeated_batch_is_learned():
    model = tiny_model(seed=5, dropout=0.0)
    C.inject_decoder_lora(model, 8, 16, 0.0, seed=2)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    params = list(C.lora_parameters(model).values())
    opt = torch.optim.AdamW(params, lr=3e-2, weight_decay=0.0)
    ex = make_example(10, seed=9)
    C.set_train_mode(model)
    losses = []
    for _ in range(60):
        loss = C.token_losses(model, ex)
        loss.mean().backward()
        opt.step(), opt.zero_grad()
        losses.append(float(loss.mean()))
    assert losses[-1] < 0.8 * losses[0]
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    assert any(float(p.abs().sum()) > 0 for n, p in C.lora_parameters(model).items() if ".lora_B." in n)


def test_lora_checkpoint_roundtrip_and_resume_state(tmp_path):
    model = tiny_model(seed=1, dropout=0.0)
    C.inject_decoder_lora(model, 8, 16, 0.0, seed=3)
    for n, p in C.lora_parameters(model).items():
        p.data.normal_(0, 0.1)
    state = C.lora_state(model)
    path = tmp_path / "x.safetensors"
    C.save_lora_safetensors(path, state, {"rank": 8, "alpha": 16, "digest": C.state_digest(state)})
    loaded, meta = C.load_lora_safetensors(path)
    other = tiny_model(seed=1, dropout=0.0)
    C.inject_decoder_lora(other, 8, 16, 0.0, seed=99)
    C.load_lora_state(other, loaded)
    assert C.state_digest(C.lora_state(other)) == C.state_digest(state) == json_digest(meta)
    ex = make_example(7)
    model.eval(), other.eval()
    assert torch.equal(C.token_losses(model, ex), C.token_losses(other, ex))
    random.seed(1), torch.manual_seed(1)
    saved = C.rng_state()
    a = (random.random(), float(torch.rand(1)))
    C.restore_rng(saved)
    assert a == (random.random(), float(torch.rand(1)))


def json_digest(meta):
    return json.loads(meta["digest"]) if meta["digest"].startswith('"') else meta["digest"]


# ---------------------------------------------------------------- export
def test_export_merges_exactly_the_lora_modules_and_leaves_the_base_untouched(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    import sft_export as X
    base = tmp_path / "base"
    base.mkdir()
    g = torch.Generator().manual_seed(0)
    names = ["text_decoder.model.layers.0.self_attn.q_proj.weight", "text_decoder.model.layers.0.mlp.down_proj.weight", "speech_encoder.layers.0.self_attn.q_proj.weight", "ln_speech.weight"]
    shapes = {names[0]: (16, 8), names[1]: (8, 20), names[2]: (16, 8), names[3]: (8,)}
    tensors = {n: torch.randn(shapes[n], generator=g).to(torch.bfloat16) for n in names}
    save_file({k: v for k, v in tensors.items() if k != names[1]}, str(base / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file({names[1]: tensors[names[1]]}, str(base / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {n: ("model-00002-of-00002.safetensors" if n == names[1] else "model-00001-of-00002.safetensors") for n in names}}))
    for f in X.COPY_FILES:
        if not (base / f).exists():
            (base / f).write_text("{}")
    r = 4
    lora = {"text_decoder.model.layers.0.self_attn.q_proj.lora_A": torch.randn(r, 8, generator=g), "text_decoder.model.layers.0.self_attn.q_proj.lora_B": torch.randn(16, r, generator=g) * 0.1,
            "text_decoder.model.layers.0.mlp.down_proj.lora_A": torch.randn(r, 20, generator=g), "text_decoder.model.layers.0.mlp.down_proj.lora_B": torch.randn(8, r, generator=g) * 0.1}
    C.save_lora_safetensors(tmp_path / "l.safetensors", lora, {"rank": r, "alpha": 8, "digest": "abc", "run_id": "t", "epoch": 1})
    before = {p.name: X.sha256_file(p) for p in base.iterdir()}
    manifest = X.export(base, tmp_path / "l.safetensors", tmp_path / "out", "test-id")
    assert before == {p.name: X.sha256_file(p) for p in base.iterdir()}
    out = {}
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        with safe_open(str(tmp_path / "out" / shard), framework="pt") as f:
            out.update({k: f.get_tensor(k) for k in f.keys()})
    scale = 8 / r
    expect = (tensors[names[0]].float() + scale * lora[names[0][:-7] + ".lora_B"] @ lora[names[0][:-7] + ".lora_A"]).to(torch.bfloat16)
    assert torch.equal(out[names[0]], expect)
    assert torch.equal(out[names[2]], tensors[names[2]]) and torch.equal(out[names[3]], tensors[names[3]])       # encoder and norms byte for byte
    assert manifest["status"] == "complete" and manifest["kind"] == "complete_checkpoint" and "adapter_config.json" not in manifest["files"]
    assert manifest["files"]["model-00001-of-00002.safetensors"]["sha256"] == X.sha256_file(tmp_path / "out" / "model-00001-of-00002.safetensors")


def test_targeted_pool_larger_k_draws_further_rounds_and_keeps_old_plans():
    from collections import Counter
    train = [f"c{i}" for i in range(40)]
    pool = train[:12]                                                    # an entity pool smaller than K
    for epoch_plan in D.plan_epochs(train, pool, 20, 3, 42, "targeted"):
        c = Counter(epoch_plan)
        assert len(epoch_plan) == 60 and set(epoch_plan) == set(train)   # matched budget: 40 clips + K=20 extras
        assert all(c[x] >= 2 for x in pool) and sum(1 for x in pool if c[x] == 3) == 8 and all(c[x] == 1 for x in train[12:])   # one full round (12) + 8 of the pool a second time
    # K that fits the pool is the single-round plan of the original implementation (earlier runs' plans are reproduced)
    import random
    rng = random.Random("42:targeted:0")
    extra = rng.sample(sorted(train[:15]), 15)
    expected = sorted(train) + extra
    random.Random("42:shuffle:targeted:0").shuffle(expected)
    assert D.plan_epochs(train, train[:15], 15, 1, 42, "targeted")[0] == expected
    with pytest.raises(ValueError):
        D.plan_epochs(train, [], 5, 1, 42, "targeted")
