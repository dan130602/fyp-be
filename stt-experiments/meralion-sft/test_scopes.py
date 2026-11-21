"""Training scopes beyond the decoder (adapter-lora, adapter-full, encoder-lora): the exact layers each one targets, which parameters train, gradient flow through the frozen
components, module modes, BF16 computation of the full adapter, exporter enforcement of the declared scope, the scope pre-flight and end-to-end runs. Stand-in model with the pinned
MERaLiON module names (the real classes are exercised by scope_check_job.sbatch on the GPU)."""
import argparse
import json
from pathlib import Path

import pytest
import torch

import sft_core as C
import sft_export as X
import test_sft as H
import test_sft_run as R
import sft_train as T

ADAPTER = ["speech_audio_adapter.mlp_adapter.0", "speech_audio_adapter.out_proj"]
ENCODER = sorted(f"speech_encoder.layers.0.{n}" for n in ("fc1", "fc2", "self_attn.k_proj", "self_attn.out_proj", "self_attn.q_proj", "self_attn.v_proj"))


@pytest.fixture(autouse=True)
def small_speech_token(monkeypatch):
    monkeypatch.setattr(C, "SPEECH_TOKEN_ID", H.SPEECH)


def inject(model, scope, seed=42, dropout=0.05):
    return C.inject_adapter_full(model) if scope == "adapter-full" else C.inject_lora(model, scope, 8, 16, dropout, seed)


def test_each_scope_targets_exactly_its_own_layers_and_never_a_similarly_named_one():
    model = H.tiny_model()
    assert C.expected_targets(model, "adapter-lora") == ADAPTER
    assert C.expected_targets(model, "encoder-lora") == ENCODER
    decoder = C.expected_targets(model, "decoder")
    assert len(decoder) == 14 and all(n.startswith("text_decoder.model.layers.") for n in decoder)
    assert not set(decoder) & set(ENCODER) and not set(decoder) & set(ADAPTER)
    with pytest.raises(ValueError):
        C.scope_pattern(model, "everything")


@pytest.mark.parametrize("scope,names", [("adapter-lora", ADAPTER), ("encoder-lora", ENCODER)])
def test_lora_scopes_train_only_their_matrices_start_as_the_base_model_and_are_seeded(scope, names):
    base = H.tiny_model(seed=3, dropout=0.0)
    model = H.tiny_model(seed=3, dropout=0.0)
    targets = inject(model, scope)
    assert targets == names
    counts = C.check_trainable(model, targets, scope)
    assert counts["scope"] == scope and counts["lora_modules"] == len(names) and counts["trainable"] == sum(p.numel() for p in C.lora_parameters(model).values())
    assert all(n.startswith(C.SCOPE_PREFIX[scope]) for n, p in model.named_parameters() if p.requires_grad)
    assert sum(counts["by_component"].values()) == counts["trainable"]
    base.eval(), model.eval()
    ex = H.make_example(8)
    assert torch.equal(C.token_losses(base, ex), C.token_losses(model, ex))              # B = 0: the adapted model IS the base model
    other = H.tiny_model(seed=3, dropout=0.0)
    inject(other, scope)
    assert C.state_digest(C.trainable_state(model, scope)) == C.state_digest(C.trainable_state(other, scope))


@pytest.mark.parametrize("scope", ["adapter-lora", "adapter-full", "encoder-lora"])
def test_gradients_reach_only_the_scope_through_the_frozen_downstream_components(scope):
    model = H.tiny_model(seed=4, dropout=0.0)
    inject(model, scope)
    C.set_train_mode(model, scope)
    C.token_losses(model, H.make_example(9, entity_positions=[1])).mean().backward()
    mine = C.trainable_parameters(model, scope)
    assert all(p.grad is None for p in model.parameters() if not any(p is q for q in mine.values()))   # nothing frozen got a gradient (not the decoder, not the encoder, not the embeddings)
    flows = [n for n, p in mine.items() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    if scope == "adapter-full":
        assert sorted(flows) == sorted(mine)                              # every adapter tensor (the gradient lands on the FP32 master)
    else:
        assert sorted(flows) == sorted(n for n in mine if ".lora_B." in n)  # every injected layer's B (B = 0 at the start): the gradient really crossed the frozen adapter / decoder


@pytest.mark.parametrize("scope", ["adapter-lora", "adapter-full", "encoder-lora"])
def test_mode_policy_per_scope_keeps_everything_else_in_eval(scope):
    model = H.tiny_model(dropout=0.1)
    inject(model, scope)
    model.train()                                                       # a careless recursive train() switches dropout on everywhere
    with pytest.raises(AssertionError):
        C.check_mode_policy(model, True, scope)
    C.set_train_mode(model, scope)
    C.check_mode_policy(model, True, scope)
    assert not model.text_decoder.training and not model.speech_encoder.training and not model.ln_speech.training
    assert not model.speech_audio_adapter.mlp_adapter[2].training                           # the adapter's own native dropout is OFF in both adapter scopes (and in every other scope)
    if scope != "adapter-full":
        assert all(m.training for m in C.lora_dropouts(model).values()) and C.lora_dropouts(model)
    C.set_eval_mode(model)
    C.check_mode_policy(model, False, scope)


def test_native_adapter_dropout_off_does_not_switch_off_the_gradients_of_the_full_adapter():
    model = H.tiny_model(seed=2, dropout=0.5)                            # a large native dropout makes any leak of it obvious
    C.inject_adapter_full(model)
    C.set_train_mode(model, "adapter-full")
    C.check_mode_policy(model, True, "adapter-full")
    assert not any(m.training for m in model.modules())
    ex = H.make_example(7)
    a, b = C.token_losses(model, ex).detach(), C.token_losses(model, ex).detach()
    assert torch.equal(a, b)                                            # no stochastic dropout in the adapter during training
    masters = C.trainable_parameters(model, "adapter-full")
    C.token_losses(model, ex).mean().backward()
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith("speech_audio_adapter."))
    assert all(m.grad is not None and float(m.grad.abs().sum()) > 0 for m in masters.values())


# ---------------------------------------------------------------- the full adapter: BF16 computation, FP32 master weights
def test_adapter_full_trains_every_adapter_tensor_through_fp32_masters_and_nothing_else():
    model = H.tiny_model(seed=2, dropout=0.0)
    names = C.inject_adapter_full(model)
    assert names == sorted(["speech_audio_adapter.mlp_adapter.0.weight", "speech_audio_adapter.mlp_adapter.0.bias", "speech_audio_adapter.out_proj.weight", "speech_audio_adapter.out_proj.bias"])
    counts = C.check_trainable(model, names, "adapter-full")
    assert counts["trainable"] == 2 * (32 * 32 + 32) and counts["optimiser_state_mb"] == round(8 * counts["trainable"] / 2**20, 2)
    masters = C.trainable_parameters(model, "adapter-full")
    assert all(p.dtype == torch.float32 for p in masters.values()) and all(torch.equal(masters[n].data, model.get_parameter(n).data) for n in masters)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW(list(masters.values()), lr=3e-2, weight_decay=0.0)
    ex = H.make_example(10, seed=9)
    C.set_train_mode(model, "adapter-full")
    digest0 = C.state_digest(C.trainable_state(model, "adapter-full"))
    losses = []
    for _ in range(40):
        loss = C.token_losses(model, ex)
        loss.mean().backward()
        opt.step(), opt.zero_grad()
        C.sync_adapter_master(model)
        losses.append(float(loss.mean().detach()))
    assert losses[-1] < losses[0] and C.state_digest(C.trainable_state(model, "adapter-full")) != digest0
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    assert all(torch.equal(masters[n].data, model.get_parameter(n).data) for n in masters)           # the working weights are the (rounded) masters


def test_adapter_full_computes_in_bf16_never_fp32_and_folds_gradients_into_fp32_masters():
    model = H.tiny_model(seed=1, dropout=0.0).to(torch.bfloat16)
    C.inject_adapter_full(model)
    seen = {}

    def spy(name):
        def hook(module, args, out):
            seen[name] = (args[0].dtype, module.weight.dtype, out.dtype)
        return hook
    for name in ("mlp_adapter.0", "out_proj"):
        model.speech_audio_adapter.get_submodule(name).register_forward_hook(spy(name))
    ex = H.make_example(6)
    ex.input_features = ex.input_features.to(torch.bfloat16)
    C.set_train_mode(model, "adapter-full")
    C.token_losses(model, ex).mean().backward()
    assert seen == {"mlp_adapter.0": (torch.bfloat16,) * 3, "out_proj": (torch.bfloat16,) * 3}      # the Linear layers see BF16 input, BF16 weights and produce BF16: the inference computation
    masters = C.trainable_parameters(model, "adapter-full")
    assert all(p.dtype == torch.float32 for p in masters.values()) and all(model.get_parameter(n).dtype == torch.bfloat16 for n in masters)
    assert all(m.grad is not None and m.grad.dtype == torch.float32 for m in masters.values()) and all(model.get_parameter(n).grad is None for n in masters)
    once = {n: m.grad.clone() for n, m in masters.items()}
    for m in masters.values():
        m.grad = None
    for _ in range(2):                                                  # two micro-batches: summed in FP32
        C.token_losses(model, ex).mean().backward()
    assert all(torch.equal(masters[n].grad, 2 * once[n]) for n in masters)
    opt = torch.optim.AdamW(list(masters.values()), lr=1e-2, weight_decay=0.0)
    before = {n: model.get_parameter(n).detach().clone() for n in masters}
    opt.step()
    assert all(torch.equal(model.get_parameter(n), before[n]) for n in masters)                        # until the sync, the working weights are untouched
    C.sync_adapter_master(model)
    assert all(torch.equal(model.get_parameter(n).data, masters[n].data.to(torch.bfloat16)) for n in masters) and any(not torch.equal(model.get_parameter(n), before[n]) for n in masters)
    state = C.trainable_state(model, "adapter-full")
    other = H.tiny_model(seed=1, dropout=0.0).to(torch.bfloat16)
    C.inject_adapter_full(other)
    C.load_trainable_state(other, state, "adapter-full")
    assert C.state_digest(C.trainable_state(other, "adapter-full")) == C.state_digest(state)
    assert all(torch.equal(other.get_parameter(n), model.get_parameter(n)) for n in masters)           # loading syncs the working weights too
    with pytest.raises(ValueError):
        C.load_trainable_state(other, {"speech_audio_adapter.out_proj.weight": torch.zeros(32, 32)}, "adapter-full")
    model.speech_audio_adapter.float()                                  # an FP32 adapter in a BF16 model is exactly the deviation the guard refuses
    with pytest.raises(ValueError, match="BF16 computation"):
        C.check_trainable(model, [], "adapter-full")


# ---------------------------------------------------------------- encoder checkpointing
def test_encoder_checkpointing_gives_the_same_gradients_and_leaves_inference_alone():
    def grads(checkpointed):
        model = H.tiny_model(seed=6, dropout=0.0)
        inject(model, "encoder-lora", dropout=0.05)
        for n, p in C.lora_parameters(model).items():
            if ".lora_B." in n:
                p.data.normal_(0, 0.02)
        if checkpointed:
            assert C.enable_encoder_checkpointing(model) == 1 and C.enable_encoder_checkpointing(model) == 1   # idempotent
        C.set_train_mode(model, "encoder-lora")
        C.check_mode_policy(model, True, "encoder-lora")
        torch.manual_seed(5)                                            # same LoRA dropout masks in both runs (checkpoint restores the RNG when it recomputes)
        C.token_losses(model, H.make_example(9)).mean().backward()
        out = {n: p.grad.clone() for n, p in C.lora_parameters(model).items()}
        model.eval()
        with torch.no_grad():
            ce = C.token_losses(model, H.make_example(9))
        return out, ce
    plain, ce_plain = grads(False)
    wrapped, ce_wrapped = grads(True)
    assert plain.keys() == wrapped.keys() and any(float(g.abs().sum()) > 0 for g in plain.values())
    for n in plain:
        assert torch.allclose(plain[n], wrapped[n], atol=1e-7, rtol=1e-5), n
    assert torch.equal(ce_plain, ce_wrapped)


# ---------------------------------------------------------------- export: the declared scope is enforced
def base_dir(tmp_path, tensors):
    from safetensors.torch import save_file
    base = tmp_path / "base"
    base.mkdir()
    names = sorted(tensors)
    save_file({k: tensors[k] for k in names[:len(names) // 2]}, str(base / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file({k: tensors[k] for k in names[len(names) // 2:]}, str(base / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    weight_map = {k: ("model-00001-of-00002.safetensors" if i < len(names) // 2 else "model-00002-of-00002.safetensors") for i, k in enumerate(names)}
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for f in X.COPY_FILES:
        if not (base / f).exists():
            (base / f).write_text("{}")
    return base


def read_export(folder):
    from safetensors import safe_open
    out = {}
    for shard in sorted(Path(folder).glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            out.update({k: f.get_tensor(k) for k in f.keys()})
    return out


SHAPES = {"speech_audio_adapter.mlp_adapter.0.weight": (16, 12), "speech_audio_adapter.mlp_adapter.0.bias": (16,), "speech_audio_adapter.out_proj.weight": (8, 16),
          "speech_audio_adapter.out_proj.bias": (8,), "ln_speech.weight": (12,), "text_decoder.model.layers.0.self_attn.q_proj.weight": (8, 8),
          "text_decoder.model.layers.0.mlp.down_proj.weight": (8, 20), "speech_encoder.layers.0.fc1.weight": (20, 12), "speech_encoder.layers.0.fc2.weight": (12, 20)}


def small_base(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {n: torch.randn(s, generator=g).to(torch.bfloat16) for n, s in SHAPES.items()}
    return base_dir(tmp_path, tensors), tensors, g


def full_payload(g):
    return {n: torch.randn(s, generator=g) for n, s in SHAPES.items() if n.startswith("speech_audio_adapter.")}


def lora_payload(g, modules, r=4):
    out = {}
    for m in modules:
        o, i = SHAPES[f"{m}.weight"]
        out[f"{m}.lora_A"], out[f"{m}.lora_B"] = torch.randn(r, i, generator=g), torch.randn(o, r, generator=g) * 0.1
    return out


def save(tmp_path, payload, **meta):
    meta = {"digest": "abc", "run_id": "t", "epoch": 1, **meta}
    C.save_lora_safetensors(tmp_path / "ckpt.safetensors", payload, {k: v for k, v in meta.items() if v is not None})
    return tmp_path / "ckpt.safetensors"


def test_export_replaces_the_adapter_tensors_of_a_full_checkpoint_and_nothing_else(tmp_path):
    base, tensors, g = small_base(tmp_path)
    master = full_payload(g)
    path = save(tmp_path, master, scope="adapter-full")
    before = {p.name: X.sha256_file(p) for p in base.iterdir()}
    manifest = X.export(base, path, tmp_path / "out", "test-full")
    assert before == {p.name: X.sha256_file(p) for p in base.iterdir()}
    out = read_export(tmp_path / "out")
    assert sorted(out) == sorted(tensors)
    for n in SHAPES:
        expect = master[n].to(torch.bfloat16) if n in master else tensors[n]
        assert out[n].dtype == torch.bfloat16 and torch.equal(out[n], expect), n
    d = manifest["derivation"]
    assert d["scope"] == "adapter-full" and d["rank"] is None and d["merged_tensors"] == 4 and "master" in d["merge_dtype"]


@pytest.mark.parametrize("scope,modules", [("adapter-lora", ["speech_audio_adapter.mlp_adapter.0", "speech_audio_adapter.out_proj"]),
                                           ("encoder-lora", ["speech_encoder.layers.0.fc1", "speech_encoder.layers.0.fc2"]),
                                           ("decoder", ["text_decoder.model.layers.0.self_attn.q_proj", "text_decoder.model.layers.0.mlp.down_proj"])])
def test_export_merges_a_complete_lora_scope_exactly(tmp_path, scope, modules):
    base, tensors, g = small_base(tmp_path)
    payload = lora_payload(g, modules)
    path = save(tmp_path, payload, rank=4, alpha=8, scope=scope)
    manifest = X.export(base, path, tmp_path / "out", "t")
    out = read_export(tmp_path / "out")
    for n in SHAPES:
        m = n[:-len(".weight")] if n.endswith(".weight") else None
        expect = (tensors[n].float() + 2.0 * payload[f"{m}.lora_B"] @ payload[f"{m}.lora_A"]).to(torch.bfloat16) if m in modules else tensors[n]
        assert torch.equal(out[n], expect), n
    assert manifest["derivation"]["scope"] == scope and manifest["derivation"]["merged_tensors"] == len(modules)


def test_export_without_declared_scope_means_decoder_lora_as_before(tmp_path):
    base, tensors, g = small_base(tmp_path)
    path = save(tmp_path, lora_payload(g, ["text_decoder.model.layers.0.self_attn.q_proj", "text_decoder.model.layers.0.mlp.down_proj"]), rank=4, alpha=8)       # no `scope`: E01-E04 checkpoints
    assert X.export(base, path, tmp_path / "out", "t")["derivation"]["scope"] == "decoder"


def refused(tmp_path, payload, match, **meta):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base, tensors, g = small_base(tmp_path)
    path = save(tmp_path, payload(g) if callable(payload) else payload, **meta)
    with pytest.raises(SystemExit, match=match):
        X.export(base, path, tmp_path / "out", "bad")
    assert not (tmp_path / "out").exists()                              # nothing was written


def test_exporter_refuses_a_payload_that_is_not_exactly_the_declared_scope(tmp_path):
    q, down = "text_decoder.model.layers.0.self_attn.q_proj", "text_decoder.model.layers.0.mlp.down_proj"
    # the reproduced defect: declared adapter-full, payload is a decoder weight (it used to export silently, changing the decoder and leaving the adapter alone)
    refused(tmp_path / "1", lambda g: {"text_decoder.model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, generator=g)}, "outside the scope", scope="adapter-full")
    refused(tmp_path / "2", lambda g: {k: v for k, v in full_payload(g).items() if "out_proj.bias" not in k}, "missing", scope="adapter-full")                              # incomplete adapter
    refused(tmp_path / "3", lambda g: {**full_payload(g), "ln_speech.weight": torch.randn(12, generator=g)}, "outside the scope", scope="adapter-full")                 # adapter + something extra
    refused(tmp_path / "4", lambda g: {**full_payload(g), **lora_payload(g, [q])}, "mixed payload", scope="adapter-full")                                                # full + LoRA
    refused(tmp_path / "5", lambda g: lora_payload(g, [q]), "missing", scope="decoder", rank=4, alpha=8)                                                              # decoder LoRA covering 1 of 2 modules
    refused(tmp_path / "6", lambda g: lora_payload(g, [q, down, "speech_audio_adapter.out_proj"]), "outside the scope", scope="decoder", rank=4, alpha=8)               # decoder LoRA + an adapter module
    refused(tmp_path / "7", lambda g: lora_payload(g, [q, down]), "does not match its declared scope", scope="adapter-lora", rank=4, alpha=8)                            # decoder payload declared as adapter-lora
    refused(tmp_path / "8", lambda g: {**lora_payload(g, [q, down]), "ln_speech.weight": torch.randn(12, generator=g)}, "mixed payload", scope="decoder", rank=4, alpha=8)  # LoRA + a whole tensor
    refused(tmp_path / "9", lambda g: full_payload(g), "unknown scope", scope="everything")
    refused(tmp_path / "10", lambda g: lora_payload(g, [q, down], r=4), "do not fit", scope="decoder", rank=2, alpha=8)                                                  # rank in metadata disagrees with the matrices
    refused(tmp_path / "11", lambda g: {k: v.to(torch.bfloat16) for k, v in full_payload(g).items()}, "FP32", scope="adapter-full")
    refused(tmp_path / "12", lambda g: {k: torch.full_like(v, float("nan")) for k, v in full_payload(g).items()}, "finite", scope="adapter-full")
    refused(tmp_path / "13", lambda g: {**full_payload(g), "speech_audio_adapter.out_proj.weight": torch.zeros(3, 3)}, "do not fit", scope="adapter-full")                  # wrong shape


def test_real_decoder_checkpoint_still_passes_the_scope_validation():
    ckpt = Path(__file__).resolve().parents[2] / "evaluation-engine" / "runs" / "sft" / "runs" / "E01-LR25" / "checkpoints" / "epoch-2.safetensors"
    index = Path(__file__).resolve().parent / "test_data" / "meralion_index.json"
    if not ckpt.exists() or not index.exists():
        pytest.skip("local E01-LR25 checkpoint or the pinned index not present")
    weight_map = json.loads(index.read_text())["weight_map"]
    state, meta = C.load_lora_safetensors(ckpt)
    assert X.validate_payload(state, meta, weight_map) == "decoder" and len(X.expected_modules("decoder", weight_map)) == 182


# ---------------------------------------------------------------- the scope pre-flight
def stand_in_export_base(tmp_path, model):
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return base_dir(tmp_path, sd)


@pytest.mark.parametrize("scope", ["decoder", "adapter-lora", "adapter-full", "encoder-lora", "decoder+adapter", "decoder+adapter+encoder"])
def test_scope_check_without_an_export_is_incomplete_not_pass(tmp_path, scope):
    import sft_scope_check as SC
    import test_smoke as TS
    data, batch = TS.package(tmp_path)
    args = argparse.Namespace(scope=scope, data=data, model="unused", batch=batch, out=tmp_path / "o", export_dir=None, keep_export=False, export_id=None, base_manifest=None, manifest_out=None,
                              runtime_config_in=None, runtime_config_out=None, encoder_checkpointing=False, lr=6e-4, accumulation=4, stand_in=True)
    assert SC.check(args, (TS.stand_in(0), TS.Proc(), "prompt", R.fake_load_audio)) == 2
    report = json.loads((tmp_path / "o" / "scope_report.json").read_text())
    statuses = {k: c["status"] for k, c in report["checks"].items()}
    assert report["overall"] == "INCOMPLETE" and statuses.pop("export") == "skipped" and set(statuses.values()) == {"pass"} and set(statuses) == {"counts", "starts", "modes", "gradients", "updates", "reload"}
    assert not report["checks"]["export"]["ok"] and report["numbers"]["optimiser_state_bytes"] > 0


@pytest.mark.parametrize("scope", ["decoder", "adapter-lora", "adapter-full", "encoder-lora", "decoder+adapter", "decoder+adapter+encoder"])
def test_scope_check_with_an_export_passes_only_when_the_exported_model_reproduces_the_trained_one(tmp_path, scope):
    import sft_scope_check as SC
    import test_smoke as TS
    data, batch = TS.package(tmp_path)
    base = stand_in_export_base(tmp_path, TS.stand_in(0))
    exp = tmp_path / "exports" / "x-tmp"

    def load_export(path, device):
        model = TS.stand_in(0)
        model.load_state_dict(read_export(path), strict=True)
        return model, TS.Proc(), "prompt", R.fake_load_audio
    args = argparse.Namespace(scope=scope, data=data, model=base, batch=batch, out=tmp_path / "o", export_dir=exp, keep_export=False, export_id=None, base_manifest=None, manifest_out=None,
                              runtime_config_in=None, runtime_config_out=None, encoder_checkpointing=False, lr=6e-4, accumulation=4, stand_in=True)
    assert SC.check(args, (TS.stand_in(0), TS.Proc(), "prompt", R.fake_load_audio), load_stack=load_export) == 0
    report = json.loads((tmp_path / "o" / "scope_report.json").read_text())
    assert report["overall"] == "PASS" and all(c["status"] == "pass" for c in report["checks"].values()) and not exp.exists()
    effect = report["checks"]["reload"]["detail"]["effect_vs_base"]["mean_abs_token_ce_diff"]
    assert effect > 0                                                   # the exported model reproduces a model that really differs from the base


def test_scope_check_fails_when_the_exported_model_does_not_match(tmp_path):
    import sft_scope_check as SC
    import test_smoke as TS
    data, batch = TS.package(tmp_path)
    base = stand_in_export_base(tmp_path, TS.stand_in(0))

    def wrong_export(path, device):
        return TS.stand_in(0), TS.Proc(), "prompt", R.fake_load_audio       # ignores the export: still the untrained base model
    args = argparse.Namespace(scope="adapter-lora", data=data, model=base, batch=batch, out=tmp_path / "o", export_dir=tmp_path / "exports" / "y-tmp", keep_export=True, export_id=None, base_manifest=None,
                              manifest_out=None, runtime_config_in=None, runtime_config_out=None, encoder_checkpointing=False, lr=6e-2, accumulation=4, stand_in=True)
    assert SC.check(args, (TS.stand_in(0), TS.Proc(), "prompt", R.fake_load_audio), load_stack=wrong_export) == 1
    assert json.loads((tmp_path / "o" / "scope_report.json").read_text())["checks"]["export"]["status"] == "fail" and (tmp_path / "exports" / "y-tmp").exists()      # --keep-export keeps it


# ---------------------------------------------------------------- stamping predictions of a full-adapter run
def test_prediction_stamp_records_the_scope_and_no_rank_for_a_full_adapter(tmp_path):
    import stamp_sft_predictions as ST
    run = tmp_path / "E06-FULL"
    run.mkdir()
    config = {"run_id": "E06-FULL", "scope": "adapter-full", "rank": 8, "alpha": 16, "lr": 2e-5, "objective": "ce", "entity_weight": 2.0, "sampling": "uniform", "k": 1067, "seed": 42}
    (run / "config.json").write_text(json.dumps({"config": config}))
    (run / "epoch_summary.jsonl").write_text(json.dumps({"epoch": 2, "checkpoint": "epoch-2.safetensors", "checkpoint_digest": "dd"}) + "\n")
    src = tmp_path / "pred"
    src.mkdir()
    (src / "CALL_001.json").write_text(json.dumps({"call_id": "CALL_001", "settings": {"adapter": None, "training": False}}))
    n, adapter = ST.stamp(src, tmp_path / "stamped", run, 2)
    doc = json.loads((tmp_path / "stamped" / "CALL_001.json").read_text())
    assert n == 1 and doc["settings"]["training"] is True and doc["settings"]["adapter"] == adapter
    assert adapter["method"] == "full-adapter" and adapter["scope"] == "adapter-full" and adapter["rank"] is None and adapter["alpha"] is None and adapter["checkpoint_digest"] == "dd"
    config.pop("scope")                                                  # a run from before scopes existed is decoder LoRA
    (run / "config.json").write_text(json.dumps({"config": config}))
    _n, old = ST.stamp(src, tmp_path / "stamped2", run, 2)
    assert old["method"] == "lora" and old["scope"] == "decoder" and old["rank"] == 8


# ---------------------------------------------------------------- end to end through the shared trainer
def frozen_unchanged(model, before, scope):
    prefix = C.SCOPE_PREFIX[scope]
    for k, v in model.state_dict().items():
        if "lora_" in k or "base_layer" in k:
            continue
        key = k.replace(".base_layer.", ".")
        if C.has_adapter_full(scope) and key.startswith("speech_audio_adapter."):
            continue
        assert torch.equal(before[key], v), key


@pytest.mark.parametrize("scope", ["adapter-lora", "adapter-full", "encoder-lora", "decoder+adapter", "decoder+adapter+encoder"])
def test_trainer_runs_each_scope_logs_it_and_changes_nothing_outside_it(tmp_path, scope):
    data, batch, n, m = R.make_package(tmp_path)
    model = R.stand_in_model(0)
    before = {k.replace(".base_layer.", "."): v.clone() for k, v in model.state_dict().items()}
    extra = ("--scope", scope) + (("--encoder-checkpointing",) if scope == "encoder-lora" else ())
    T.run(R.args_for(tmp_path, data, batch, "S", k=m, objective="ce", sampling="uniform", extra=extra), (model, R.FakeProcessor(), "prompt", R.fake_load_audio))
    out = tmp_path / "runs" / "S"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["config"]["scope"] == scope and cfg["parameters"]["scope"] == scope and cfg["parameters"]["trainable"] > 0
    rows = [json.loads(l) for l in (out / "train_metrics.jsonl").read_text().splitlines()]
    assert rows and all(r["grad_norm"] > 0 and r["ce"] > 0 for r in rows)                          # a gradient really reached the scope's parameters
    state, meta = C.load_lora_safetensors(out / "checkpoints" / "epoch-2.safetensors")
    lora_part = [k for k in state if ".lora_" in k]
    assert meta["scope"] == scope and (not lora_part if scope == "adapter-full" else bool(lora_part) and (len(state) > len(lora_part) if scope in C.JOINT_SCOPES else len(lora_part) == len(state)))
    frozen_unchanged(model, before, scope)
    changed = [k for k, v in model.state_dict().items() if k.replace(".base_layer.", ".") in before and not torch.equal(before[k.replace(".base_layer.", ".")], v) and "lora_" not in k]
    assert bool(changed) == C.has_adapter_full(scope)                                              # only a trained adapter rewrites base tensors; LoRA runs leave every base tensor alone
    if C.has_adapter_full(scope):                                                                  # the saved masters round to exactly the working weights the validation used
        assert all(torch.equal(model.get_parameter(k).data, state[k].to(model.get_parameter(k).dtype)) for k in state if ".lora_" not in k)


def test_epoch0_must_reproduce_the_reference_within_the_declared_tolerance(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    T.run(R.args_for(tmp_path, data, batch, "REF", k=m, objective="ce", sampling="uniform"), R.stack())
    ref = tmp_path / "runs" / "REF" / "epoch_summary.jsonl"
    T.run(R.args_for(tmp_path, data, batch, "OK", k=m, objective="ce", sampling="uniform", extra=("--scope", "adapter-full", "--epoch0-reference", str(ref))), R.stack())
    row = [json.loads(l) for l in (tmp_path / "runs" / "OK" / "epoch_summary.jsonl").read_text().splitlines()][0]
    assert row["epoch0_agreement"]["passed"] and max(row["epoch0_agreement"]["absolute_difference_nats"].values()) <= 1e-6       # BF16 computation: the untouched adapter reproduces the base model
    rows = [json.loads(l) for l in ref.read_text().splitlines()]
    rows[0]["validation"]["ce"] += 0.5
    bad = tmp_path / "bad_reference.jsonl"
    bad.write_text("".join(json.dumps(r) + chr(10) for r in rows))
    with pytest.raises(SystemExit, match="epoch-0 validation differs"):
        T.run(R.args_for(tmp_path, data, batch, "BAD", k=m, objective="ce", sampling="uniform", extra=("--scope", "adapter-lora", "--epoch0-reference", str(bad))), R.stack())
    evidence = [json.loads(l) for l in (tmp_path / "runs" / "BAD" / "epoch_summary.jsonl").read_text().splitlines()][0]
    assert evidence["epoch0_agreement"]["passed"] is False and not (tmp_path / "runs" / "BAD" / "checkpoints" / "epoch-1.safetensors").exists()      # failed evidence is preserved, no training happened


def test_encoder_checkpointing_flag_is_refused_for_other_scopes(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    with pytest.raises(SystemExit, match="encoder-lora"):
        T.run(R.args_for(tmp_path, data, batch, "Z", k=m, objective="ce", sampling="uniform", extra=("--scope", "adapter-full", "--encoder-checkpointing")), R.stack())


@pytest.mark.parametrize("scope", ["adapter-full", "decoder+adapter", "decoder+adapter+encoder"])
def test_adapter_full_interrupted_run_resumes_to_the_same_weights_as_a_continuous_run(tmp_path, monkeypatch, scope):
    data, batch, n, m = R.make_package(tmp_path)
    extra = ("--scope", scope)
    T.run(R.args_for(tmp_path, data, batch, "A", k=m, objective="ce", sampling="uniform", extra=extra), R.stack())
    calls, original = {"n": 0}, T.train_group

    def interrupting(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt
        return original(*a, **k)
    monkeypatch.setattr(T, "train_group", interrupting)
    with pytest.raises(KeyboardInterrupt):
        T.run(R.args_for(tmp_path, data, batch, "B", k=m, objective="ce", sampling="uniform", extra=extra), R.stack())
    monkeypatch.setattr(T, "train_group", original)
    T.run(R.args_for(tmp_path, data, batch, "B", k=m, objective="ce", sampling="uniform", extra=extra + ("--resume",)), R.stack())
    assert R.digest_of(tmp_path, "A", 2) == R.digest_of(tmp_path, "B", 2) and R.digest_of(tmp_path, "A", 1) == R.digest_of(tmp_path, "B", 1)


# ---------------------------------------------------------------- decoder LoRA + full adapter in one run
DEC = ["text_decoder.model.layers.0.self_attn.q_proj", "text_decoder.model.layers.0.mlp.down_proj"]


def test_joint_scope_trains_decoder_lora_and_the_whole_adapter_and_nothing_else():
    base = H.tiny_model(seed=3, dropout=0.0)
    model = H.tiny_model(seed=3, dropout=0.0)
    targets = C.inject_joint(model, 8, 16, 0.05, 42)
    assert len(targets) == 14 + 4 and targets[:14] == C.expected_targets(H.tiny_model(), "decoder")
    counts = C.check_trainable(model, targets, "decoder+adapter")
    lora_n, adapter_n = sum(p.numel() for p in C.lora_parameters(model).values()), 2 * (32 * 32 + 32)
    assert counts["trainable"] == lora_n + adapter_n and counts["by_component"] == {"decoder_lora": lora_n, "adapter_full": adapter_n}
    mine = C.trainable_parameters(model, "decoder+adapter")
    assert set(mine) == set(C.lora_parameters(model)) | set(model.speech_audio_adapter._fp32_master) and all(p.dtype == torch.float32 for p in mine.values())
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == sorted(mine)
    base.eval(), model.eval()
    ex = H.make_example(8)
    assert torch.equal(C.token_losses(base, ex), C.token_losses(model, ex))              # B = 0 and the adapter untouched: the model IS the base model
    C.set_train_mode(model, "decoder+adapter")
    C.check_mode_policy(model, True, "decoder+adapter")
    assert not model.speech_audio_adapter.mlp_adapter[2].training and all(m.training for m in C.lora_dropouts(model).values()) and C.lora_dropouts(model)
    model.train()
    with pytest.raises(AssertionError):
        C.check_mode_policy(model, True, "decoder+adapter")


def test_joint_scope_gradients_reach_every_lora_b_and_every_adapter_master_only():
    model = H.tiny_model(seed=4, dropout=0.0)
    C.inject_joint(model, 8, 16, 0.05, 42)
    C.set_train_mode(model, "decoder+adapter")
    C.token_losses(model, H.make_example(9, entity_positions=[1])).mean().backward()
    mine = C.trainable_parameters(model, "decoder+adapter")
    assert all(p.grad is None for p in model.parameters() if not any(p is q for q in mine.values()))
    flows = sorted(n for n, p in mine.items() if p.grad is not None and float(p.grad.abs().sum()) > 0)
    assert flows == sorted(n for n in mine if ".lora_B." in n or n.startswith("speech_audio_adapter."))


def test_joint_state_roundtrip_and_a_mismatched_state_is_refused():
    model = H.tiny_model(seed=1, dropout=0.0)
    C.inject_joint(model, 8, 16, 0.0, 42)
    for p in C.trainable_parameters(model, "decoder+adapter").values():
        p.data.normal_(0, 0.1)
    C.sync_adapter_master(model)
    state = C.trainable_state(model, "decoder+adapter")
    assert any(".lora_" in k for k in state) and any(".lora_" not in k for k in state)
    other = H.tiny_model(seed=1, dropout=0.0)
    C.inject_joint(other, 8, 16, 0.0, 99)
    C.load_trainable_state(other, state, "decoder+adapter")
    assert C.state_digest(C.trainable_state(other, "decoder+adapter")) == C.state_digest(state)
    ex = H.make_example(7)
    model.eval(), other.eval()
    assert torch.equal(C.token_losses(model, ex), C.token_losses(other, ex))                         # the working adapter weights followed the loaded masters too
    with pytest.raises(ValueError):
        C.load_trainable_state(other, {k: v for k, v in state.items() if ".lora_" in k}, "decoder+adapter")


def joint_payload(g, r=4):
    return {**lora_payload(g, DEC, r=r), **full_payload(g)}


def test_export_merges_the_lora_and_replaces_the_adapter_tensors_of_a_joint_checkpoint(tmp_path):
    base, tensors, g = small_base(tmp_path)
    payload = joint_payload(g)
    path = save(tmp_path, payload, rank=4, alpha=8, scope="decoder+adapter")
    manifest = X.export(base, path, tmp_path / "out", "joint")
    out = read_export(tmp_path / "out")
    for n in SHAPES:
        m = n[:-len(".weight")] if n.endswith(".weight") else None
        if m in DEC:
            expect = (tensors[n].float() + 2.0 * payload[f"{m}.lora_B"] @ payload[f"{m}.lora_A"]).to(torch.bfloat16)
        elif n.startswith("speech_audio_adapter."):
            expect = payload[n].to(torch.bfloat16)
        else:
            expect = tensors[n]
        assert torch.equal(out[n], expect), n
    d = manifest["derivation"]
    assert d["scope"] == "decoder+adapter" and d["rank"] == 4 and d["merged_tensors"] == 2 + 4 and "adapter FP32 masters rounded" in d["merge_dtype"]


def test_exporter_refuses_an_incomplete_or_misfiled_joint_checkpoint(tmp_path):
    refused(tmp_path / "1", lambda g: {k: v for k, v in joint_payload(g).items() if not k.startswith("speech_audio_adapter.out_proj.bias")}, "missing", scope="decoder+adapter", rank=4, alpha=8)
    refused(tmp_path / "2", lambda g: {k: v for k, v in joint_payload(g).items() if ".lora_" not in k or DEC[0] in k}, "missing", scope="decoder+adapter", rank=4, alpha=8)           # one decoder module dropped
    refused(tmp_path / "3", lambda g: full_payload(g), "missing", scope="decoder+adapter", rank=4, alpha=8)                                                                        # adapter only, declared joint
    refused(tmp_path / "4", lambda g: lora_payload(g, DEC), "missing", scope="decoder+adapter", rank=4, alpha=8)                                                                    # LoRA only, declared joint
    refused(tmp_path / "5", lambda g: {**joint_payload(g), "ln_speech.weight": torch.randn(12, generator=g)}, "outside the scope", scope="decoder+adapter", rank=4, alpha=8)
    refused(tmp_path / "6", lambda g: joint_payload(g), "mixed payload", scope="adapter-full")                                                                                      # joint payload declared as adapter-full
    refused(tmp_path / "7", lambda g: joint_payload(g), "mixed payload", scope="decoder", rank=4, alpha=8)                                                                          # joint payload declared as decoder LoRA


def test_joint_prediction_stamp_records_both_parts(tmp_path):
    import stamp_sft_predictions as ST
    run = tmp_path / "J"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"config": {"run_id": "J", "scope": "decoder+adapter", "rank": 8, "alpha": 16, "lr": 2.5e-5, "objective": "ce", "entity_weight": 2.0, "sampling": "uniform", "k": 1067, "seed": 42}}))
    (run / "epoch_summary.jsonl").write_text(json.dumps({"epoch": 2, "checkpoint": "epoch-2.safetensors", "checkpoint_digest": "dd"}) + chr(10))
    src = tmp_path / "pred"
    src.mkdir()
    (src / "CALL_001.json").write_text(json.dumps({"call_id": "CALL_001", "settings": {"adapter": None, "training": False}}))
    _n, adapter = ST.stamp(src, tmp_path / "stamped", run, 2)
    assert adapter["method"] == "lora+full-adapter" and adapter["scope"] == "decoder+adapter" and adapter["rank"] == 8 and adapter["alpha"] == 16


# ---------------------------------------------------------------- decoder LoRA + full adapter + encoder LoRA in one run
ENC2 = ["speech_encoder.layers.0.fc1", "speech_encoder.layers.0.fc2"]
ALL3 = "decoder+adapter+encoder"


def test_three_component_scope_adds_encoder_lora_to_the_joint_scope_and_nothing_else():
    base = H.tiny_model(seed=3, dropout=0.0)
    model = H.tiny_model(seed=3, dropout=0.0)
    targets = C.inject_joint(model, 8, 16, 0.05, 42, encoder=True)
    fresh = H.tiny_model()
    assert targets[:14] == C.expected_targets(fresh, "decoder") and targets[14:20] == C.expected_targets(fresh, "encoder-lora") and len(targets) == 14 + 6 + 4
    counts = C.check_trainable(model, targets, ALL3)
    comp = counts["by_component"]
    assert set(comp) == {"decoder_lora", "adapter_full", "encoder_lora"} and sum(comp.values()) == counts["trainable"] and comp["encoder_lora"] > 0
    mine = C.trainable_parameters(model, ALL3)
    assert set(mine) == set(C.lora_parameters(model)) | set(model.speech_audio_adapter._fp32_master) and all(p.dtype == torch.float32 for p in mine.values())
    assert any(n.startswith("speech_encoder.") for n in mine) and any(n.startswith("text_decoder.") for n in mine)
    base.eval(), model.eval()
    ex = H.make_example(8)
    assert torch.equal(C.token_losses(base, ex), C.token_losses(model, ex))              # B = 0 everywhere, adapter untouched: the model IS the base model
    C.set_train_mode(model, ALL3)
    C.check_mode_policy(model, True, ALL3)
    drops = C.lora_dropouts(model)
    assert any(n.startswith("speech_encoder.") for n in drops) and any(n.startswith("text_decoder.") for n in drops) and all(m.training for m in drops.values())
    assert not model.speech_audio_adapter.mlp_adapter[2].training
    other = H.tiny_model(seed=3, dropout=0.0)
    C.inject_joint(other, 8, 16, 0.05, 42, encoder=True)
    assert C.state_digest(C.trainable_state(model, ALL3)) == C.state_digest(C.trainable_state(other, ALL3))     # deterministic initialisation


def test_three_component_gradients_reach_decoder_encoder_and_adapter_only():
    model = H.tiny_model(seed=4, dropout=0.0)
    C.inject_joint(model, 8, 16, 0.05, 42, encoder=True)
    C.set_train_mode(model, ALL3)
    C.token_losses(model, H.make_example(9, entity_positions=[1])).mean().backward()
    mine = C.trainable_parameters(model, ALL3)
    assert all(p.grad is None for p in model.parameters() if not any(p is q for q in mine.values()))
    flows = sorted(n for n, p in mine.items() if p.grad is not None and float(p.grad.abs().sum()) > 0)
    assert flows == sorted(n for n in mine if ".lora_B." in n or n.startswith("speech_audio_adapter."))         # every LoRA B of both components and every adapter master
    state = C.trainable_state(model, ALL3)
    other = H.tiny_model(seed=4, dropout=0.0)
    C.inject_joint(other, 8, 16, 0.0, 7, encoder=True)
    C.load_trainable_state(other, state, ALL3)
    assert C.state_digest(C.trainable_state(other, ALL3)) == C.state_digest(state)


def test_export_of_the_three_component_checkpoint_merges_both_loras_and_replaces_the_adapter(tmp_path):
    base, tensors, g = small_base(tmp_path)
    payload = {**lora_payload(g, DEC + ENC2), **full_payload(g)}
    path = save(tmp_path, payload, rank=4, alpha=8, scope=ALL3)
    manifest = X.export(base, path, tmp_path / "out", "all3")
    out = read_export(tmp_path / "out")
    for n in SHAPES:
        m = n[:-len(".weight")] if n.endswith(".weight") else None
        if m in DEC + ENC2:
            expect = (tensors[n].float() + 2.0 * payload[f"{m}.lora_B"] @ payload[f"{m}.lora_A"]).to(torch.bfloat16)
        elif n.startswith("speech_audio_adapter."):
            expect = payload[n].to(torch.bfloat16)
        else:
            expect = tensors[n]
        assert torch.equal(out[n], expect), n
    assert manifest["derivation"]["scope"] == ALL3 and manifest["derivation"]["merged_tensors"] == 2 + 2 + 4


def test_exporter_refuses_incomplete_or_misdeclared_three_component_checkpoints(tmp_path):
    full = lambda g: {**lora_payload(g, DEC + ENC2), **full_payload(g)}                                                   # noqa: E731
    refused(tmp_path / "1", lambda g: {k: v for k, v in full(g).items() if "layers.0.fc2" not in k}, "missing", scope=ALL3, rank=4, alpha=8)                                       # an encoder module dropped
    refused(tmp_path / "2", lambda g: {**lora_payload(g, DEC), **full_payload(g)}, "missing", scope=ALL3, rank=4, alpha=8)                                                          # no encoder part at all
    refused(tmp_path / "3", lambda g: full(g), "outside the scope", scope="decoder+adapter", rank=4, alpha=8)                                                                       # encoder LoRA in a decoder+adapter checkpoint
    refused(tmp_path / "4", lambda g: {k: v for k, v in full(g).items() if not k.startswith("speech_audio_adapter.")}, "missing", scope=ALL3, rank=4, alpha=8)                    # adapter dropped
