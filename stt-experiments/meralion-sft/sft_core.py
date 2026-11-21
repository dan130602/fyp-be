"""Model-facing building blocks of the MERaLiON SFT trainer (decoder LoRA, CE / entity-weighted CE, mode policy, checkpoints).

Verified against the pinned `modeling_meralion3.py` (revision 9f226d7), not assumed:
  * forward(input_ids, attention_mask, input_features, feature_attention_mask, labels=None, ...) builds inputs_embeds = text embeddings with every
    <SpeechHere> position (id 255999) replaced by the adapter output, then calls Gemma2ForCausalLM; with labels it returns HF's own shifted mean loss.
    We never pass labels: per-token losses are computed here from the logits, shifted exactly once (logits[p-1] scores token p).
  * the speech encoder is a bare WhisperEncoder: SpecAugment (`apply_spec_augment`) lives in WhisperModel and is NOT applied; the adapter has
    Dropout(0.01) in mlp_adapter, so frozen components are kept in eval mode (set_train_mode) -- frozen parameters do not switch dropout off.
  * the forward reads `self.text_decoder.base_model.embed_tokens`; a PeftModel wrapper would move that path, so LoRA is INJECTED in place
    (peft.inject_adapter_in_model) with an anchored regex that cannot match encoder or adapter modules.
"""
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SPEECH_TOKEN_ID = 255999
DECODER_TARGET_PATTERN = r"text_decoder\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
ENCODER_TARGET_PATTERN = r"speech_encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"          # Whisper names: out_proj (the decoder's is o_proj), fc1/fc2
FROZEN_MODULES = ("speech_encoder", "ln_speech", "speech_audio_adapter")
SCOPES = ("decoder", "adapter-lora", "adapter-full", "encoder-lora", "decoder+adapter", "decoder+adapter+encoder")        # what a run trains; everything else stays frozen
JOINT_SCOPES = ("decoder+adapter", "decoder+adapter+encoder")                                                              # LoRA part(s) AND the full adapter (FP32 masters) in one run
SCOPE_PREFIX = {"decoder": "text_decoder.model.layers.", "adapter-lora": "speech_audio_adapter.", "adapter-full": "speech_audio_adapter.", "encoder-lora": "speech_encoder.layers.",
                "decoder+adapter": ("text_decoder.model.layers.", "speech_audio_adapter."),
                "decoder+adapter+encoder": ("text_decoder.model.layers.", "speech_audio_adapter.", "speech_encoder.layers.")}
SCOPE_GROUPS = {"decoder": ("self_attn", "mlp"), "encoder-lora": ("self_attn", "fc1", "fc2"), "adapter-lora": ("mlp_adapter", "out_proj"), "adapter-full": ("mlp_adapter", "out_proj"), "decoder+adapter": (), "decoder+adapter+encoder": ()}


def has_adapter_full(scope):
    """The scope trains the whole speech adapter (FP32 masters, BF16 computation): adapter-full, and decoder+adapter (decoder LoRA and the full adapter together)."""
    return scope == "adapter-full" or scope in JOINT_SCOPES
NA = None


# ------------------------------------------------------------------ LoRA (scoped injection)
def scope_pattern(model, scope):
    """The anchored regex of the Linear layers one LoRA scope covers: decoder = the 182 attention + MLP projections; encoder-lora = q/k/v/out_proj + fc1/fc2 of every encoder layer;
    adapter-lora = every Linear of the speech adapter (read from the model itself, so a different adapter shape needs no change here)."""
    if scope == "decoder":
        return DECODER_TARGET_PATTERN
    if scope == "encoder-lora":
        return ENCODER_TARGET_PATTERN
    if scope == "adapter-lora":
        linears = sorted(f"speech_audio_adapter.{n}" for n, m in model.speech_audio_adapter.named_modules() if isinstance(m, torch.nn.Linear))
        if not linears:
            raise ValueError("the speech adapter has no Linear layers")
        return "|".join(re.escape(n) for n in linears)
    raise ValueError(f"no LoRA pattern for scope {scope!r}")


def expected_targets(model, scope="decoder"):
    pattern = scope_pattern(model, scope)
    names = [n for n, m in model.named_modules() if re.fullmatch(pattern, n)]
    bad = [n for n in names if not isinstance(model.get_submodule(n), torch.nn.Linear)]
    if bad or not names:
        raise ValueError(f"unexpected {scope} LoRA targets: {bad or 'none found'}")
    return sorted(names)


def expected_decoder_targets(model):
    return expected_targets(model, "decoder")


def inject_lora(model, scope="decoder", rank=8, alpha=16, dropout=0.05, seed=42, freeze=True):
    """Freeze everything, inject LoRA into the Linear layers of ONE scope only (anchored regex: it cannot match another component), FP32 adapters, deterministic init. Returns the target names."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer
    targets = expected_targets(model, scope)
    before = {n for n, m in model.named_modules() if isinstance(m, LoraLayer)}                       # LoRA already injected by an earlier call (joint scopes)
    if freeze:
        model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=scope_pattern(model, scope)), model, adapter_name="default")
    injected = sorted({n for n, m in model.named_modules() if isinstance(m, LoraLayer)} - before)
    if injected != targets:
        raise ValueError(f"LoRA landed on {len(injected)} modules, expected the {len(targets)} {scope} projections: {sorted(set(injected) ^ set(targets))[:6]}")
    generator = torch.Generator().manual_seed(seed)
    for name in targets:                                     # explicit FP32 re-initialisation in a fixed order: identical for every run with the same seed
        layer = model.get_submodule(name)
        a, b = layer.lora_A["default"], layer.lora_B["default"]
        a.float(), b.float()
        bound = math.sqrt(6.0 / ((1 + 5) * a.weight.shape[1]))   # kaiming_uniform_(a=sqrt(5)) == U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        a.weight.data = (torch.rand(a.weight.shape, generator=generator, dtype=torch.float32) * 2 - 1).mul_(bound).to(a.weight.device)
        b.weight.data = torch.zeros_like(b.weight, dtype=torch.float32)
        a.weight.requires_grad_(True), b.weight.requires_grad_(True)
    return targets


def inject_decoder_lora(model, rank=8, alpha=16, dropout=0.05, seed=42):
    return inject_lora(model, "decoder", rank, alpha, dropout, seed)


def inject_adapter_full(model, freeze=True):
    """Full fine-tuning of the speech adapter with BF16 COMPUTATION (the forward and backward passes run on the BF16 weights, exactly as inference does) and FP32 MASTER weights
    (the standard mixed-precision recipe): the adapter keeps its pretrained BF16 parameters as the working weights; an FP32 copy of each is what AdamW updates. A hook folds every BF16 gradient
    into the master's FP32 gradient as soon as autograd has accumulated it (so the 8 micro-batches of a step are summed in FP32), and sync_adapter_master() rounds the updated masters back into the
    working weights after each optimiser step. Everything else is frozen. Returns the parameter names."""
    if freeze:
        model.requires_grad_(False)
    adapter = model.speech_audio_adapter
    master = {}
    for name, p in adapter.named_parameters():
        p.requires_grad_(True)
        m = torch.nn.Parameter(p.detach().float().clone())               # starts as the exact pretrained values

        def fold(param, m=m):
            g = param.grad.float()
            m.grad = g if m.grad is None else m.grad + g
            param.grad = None
        p.register_post_accumulate_grad_hook(fold)
        master[f"speech_audio_adapter.{name}"] = m
    adapter._fp32_master = master                                        # a plain dict: the masters are not module parameters (they are never part of the forward pass)
    return sorted(master)


def inject_joint(model, rank=8, alpha=16, dropout=0.05, seed=42, encoder=False):
    """decoder LoRA (the 182 attention + MLP projections, exactly as E01) AND the full speech adapter (BF16 computation, FP32 masters) trained together; with encoder=True also LoRA on the
    speech encoder's attention + MLP projections (192). Everything else frozen. Returns the LoRA target names followed by the adapter parameter names."""
    model.requires_grad_(False)
    lora_targets = inject_lora(model, "decoder", rank, alpha, dropout, seed, freeze=False)
    if encoder:
        lora_targets += inject_lora(model, "encoder-lora", rank, alpha, dropout, seed, freeze=False)
    return lora_targets + inject_adapter_full(model, freeze=False)


def sync_adapter_master(model):
    """Round the FP32 master weights into the BF16 working weights (what the forward pass, validation and the exported model use). Call after every optimiser step and after loading a state."""
    with torch.no_grad():
        for name, m in model.speech_audio_adapter._fp32_master.items():
            model.get_parameter(name).copy_(m)


def enable_encoder_checkpointing(model):
    """Gradient checkpointing for the speech encoder's layers (memory for compute): each layer is recomputed in the backward pass. Wraps the layer's forward instead of switching the encoder to train
    mode (the eval-mode policy is untouched, and no layer-drop draws are made); torch's checkpoint restores the RNG state on recompute, so LoRA dropout masks are identical in both passes.
    Only active while gradients are being recorded; validation and inference run the original forward."""
    from torch.utils.checkpoint import checkpoint
    layers = list(model.speech_encoder.layers)
    for layer in layers:
        if getattr(layer, "_checkpointed", False):
            continue
        inner = layer.forward

        def forward(*args, _inner=inner, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_inner, *args, use_reentrant=False, **kwargs)
            return _inner(*args, **kwargs)
        layer.forward = forward
        layer._checkpointed = True
    return len(layers)


def lora_parameters(model):
    return {n: p for n, p in model.named_parameters() if ".lora_A." in n or ".lora_B." in n}


def trainable_parameters(model, scope="decoder"):
    """{name: parameter} of what the scope trains: the LoRA matrices, and/or (adapter-full, decoder+adapter) the FP32 master weights of every speech-adapter parameter."""
    if scope == "adapter-full":
        return dict(model.speech_audio_adapter._fp32_master)                                 # the FP32 master weights (inject_adapter_full): what the optimiser updates and the checkpoint stores
    if scope in JOINT_SCOPES:
        return {**lora_parameters(model), **model.speech_audio_adapter._fp32_master}
    return lora_parameters(model)


def joint_components(mine, scope):
    out = {"decoder_lora": sum(p.numel() for name, p in mine.items() if name.startswith("text_decoder.")), "adapter_full": sum(p.numel() for name, p in mine.items() if name.startswith("speech_audio_adapter."))}
    if "encoder" in scope:
        out["encoder_lora"] = sum(p.numel() for name, p in mine.items() if name.startswith("speech_encoder."))
    return out


def check_trainable(model, targets, scope="decoder"):
    """Exactly the scope's parameters are trainable (the LoRA matrices of its layers, or the whole speech adapter), all FP32, and nothing outside its component
    (decoder scope: nothing in the encoder, ln_speech, adapter, embeddings or head)."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    mine = trainable_parameters(model, scope)
    if sorted(trainable) != sorted(mine):
        raise ValueError(f"trainable set differs from the {scope} set: {sorted(set(trainable) ^ set(mine))[:6]}")
    if any(not n.startswith(SCOPE_PREFIX[scope]) for n in trainable) or any(p.dtype != torch.float32 for p in mine.values()):
        raise ValueError(f"trainable parameters outside the {scope} component, or not FP32")
    if has_adapter_full(scope):
        working = model.text_decoder.model.embed_tokens.weight.dtype
        wrong = [n for n in mine if n.startswith("speech_audio_adapter.") and model.get_parameter(n).dtype != working]
        if wrong:
            raise ValueError(f"the adapter's working weights must stay {working} (BF16 computation; only the masters are FP32): {wrong[:3]}")
    n = sum(p.numel() for p in mine.values())
    total = sum(p.numel() for p in model.parameters())
    return {"scope": scope, "trainable": n, "total": total, "trainable_percent": round(100.0 * n / total, 4), "lora_modules": len(targets),
            "by_component": (joint_components(mine, scope) if scope in JOINT_SCOPES else {c: sum(p.numel() for name, p in mine.items() if c in name) for c in SCOPE_GROUPS[scope]}),
            "optimiser_state_mb": round(8 * n / 2**20, 2), "fp32_weights_and_grads_mb": round(8 * n / 2**20, 2)}      # AdamW exp_avg + exp_avg_sq, and FP32 weight + gradient: 4 bytes each per trainable parameter


def lora_state(model):
    """{'<module>.lora_A': tensor, '<module>.lora_B': tensor} (FP32, CPU): the lightweight checkpoint."""
    out = {}
    for n, p in lora_parameters(model).items():
        module, rest = n.split(".lora_", 1)
        out[f"{module}.lora_{rest.split('.')[0]}"] = p.detach().float().cpu().contiguous()
    return out


def load_lora_state(model, state):
    params = lora_parameters(model)
    keyed = {}
    for n, p in params.items():
        module, rest = n.split(".lora_", 1)
        keyed[f"{module}.lora_{rest.split('.')[0]}"] = p
    if sorted(keyed) != sorted(state):
        raise ValueError("LoRA checkpoint keys differ from the model's")
    with torch.no_grad():
        for key, p in keyed.items():
            if tuple(state[key].shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch for {key}")
            p.copy_(state[key].to(p.device, p.dtype))


def trainable_state(model, scope="decoder"):
    """The checkpoint payload (FP32, CPU): LoRA A/B per module, and/or (adapter-full, decoder+adapter) the adapter's FP32 masters by parameter name."""
    if scope == "adapter-full":
        return {n: p.detach().float().cpu().contiguous().clone() for n, p in trainable_parameters(model, scope).items()}      # the FP32 masters, as a snapshot (never a view of the live tensors)
    if scope in JOINT_SCOPES:
        masters = {n: p.detach().float().cpu().contiguous().clone() for n, p in model.speech_audio_adapter._fp32_master.items()}
        return {**lora_state(model), **masters}
    return lora_state(model)


def load_trainable_state(model, state, scope="decoder"):
    if scope in JOINT_SCOPES:
        load_lora_state(model, {k: v for k, v in state.items() if ".lora_" in k})
        state = {k: v for k, v in state.items() if ".lora_" not in k}
        scope = "adapter-full"                                          # the adapter half is loaded exactly like an adapter-full checkpoint
    if scope != "adapter-full":
        return load_lora_state(model, state)
    params = dict(model.speech_audio_adapter._fp32_master)
    if sorted(params) != sorted(state):
        raise ValueError("adapter checkpoint keys differ from the model's")
    with torch.no_grad():
        for key, p in params.items():
            if tuple(state[key].shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch for {key}")
            p.copy_(state[key].to(p.device, p.dtype))
    sync_adapter_master(model)                                          # the working BF16 weights follow the loaded masters


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ train / eval mode policy
def lora_dropouts(model):
    return {n: m for n, m in model.named_modules() if type(m).__name__ == "Dropout" and "lora_dropout" in n}


def set_train_mode(model, scope="decoder"):
    """Train mode for what the scope trains and eval mode everywhere else (frozen weights do not switch their own dropout off, so it is done here).
    decoder: the decoder trains (LoRA dropout on); adapter-lora / encoder-lora: only that component's LoRA dropout modules; adapter-full: nothing is in train mode, i.e. the adapter's own native
    Dropout(0.01) stays OFF (decision 2026-11-18, for a controlled comparison with adapter-lora). That does not touch gradients: requires_grad is independent of the module mode."""
    if scope == "decoder":
        model.train()
        for name in FROZEN_MODULES:
            getattr(model, name).eval()
        return
    model.eval()
    if scope != "adapter-full":
        for module in lora_dropouts(model).values():
            module.train()


def set_eval_mode(model):
    model.eval()


def check_mode_policy(model, training, scope="decoder"):
    """Raise unless the module modes match the policy (re-checked before every optimiser step and after every recursive mode change)."""
    if scope != "decoder":
        trained = lora_dropouts(model) if scope != "adapter-full" else {}                      # adapter-full: every module, the adapter's native dropout included, stays in eval mode
        for name, module in model.named_modules():
            want = training if name in trained else False
            if module.training != want:
                raise AssertionError(f"{name or 'model'} ({type(module).__name__}) must be in {'train' if want else 'eval'} mode (scope {scope}, training={training})")
        return
    for name in FROZEN_MODULES:
        for module in getattr(model, name).modules():
            if module.training:
                raise AssertionError(f"{name} must stay in eval mode (found a module in train mode: {type(module).__name__})")
    decoder_training = {m.training for n, m in model.text_decoder.named_modules() if type(m).__name__ == "Dropout" and "lora_dropout" in n}
    if decoder_training and decoder_training != {training}:
        raise AssertionError(f"LoRA dropout mode {decoder_training} does not match training={training}")


# ------------------------------------------------------------------ examples and per-token losses
class Example:
    __slots__ = ("clip_id", "input_ids", "attention_mask", "input_features", "feature_attention_mask", "prefix", "entity", "n", "n_entity", "audio_seconds", "language")

    def to(self, device, dtype):
        self.input_ids, self.attention_mask = self.input_ids.to(device), self.attention_mask.to(device)
        self.input_features = self.input_features.to(device=device, dtype=dtype)
        self.feature_attention_mask = self.feature_attention_mask.to(device)
        self.entity = self.entity.to(device)
        return self


def build_example(clip, target_ids, entity_flags, audio, processor, prompt, load_audio, cached=None):
    """Prompt (processor: chat prompt + audio features, <SpeechHere> expanded to the processor's fixed_speech_embeds_length ids: 300 for the pinned 5x model, 500 for the 3x adapter) + supervised target ids. `audio` is the clip's 16 kHz slice.
    `cached` (sft_features.FeatureCache.get) is the identical processor output computed earlier for this exact audio slice; nothing else differs."""
    inputs = cached if cached is not None else processor(text=[prompt], audios=[load_audio(audio)], return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"]
    expected_speech = int(getattr(processor, "fixed_speech_embeds_length", 300))
    if tuple(inputs["input_features"].shape[-2:]) != (128, 3000) or int((prompt_ids == SPEECH_TOKEN_ID).sum()) != expected_speech:
        raise ValueError(f"{clip['clip_id']}: unexpected processor output (features {tuple(inputs['input_features'].shape)}, speech tokens {int((prompt_ids == SPEECH_TOKEN_ID).sum())})")
    target = torch.tensor([target_ids], dtype=prompt_ids.dtype)
    ex = Example()
    ex.clip_id, ex.language = clip["clip_id"], clip["language"]
    ex.input_ids = torch.cat([prompt_ids, target], dim=1)
    ex.attention_mask = torch.ones_like(ex.input_ids)
    ex.input_features, ex.feature_attention_mask = inputs["input_features"], inputs["feature_attention_mask"]
    ex.prefix = int(prompt_ids.shape[1])
    ex.entity = torch.tensor(entity_flags, dtype=torch.bool)
    ex.n, ex.n_entity = len(target_ids), int(sum(entity_flags))
    ex.audio_seconds = clip["duration_seconds"]
    if ex.entity.numel() != ex.n:
        raise ValueError("entity mask length differs from the target length")
    return ex


def token_losses(model, ex):
    """Per supervised-token cross-entropy [T] (FP32, differentiable). Position p>=prefix is scored by logits[p-1]: the shift happens here, once."""
    out = model(input_ids=ex.input_ids, attention_mask=ex.attention_mask, input_features=ex.input_features, feature_attention_mask=ex.feature_attention_mask, use_cache=False)
    logits = out.logits[0, ex.prefix - 1: ex.input_ids.shape[1] - 1]
    return F.cross_entropy(logits.float(), ex.input_ids[0, ex.prefix:], reduction="none")


def weights_for(entity, entity_weight):
    return torch.where(entity, torch.full_like(entity, entity_weight, dtype=torch.float32), torch.ones_like(entity, dtype=torch.float32))


def group_denominator(sizes, objective, entity_weight):
    """Sum of the objective's token weights over the whole accumulation group, from [(supervised tokens, entity tokens)] (known before any forward pass)."""
    w = 1.0 if objective == "ce" else float(entity_weight)
    return float(sum(n - ne + w * ne for n, ne in sizes))


class Sums:
    """Token-pooled running sums for CE, fixed-weight WCE@2, EntCE and NonEntCE (floats; no per-clip averaging)."""
    FIELDS = ("tokens", "ent_tokens", "loss_all", "loss_ent", "loss_wce2", "objective")

    def __init__(self):
        self.v = dict.fromkeys(self.FIELDS, 0.0)

    def add(self, loss, entity, objective_value=0.0):
        v = self.v
        loss = loss.detach().double()
        ent = entity.to(loss.device)
        v["tokens"] += loss.numel()
        v["ent_tokens"] += float(ent.sum())
        v["loss_all"] += float(loss.sum())
        v["loss_ent"] += float(loss[ent].sum())
        v["loss_wce2"] += float((loss * weights_for(ent, 2.0).to(loss.device)).sum())
        v["objective"] += float(objective_value)

    def merge(self, other):
        for k in self.FIELDS:
            self.v[k] += other.v[k]

    def metrics(self):
        v = self.v
        n, e = v["tokens"], v["ent_tokens"]
        return {"tokens": int(n), "entity_tokens": int(e),
                "ce": v["loss_all"] / n if n else NA, "wce2": v["loss_wce2"] / (n + e) if n else NA,
                "ent_ce": v["loss_ent"] / e if e else NA, "non_ent_ce": (v["loss_all"] - v["loss_ent"]) / (n - e) if n - e else NA}


def micro_objective(loss, entity, objective, entity_weight, denominator):
    """The backpropagated quantity of one micro-batch: sum_t w_t * loss_t / (group-wide sum of w): token-pooled across the accumulation group."""
    w = torch.ones_like(loss) if objective == "ce" else weights_for(entity.to(loss.device), entity_weight)
    return (w * loss).sum() / denominator


# ------------------------------------------------------------------ schedule
def lr_factor(update, total, warmup_ratio=0.05):
    """Linear warm-up over ceil(ratio*total) updates then linear decay; `update` is 1-based and advances once per OPTIMISER step."""
    warm = max(1, math.ceil(warmup_ratio * total))
    if update <= warm:
        return update / warm
    return max(0.0, 1.0 - (update - warm) / (total - warm + 1))


# ------------------------------------------------------------------ checkpoint files
def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_lora_safetensors(path, state, meta):
    from safetensors.torch import save_file
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save_file(state, str(tmp), metadata={k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()})
    os.replace(tmp, path)


def load_lora_safetensors(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


def now():
    return time.monotonic()
