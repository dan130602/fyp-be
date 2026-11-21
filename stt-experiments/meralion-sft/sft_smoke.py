"""Bounded smoke / benchmark of the real MERaLiON stack (one process, one GPU, or CPU as a login-node pre-flight with --cpu-preflight).

    python -u sft_smoke.py --data <sft-data-v1> --model <model dir> --batch <audio batch> --out <smoke dir> [--cpu-preflight]

Order (cheapest and most informative first; every check is recorded in <out>/smoke_report.json the moment it finishes, a failing check does not stop later ones):
  A  base-model facts: module paths, parameter counts, encoder determinism in train mode (SpecAugment/dropout), adapter dropout, prompt/target token formats, native loss vs ours
  B  the PRODUCTION trainer (`sft_train.run`) for a few optimiser steps on real clips: data manifest + token fingerprint checks, LoRA injection, logs, epoch checkpoint, validation
  C  gradient flow, frozen isolation, CE==WCE@1, accumulation equivalence on the real model (bf16: cosine/relative error; exactness is proven in the CPU tests)
  D  throughput and peak VRAM at the real settings (accumulation 8), validation speed
  E  repeated-batch learning sanity at a deliberately larger LR (smoke only), checkpoint reload equality, optimiser-state reload
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

import sft_core as C
import sft_data as D
import sft_train as T

REPORT = {"checks": {}, "numbers": {}}


REQUIRED_FULL = ("A0_cluster_tokenizer_reproduces_the_frozen_target_token_ids", "A1_module_paths_and_counts", "A2_dropout_and_spec_augment_facts", "A3_prompt_and_target_format",
                 "A4_base_teacher_forced_ce_and_native_loss_agreement", "B_production_trainer_few_steps", "C1_gradients_isolation_and_updates",
                 "C2_wce1_equals_ce_and_accumulation_equivalence", "D_throughput_vram_and_validation_speed", "E1_repeated_batch_learning_and_reload",
                 "E2_merge_arithmetic_matches_the_lora_forward")
REQUIRED_PREFLIGHT = ("A0_cluster_tokenizer_reproduces_the_frozen_target_token_ids", "A1_module_paths_and_counts", "A3_prompt_and_target_format", "A5_lora_injection_structure_on_the_real_model")


def record(out, name, ok, detail=None, status=None):
    """status: pass | fail | skipped. A skipped check never counts as passed."""
    REPORT["checks"][name] = {"status": status or ("pass" if ok else "fail"), "ok": bool(ok) and status != "skipped", "detail": detail}
    (Path(out) / "smoke_report.json").write_text(json.dumps(REPORT, indent=1, default=str), encoding="utf-8")
    print(f"[smoke] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def numbers(out, **kw):
    REPORT["numbers"].update(kw)
    (Path(out) / "smoke_report.json").write_text(json.dumps(REPORT, indent=1, default=str), encoding="utf-8")


CPU_ONLY = False


def guarded(out, name, heavy=False):
    """heavy checks (a forward/backward pass of the real model) are skipped in the CPU pre-flight, which only checks structure and the token/processor path."""
    def wrap(fn):
        def inner(*a, **k):
            started = time.monotonic()
            if heavy and CPU_ONLY:
                record(out, name, False, "skipped in CPU pre-flight (needs a forward pass)", status="skipped")
                return
            try:
                result = fn(*a, **k)
                record(out, name, True, "ok" if result is None else result)
            except Exception as error:                       # noqa: BLE001 - a failed check is evidence, not a crash
                import traceback
                record(out, name, False, f"{type(error).__name__}: {error}\n{traceback.format_exc()[-1500:]}")
            numbers(out, **{f"seconds[{name}]": round(time.monotonic() - started, 1)})
        inner()                                              # decorated checks run where they are defined
        return inner
    return wrap


def fingerprint(model):
    """{param: (sum, abs-sum)} in fp64 for every parameter that is not LoRA: equal before and after means untouched."""
    out = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_" not in n:
                out[n] = (float(p.double().sum()), float(p.double().abs().sum()))
    return out


def pick_clips(data):
    clips = [json.loads(l) for l in (Path(data) / "clips_train.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(7)
    chosen, seen = [], set()
    for lang in D.LANGUAGES:
        pool = sorted([c for c in clips if c["language"] == lang and c["entity_spans"] and c["duration_seconds"] > 18], key=lambda c: c["clip_id"])
        pick = pool[rng.randrange(len(pool))]
        chosen.append(pick["clip_id"]), seen.add(pick["clip_id"])
    longest = max(clips, key=lambda c: (c["duration_seconds"], len(c["text"])))
    chosen.append(longest["clip_id"])
    pieces = sorted((c for c in clips if "parent" in c and c["entity_spans"]), key=lambda c: c["clip_id"])      # a recovered long-turn segment (v2 packages), when there is one
    if pieces:
        chosen.append(pieces[len(pieces) // 2]["clip_id"])
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu-preflight", action="store_true", help="run on CPU (login node): structure checks only, no forward pass")
    p.add_argument("--stand-in-test", action="store_true", help="tests only: run EVERY check on CPU (the caller patches sft_train.load_stack with a stand-in model)")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    global CPU_ONLY
    CPU_ONLY = args.cpu_preflight
    cpu = args.cpu_preflight or args.stand_in_test
    if args.cpu_preflight:
        torch.set_num_threads(int(os.environ.get("SMOKE_THREADS", "16")))
    device = "cpu" if cpu else "cuda"
    if not cpu and not torch.cuda.is_available():
        sys.exit("no GPU")
    t_start = time.monotonic()
    numbers(out, device=T.gpu_name() if not cpu else "cpu", versions=T.package_versions(), torch_cuda=torch.version.cuda,
            bf16_supported=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()), cpu_preflight=args.cpu_preflight)
    started = time.monotonic()
    model, processor, prompt, load_audio = T.load_stack(args.model, device)
    numbers(out, load_seconds=round(time.monotonic() - started, 1), vram_after_load_gb=T.peak_gb())
    dtype = model.text_decoder.model.embed_tokens.weight.dtype
    n_expected = model.text_decoder.config.num_hidden_layers * 7          # q,k,v,o + gate,up,down per decoder layer: 182 for the real 26-layer model
    tokenizer = processor.tokenizer
    data = args.data
    clips = {c["clip_id"]: c for c in (json.loads(l) for l in (data / "clips_train.jsonl").read_text(encoding="utf-8").splitlines())}
    val_clips = [json.loads(l) for l in (data / "clips_validation.jsonl").read_text(encoding="utf-8").splitlines()]
    store = T.WaveStore(args.batch, {c["call_id"] for c in list(clips.values()) + val_clips}, load_audio)
    probe_ids = pick_clips(data)
    probe_set = T.ClipSet([clips[i] for i in probe_ids], tokenizer, store, processor, prompt, load_audio)
    base_fp = fingerprint(model)

    # ------------------------------------------------------------ A
    @guarded(out, "A0_cluster_tokenizer_reproduces_the_frozen_target_token_ids")
    def _():
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        result = {}
        for name, rows in (("TRAIN", list(clips.values())), ("VALIDATION", val_clips)):
            ids = [D.tokenise_clip(tokenizer, c)[0] for c in rows]
            result[name] = D.tokens_fingerprint(ids) == manifest["token_fingerprints"][name]
        assert all(result.values()), result
        return result

    @guarded(out, "A1_module_paths_and_counts")
    def _():
        tops = sorted({n.split(".")[0] for n, _ in model.named_children()})
        counts = {top: sum(p.numel() for p in getattr(model, top).parameters()) for top in tops}
        targets = C.expected_decoder_targets(model)
        assert len(targets) == n_expected and tops == ["ln_speech", "speech_audio_adapter", "speech_encoder", "text_decoder"], tops
        enc_like = [n for n, m in model.speech_encoder.named_modules() if n.endswith(("q_proj", "v_proj"))]
        assert enc_like and not set(enc_like) & {t.replace("text_decoder.", "") for t in targets}
        return {"top_modules": counts, "decoder_targets": len(targets), "expected": n_expected, "encoder_q_v_projections_not_targeted": len(enc_like), "example": targets[0]}

    @guarded(out, "A2_dropout_and_spec_augment_facts", heavy=True)
    def _():
        adapter_dropout = [(n, m.p) for n, m in model.speech_audio_adapter.named_modules() if type(m).__name__ == "Dropout"]
        ex = probe_set.example(probe_ids[0]).to(device, dtype)
        feats, fam = ex.input_features, ex.feature_attention_mask
        outs = []
        with torch.no_grad():
            model.speech_encoder.train()                      # the worst case: train mode on the encoder
            for _ in range(2):
                outs.append(model.speech_encoder(feats, attention_mask=fam).last_hidden_state.float().cpu())
            model.speech_encoder.eval()
            outs.append(model.speech_encoder(feats, attention_mask=fam).last_hidden_state.float().cpu())
        cfg = model.config.speech_config
        deterministic = bool(torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2]))
        assert deterministic, "encoder output differs between train and eval mode: stochastic masking/dropout is active"
        return {"adapter_dropout_modules": adapter_dropout, "encoder_train_vs_eval_identical": deterministic, "config_apply_spec_augment": cfg.apply_spec_augment,
                "config_dropouts": {"dropout": cfg.dropout, "attention_dropout": cfg.attention_dropout, "activation_dropout": cfg.activation_dropout, "layerdrop": cfg.encoder_layerdrop},
                "conclusion": "SpecAugment is not applied (WhisperEncoder called directly); adapter Dropout stays off only because we keep the module in eval mode"}

    @guarded(out, "A3_prompt_and_target_format")
    def _():
        ex = probe_set.example(probe_ids[0])
        speech = int((ex.input_ids == C.SPEECH_TOKEN_ID).sum())
        tail = ex.input_ids[0, -3:].tolist()
        decoded_prefix = tokenizer.decode(ex.input_ids[0, :ex.prefix][ex.input_ids[0, :ex.prefix] != C.SPEECH_TOKEN_ID])
        assert ex.prefix == 330 and speech == 300 and tail == [107, 108, 1] and ex.input_ids[0, 0] == 2
        return {"prompt_tokens": ex.prefix, "speech_placeholders": speech, "target_tokens": ex.n, "entity_tokens": ex.n_entity, "terminators": tail, "prompt_text": decoded_prefix[-120:]}

    @guarded(out, "A4_base_teacher_forced_ce_and_native_loss_agreement", heavy=True)
    def _():
        rows = []
        model.eval()
        with torch.no_grad():
            for cid in probe_ids:
                ex = probe_set.example(cid).to(device, dtype)
                mine = C.token_losses(model, ex)
                labels = ex.input_ids.clone()
                labels[:, :ex.prefix] = -100
                native = model(input_ids=ex.input_ids, attention_mask=ex.attention_mask, input_features=ex.input_features, feature_attention_mask=ex.feature_attention_mask, use_cache=False, labels=labels).loss
                rows.append({"clip": cid, "mine": float(mine.mean()), "native": float(native), "ent_ce": float(mine[ex.entity.to(mine.device)].mean()) if ex.n_entity else None})
                assert math.isfinite(rows[-1]["mine"]) and abs(rows[-1]["mine"] - rows[-1]["native"]) < 5e-3 * max(1.0, rows[-1]["native"]), rows[-1]
        return rows

    # ------------------------------------------------------------ B
    run_args = T.parse(["--run-id", "smoke-run", "--out-root", str(out), "--data", str(data), "--model", str(args.model), "--batch", str(args.batch), "--objective", "wce",
                        "--sampling", "targeted", "--k", "8", "--epochs", "1", "--accumulation", "2" if cpu else "4", "--max-updates", "2" if cpu else "4",
                        "--validate-clips", "4" if cpu else "12", "--log-every", "1", "--save-every", "2"])

    @guarded(out, "B_production_trainer_few_steps", heavy=True)
    def _():
        started = time.monotonic()
        # the production path loads the manifest, verifies hashes + token fingerprints, injects LoRA, trains, validates, checkpoints
        T.run(run_args, (model, processor, prompt, load_audio))
        run_dir = out / "smoke-run"
        rows = [json.loads(l) for l in (run_dir / "train_metrics.jsonl").read_text().splitlines()]
        assert rows and all(math.isfinite(r["objective_loss"]) for r in rows) and (run_dir / "checkpoints" / "epoch-1.safetensors").exists()
        return {"seconds": round(time.monotonic() - started, 1), "first_row": rows[0], "files": sorted(p.name for p in run_dir.iterdir()),
                "lora_trainable": json.loads((run_dir / "config.json").read_text())["parameters"]}

    if args.cpu_preflight:
        @guarded(out, "A5_lora_injection_structure_on_the_real_model")
        def _():
            targets = C.inject_decoder_lora(model, 8, 16, 0.05, 42)
            counts = C.check_trainable(model, targets)
            C.set_train_mode(model)
            C.check_mode_policy(model, True)
            C.set_eval_mode(model)
            C.check_mode_policy(model, False)
            assert len(targets) == n_expected
            return {"targets": len(targets), "parameters": counts, "lora_dtype": "float32", "mode_policy": "frozen encoder/ln_speech/adapter stay eval after set_train_mode"}
    injected = any(".lora_A." in n for n, _ in model.named_parameters())

    # ------------------------------------------------------------ C
    @guarded(out, "C1_gradients_isolation_and_updates", heavy=True)
    def _():
        assert injected, "LoRA was not injected (step B failed)"
        targets = sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters() if ".lora_A." in n})
        counts = C.check_trainable(model, targets)
        T_lora = C.lora_parameters(model)
        assert all(n.startswith("text_decoder.model.layers.") for n in T_lora)        # nothing in the encoder, ln_speech, adapter, embeddings or head
        assert sum(1 for n in T_lora if ".lora_A." in n) == n_expected == sum(1 for n in T_lora if ".lora_B." in n)
        after_fp = {n.replace(".base_layer", ""): v for n, v in fingerprint(model).items()}
        assert after_fp.keys() == base_fp.keys(), sorted(set(after_fp) ^ set(base_fp))[:3]
        moved = [n for n in base_fp if base_fp[n] != after_fp[n]]
        assert not moved, f"frozen parameters changed: {moved[:3]}"
        b_norm = sum(float(p.detach().abs().sum()) for n, p in T_lora.items() if ".lora_B." in n)
        assert b_norm > 0, "LoRA B never moved: no update happened"
        return {"parameters": counts, "frozen_params_compared": len(base_fp), "lora_B_abs_sum_after_training": b_norm}

    @guarded(out, "C2_wce1_equals_ce_and_accumulation_equivalence", heavy=True)
    def _():
        assert injected
        ids = probe_ids[:2]
        exs = [probe_set.example(i).to(device, dtype) for i in ids]
        if exs[0].n == exs[1].n:
            exs = [probe_set.example(probe_ids[0]).to(device, dtype), probe_set.example(probe_ids[-1]).to(device, dtype)]
        model.eval()
        params = list(C.lora_parameters(model).values())
        # WCE@1 == CE on the real model
        with torch.no_grad():
            l = C.token_losses(model, exs[0])
            d = float(l.numel())
            a = float(C.micro_objective(l, exs[0].entity, "ce", 2.0, d))
            b = float(C.micro_objective(l, exs[0].entity, "wce", 1.0, C.group_denominator([(exs[0].n, exs[0].n_entity)], "wce", 1.0)))
        assert abs(a - b) < 1e-6
        den = C.group_denominator([(e.n, e.n_entity) for e in exs], "wce", 2.0)

        def flat():
            return torch.cat([p_.grad.detach().float().flatten() for p_ in params])
        for p_ in params:
            p_.grad = None
        for e in exs:                                          # (a) what train_group does: one backward per micro-batch
            C.micro_objective(C.token_losses(model, e), e.entity, "wce", 2.0, den).backward()
        accumulated = flat()
        for p_ in params:
            p_.grad = None
        sum(C.micro_objective(C.token_losses(model, e), e.entity, "wce", 2.0, den) for e in exs).backward()      # (b) ONE backward on the summed group objective, identical forward numerics
        summed = flat()
        cos = float(torch.nn.functional.cosine_similarity(accumulated, summed, dim=0))
        rel = float((accumulated - summed).norm() / summed.norm())
        assert cos > 0.99999 and rel < 1e-3, (cos, rel)          # accumulation logic is exact up to float summation order
        # (c) informational: a padded batch of the same clips differs only by bf16 batching numerics (not by the loss/accumulation logic, which fp32 CPU tests prove exact)
        for p_ in params:
            p_.grad = None
        m = max(e.input_ids.shape[1] for e in exs)
        pad = lambda t: torch.nn.functional.pad(t[0], (0, m - t.shape[1]))      # noqa: E731
        logits = model(input_ids=torch.stack([pad(e.input_ids) for e in exs]), attention_mask=torch.stack([pad(e.attention_mask) for e in exs]),
                       input_features=torch.cat([e.input_features for e in exs]), feature_attention_mask=torch.cat([e.feature_attention_mask for e in exs]), use_cache=False).logits
        tot, wt = 0.0, 0.0
        for i, e in enumerate(exs):
            tok = torch.nn.functional.cross_entropy(logits[i, e.prefix - 1:e.input_ids.shape[1] - 1].float(), e.input_ids[0, e.prefix:], reduction="none")
            w = C.weights_for(e.entity.to(tok.device), 2.0)
            tot, wt = tot + (w * tok).sum(), wt + float(w.sum())
        (tot / wt).backward()
        batched = flat()
        cos_b = float(torch.nn.functional.cosine_similarity(accumulated, batched, dim=0))
        rel_b = float((accumulated - batched).norm() / batched.norm())
        for p_ in params:
            p_.grad = None
        assert cos_b > 0.9, cos_b
        return {"accumulated_vs_summed_graph": {"cosine": cos, "relative_difference": rel}, "wce1_equals_ce": True, "lengths": [e.n for e in exs],
                "informational_padded_batch_bf16": {"cosine": cos_b, "relative_difference": rel_b, "note": "bf16 batching/padding numerics, not a logic difference"}}

    # ------------------------------------------------------------ D (GPU only)
    if not args.cpu_preflight:
        @guarded(out, "D_throughput_vram_and_validation_speed")
        def _():
            assert injected
            all_ids = sorted(clips)
            rng = random.Random(11)
            bench = rng.sample(all_ids, 24)
            bench_set = T.ClipSet([clips[i] for i in bench], tokenizer, store, processor, prompt, load_audio)
            params = list(C.lora_parameters(model).values())
            opt = torch.optim.AdamW(params, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, fused=torch.cuda.is_available())
            C.set_train_mode(model)
            T.cuda_sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            steps, seconds, micro_tokens = [], [], 0
            for s in range(3):                                   # real settings: accumulation 8; the first step includes warm-up
                group = bench[s * 8:(s + 1) * 8]
                sums = C.Sums()
                a = argparse.Namespace(objective="wce", entity_weight=2.0, lr=5e-5, clip=1.0, warmup=0.05, scope="decoder")
                _gn, _lr, sec = T.train_group(model, bench_set.example, group, bench_set, a, device, dtype, params, s + 1, 1000, opt, sums)
                T.cuda_sync()
                steps.append(round(sec, 2))
                micro_tokens += int(sums.v["tokens"])
            peak_train = T.peak_gb()
            C.set_eval_mode(model)
            val_set = T.ClipSet(val_clips[:20], tokenizer, store, processor, prompt, load_audio)
            t0 = time.monotonic()
            T.evaluate_ce(model, val_set, device, dtype)
            T.cuda_sync()
            val_seconds = time.monotonic() - t0
            C.set_train_mode(model)
            steady = steps[1:]
            numbers(out, seconds_per_optimiser_step_accum8=steps, steady_seconds_per_microstep=round(sum(steady) / len(steady) / 8, 3), peak_vram_training_gb=peak_train,
                    validation_seconds_per_clip=round(val_seconds / 20, 3), est_validation_epoch_seconds_575_clips=round(val_seconds / 20 * 575, 0))
            return {"optimiser_step_seconds_accum8": steps, "peak_vram_gb": peak_train, "validation_20_clips_seconds": round(val_seconds, 1)}

    # ------------------------------------------------------------ E
    @guarded(out, "E1_repeated_batch_learning_and_reload", heavy=True)
    def _():
        assert injected
        ids = probe_ids[:2]
        ds = T.ClipSet([clips[i] for i in ids], tokenizer, store, processor, prompt, load_audio)
        params = list(C.lora_parameters(model).values())
        opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=0.0, fused=torch.cuda.is_available())
        losses = []
        C.set_train_mode(model)
        for step in range(8 if cpu else 16):
            s = C.Sums()
            a = argparse.Namespace(objective="ce", entity_weight=2.0, lr=6e-4, clip=1.0, warmup=0.0, scope="decoder")
            T.train_group(model, ds.example, ids, ds, a, device, dtype, params, 50, 1000, opt, s)    # lr factor ~0.95; smoke-only LR
            losses.append(s.metrics()["ce"])
        assert losses[-1] < 0.9 * losses[0], losses
        # checkpoint reload equality
        state = C.lora_state(model)
        path = out / "reload.safetensors"
        C.save_lora_safetensors(path, state, {"rank": 8, "alpha": 16, "digest": C.state_digest(state)})
        model.eval()
        ex = ds.example(ids[0]).to(device, dtype)
        with torch.no_grad():
            before = C.token_losses(model, ex).detach().cpu()
        zero = {k: torch.zeros_like(v) for k, v in state.items()}
        C.load_lora_state(model, zero)
        with torch.no_grad():
            zeroed = C.token_losses(model, ex).detach().cpu()
        loaded, _ = C.load_lora_safetensors(path)
        C.load_lora_state(model, loaded)
        with torch.no_grad():
            after = C.token_losses(model, ex).detach().cpu()
        assert torch.equal(before, after) and not torch.equal(before, zeroed)
        # optimiser state survives torch.save / load (fused AdamW)
        torch.save(opt.state_dict(), out / "opt.pt")
        opt2 = torch.optim.AdamW(params, lr=2e-3, fused=torch.cuda.is_available())
        opt2.load_state_dict(torch.load(out / "opt.pt", map_location="cpu", weights_only=False))
        assert len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"])
        return {"ce_by_step": [round(x, 4) for x in losses], "reload_identical": True, "optimiser_state_reload": True}

    @guarded(out, "E2_merge_arithmetic_matches_the_lora_forward", heavy=True)
    def _():
        import sft_export as X
        assert injected
        ex = probe_set.example(probe_ids[2]).to(device, dtype)          # a clip the repeated-batch check did NOT memorise (its CE is not ~0)
        model.eval()
        with torch.no_grad():
            before = C.token_losses(model, ex).detach().float().cpu()
            logits_before = model(input_ids=ex.input_ids, attention_mask=ex.attention_mask, input_features=ex.input_features, feature_attention_mask=ex.feature_attention_mask, use_cache=False).logits[0, ex.prefix - 1:-1].argmax(-1)
            scale, merged = 16 / 8, 0
            for name in sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters() if ".lora_A." in n}):
                layer = model.get_submodule(name)
                w = layer.base_layer.weight
                w.copy_(X.merge_tensor(w, layer.lora_A["default"].weight, layer.lora_B["default"].weight, scale))      # the exporter's own merge function
                layer.lora_B["default"].weight.zero_()
                merged += 1
            after = C.token_losses(model, ex).detach().float().cpu()
            logits_after = model(input_ids=ex.input_ids, attention_mask=ex.attention_mask, input_features=ex.input_features, feature_attention_mask=ex.feature_attention_mask, use_cache=False).logits[0, ex.prefix - 1:-1].argmax(-1)
        delta = (before - after).abs()
        agree = float((logits_before == logits_after).float().mean())
        assert merged == n_expected and float(delta.max()) < 0.25 and float(delta.mean()) < 0.02 and agree >= 0.97, (float(delta.max()), float(delta.mean()), agree)
        return {"merged_modules": merged, "max_abs_token_ce_difference": float(delta.max()), "mean_abs_token_ce_difference": float(delta.mean()), "top1_agreement": agree,
                "ce_before": float(before.mean()), "ce_after_merge": float(after.mean())}

    numbers(out, total_seconds=round(time.monotonic() - t_start, 1), peak_vram_overall_gb=T.peak_gb())
    return finish(out, args.cpu_preflight)


def finish(out, preflight, crash=None):
    """Overall verdict: PASS only when EVERY required check ran and passed (a failed, skipped or missing required check, or a crash, is FAIL). The pre-flight can only say PREFLIGHT_OK."""
    required = REQUIRED_PREFLIGHT if preflight else REQUIRED_FULL
    checks = REPORT["checks"]
    missing = [n for n in required if n not in checks]
    failed = [n for n in required if n in checks and checks[n]["status"] == "fail"]
    skipped = [n for n in required if n in checks and checks[n]["status"] == "skipped"]
    extra_failed = [n for n, v in checks.items() if n not in required and v["status"] == "fail"]
    ok = not (missing or failed or skipped or extra_failed or crash)
    overall = ("PREFLIGHT_OK (structure only: NOT a smoke pass)" if preflight else "PASS") if ok else "FAIL"
    summary = {"overall": overall, "required": list(required), "missing": missing, "failed": failed, "skipped": skipped, "other_failed": extra_failed, "crash": crash}
    REPORT["summary"] = summary
    (Path(out) / "smoke_report.json").write_text(json.dumps(REPORT, indent=1, default=str), encoding="utf-8")
    (Path(out) / "SMOKE_RESULT.txt").write_text(overall + "\n" + json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[smoke] OVERALL {overall}; missing {missing}; failed {failed + extra_failed}; skipped {skipped}", flush=True)
    return 0 if ok else 1


def run_main():
    out = None
    try:
        argv = sys.argv
        out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
        return main()
    except SystemExit as stop:
        if stop.code not in (0, None) and out is not None:
            out.mkdir(parents=True, exist_ok=True)
            finish(out, "--cpu-preflight" in sys.argv, crash=f"SystemExit: {stop.code}")
        raise
    except BaseException as error:                               # noqa: BLE001 - any crash must still produce a FAIL verdict
        import traceback
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            finish(out, "--cpu-preflight" in sys.argv, crash=f"{type(error).__name__}: {error}\n{traceback.format_exc()[-2000:]}")
        raise


if __name__ == "__main__":
    sys.exit(run_main())
