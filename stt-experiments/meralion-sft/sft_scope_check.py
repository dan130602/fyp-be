"""Real-stack pre-flight of ONE training scope (adapter-lora | adapter-full | encoder-lora; decoder works too) before any substantive training. One process, one GPU, a few minutes.

    python -u sft_scope_check.py --scope adapter-full --data <sft-data> --model <pinned model dir> --batch <audio batch> --out <dir> --export-dir <empty dir ending in -tmp> [--keep-export]

Checks (each recorded in <out>/scope_report.json the moment it finishes). A check is `pass`, `fail` or `skipped`; the overall result is PASS only when every check passed, FAIL when any failed and
INCOMPLETE (exit 2) when none failed but something was skipped: a skipped check never counts as passed.
  counts      target modules and trainable parameters equal the numbers expected for the pinned model (revision 9f226d7, rank 8)
  starts      the adapted model equals the base model before any update (LoRA: B = 0; adapter-full: the BF16 working weights are the pretrained ones)
  modes       module modes follow the scope's policy (everything outside the scope in eval mode)
  gradients   one real forward/backward: gradients land on the scope's parameters and nowhere else, and every LoRA B (or every adapter tensor) receives a non-zero gradient,
              i.e. the gradient really crossed the frozen downstream components
  updates     two real optimiser steps (accumulation 8, deliberately larger smoke LR): seconds per update, peak VRAM, optimiser-state bytes; every parameter outside the scope is unchanged
  reload      the checkpoint written for the scope reloads into the model and reproduces its token losses exactly
  export      sft_export (which refuses a payload that is not exactly the declared scope) writes the standalone BF16 model; it loads like any checkpoint (Transformers) and reproduces the trained
              model's token losses within BF16 rounding. SKIPPED without --export-dir.
This script does NOT start vLLM and does not score: scope_check_job.sbatch runs the unchanged worker + patched vLLM on two validation calls with the kept export as its next stage, and the transcripts
are scored on the laptop (the cluster has no reference text).
"""
import argparse
import gc
import json
import random
import shutil
import sys
from pathlib import Path

import torch

import sft_core as C
import sft_data as D
import sft_export as X
import sft_smoke as S
import sft_train as T

ADAPTER_LINEARS = ["speech_audio_adapter.mlp_adapter.0", "speech_audio_adapter.out_proj"]
EXPECTED = {"decoder": (182, 10_383_360), "adapter-lora": (2, 172_032), "adapter-full": (4, 55_714_304), "encoder-lora": (192, 5_898_240), "decoder+adapter": (186, 66_097_664), "decoder+adapter+encoder": (378, 71_995_904)}      # (targets, trainable parameters): pinned model, rank 8


def token_ce(model, ds, ids, device, dtype):
    """Per clip (per-token CE, arg-max tokens), eval mode, no gradient."""
    model.eval()
    out = []
    with torch.no_grad():
        for clip_id in ids:
            ex = ds.example(clip_id).to(device, dtype)
            ce = C.token_losses(model, ex).detach().float().cpu()
            logits = model(input_ids=ex.input_ids, attention_mask=ex.attention_mask, input_features=ex.input_features, feature_attention_mask=ex.feature_attention_mask, use_cache=False).logits
            out.append((ce, logits[0, ex.prefix - 1:-1].argmax(-1).cpu()))
    return out


def compare(a, b):
    diffs = [(x[0] - y[0]).abs() for x, y in zip(a, b)]
    return {"max_abs_token_ce_diff": max(float(d.max()) for d in diffs), "mean_abs_token_ce_diff": sum(float(d.mean()) for d in diffs) / len(diffs),
            "top1_agreement": min(float((x[1] == y[1]).float().mean()) for x, y in zip(a, b))}


EXPECTED_3X = {"adapter-full": (4, 39_330_304), "decoder+adapter": (186, 49_713_664)}                                 # the 3x adapter (first projection 3840 -> 6400): 24,582,400 + 14,747,904 parameters


def trace_dtypes(model, names):
    """Register forward hooks on the named Linear (or LoRA-wrapped Linear) modules; returns (records, handles). Each record = dtypes of the module's input, its base weight and its output."""
    records, handles = {}, []
    for name in names:
        module = model.get_submodule(name)
        weight = (module.base_layer if hasattr(module, "base_layer") else module).weight

        def hook(mod, args, out, name=name, weight=weight):
            records[name] = {"input": str(args[0].dtype), "weight": str(weight.dtype), "output": str(out.dtype)}
        handles.append(module.register_forward_hook(hook))
    return records, handles


def verdict(checks):
    statuses = [c["status"] for c in checks.values()]
    return "FAIL" if "fail" in statuses else ("INCOMPLETE" if "skipped" in statuses else "PASS")


def check(args, stack=None, load_stack=None):
    scope, out = args.scope, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": scope, "checks": {}, "numbers": {}}

    def rec(name, ok, detail=None, skipped=False):
        status = "skipped" if skipped else ("pass" if ok else "fail")
        report["checks"][name] = {"status": status, "ok": status == "pass", "detail": detail}
        (out / "scope_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"[scope-check {scope}] {status.upper()} {name}: {json.dumps(detail, default=str)[:600]}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, prompt, load_audio = stack or T.load_stack(args.model, device)
    dtype = model.text_decoder.model.embed_tokens.weight.dtype
    clips = {c["clip_id"]: c for c in (json.loads(l) for l in (Path(args.data) / "clips_train.jsonl").read_text(encoding="utf-8").splitlines())}
    store = T.WaveStore(args.batch, {c["call_id"] for c in clips.values()}, load_audio)
    ids = S.pick_clips(args.data)[:4]
    probe = T.ClipSet([clips[i] for i in ids], processor.tokenizer, store, processor, prompt, load_audio)
    fp0 = S.fingerprint(model)
    base = token_ce(model, probe, ids, device, dtype)

    if scope == "adapter-full":
        targets = C.inject_adapter_full(model)
    elif scope in C.JOINT_SCOPES:
        targets = C.inject_joint(model, 8, 16, 0.0, 42, encoder="encoder" in scope)
    else:
        targets = C.inject_lora(model, scope, 8, 16, 0.0, 42)
    counts = C.check_trainable(model, targets, scope)
    if args.encoder_checkpointing:
        counts["encoder_checkpointed_layers"] = C.enable_encoder_checkpointing(model)
    architecture = getattr(args, "architecture", "5x")
    if architecture == "3x" and scope not in EXPECTED_3X and not args.stand_in:
        raise SystemExit(f"no 3x expectations for scope {scope}")
    expected = None if args.stand_in else (EXPECTED_3X if architecture == "3x" else EXPECTED)[scope]
    rec("counts", expected is None or (len(targets), counts["trainable"]) == expected, {"targets": len(targets), **counts, "expected_targets_and_parameters": expected, "target_names": targets[:8]})
    init = token_ce(model, probe, ids, device, dtype)
    d0 = compare(base, init)
    rec("starts", d0["mean_abs_token_ce_diff"] < 1e-3, d0)
    C.set_train_mode(model, scope)
    C.check_mode_policy(model, True, scope)
    rec("modes", True, f"train mode: only the {scope} scope trains; everything else eval")

    mine = C.trainable_parameters(model, scope)
    trace_names = ADAPTER_LINEARS if C.has_adapter_full(scope) else targets[:2]
    traced, handles = trace_dtypes(model, trace_names)
    C.token_losses(model, probe.example(ids[0]).to(device, dtype)).mean().backward()
    for h in handles:
        h.remove()
    report["numbers"]["computation_dtypes"] = {"traced_linears": traced, "trainable_parameters_and_gradients": sorted({str(p.dtype) for p in mine.values()}), "model_working_dtype": str(dtype)}
    master_ok = not C.has_adapter_full(scope) or all(v == {"input": str(dtype), "weight": str(dtype), "output": str(dtype)} for v in traced.values())
    grads = {n: p.grad for n, p in list(model.named_parameters()) + list(mine.items()) if p.grad is not None}
    stray = sorted(set(grads) - set(mine))
    need = sorted(n for n in mine if ".lora_B." in n or (C.has_adapter_full(scope) and ".lora_" not in n))
    dead = sorted(n for n in need if n not in grads or not float(grads[n].abs().sum()) > 0)
    rec("gradients", not stray and not dead and master_ok, {"computation_dtypes": report["numbers"]["computation_dtypes"], "gradient_tensors": len(grads), "stray_outside_scope": stray[:4], "scope_tensors_without_gradient": dead[:4], "required_tensors": len(need),
                                              "grad_l2_norm": float(sum(float(g.float().pow(2).sum()) for g in grads.values()) ** 0.5)})
    for p in mine.values():
        p.grad = None

    params = list(mine.values())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=(device == "cuda"))
    pool = sorted(clips)
    bench_ids = random.Random(11).sample(pool, min(len(pool), 2 * args.accumulation))
    bench = T.ClipSet([clips[i] for i in bench_ids], processor.tokenizer, store, processor, prompt, load_audio)
    ns = argparse.Namespace(objective="ce", entity_weight=2.0, lr=args.lr, clip=1.0, warmup=0.0, scope=scope)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seconds = []
    for step in range(2):
        group = bench_ids[step * args.accumulation:(step + 1) * args.accumulation]
        _gn, _lr, sec = T.train_group(model, bench.example, group, bench, ns, device, dtype, params, step + 1, 1000, opt, C.Sums())
        T.cuda_sync()
        seconds.append(round(sec, 2))
    opt_bytes = sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
    after = {n.replace(".base_layer", ""): v for n, v in S.fingerprint(model).items()}
    inside = (lambda n: n.startswith("speech_audio_adapter.")) if C.has_adapter_full(scope) else (lambda n: False)
    moved = [n for n in fp0 if not inside(n) and fp0[n] != after.get(n)]
    changed_inside = [n for n in fp0 if inside(n) and fp0[n] != after.get(n)]
    ok = after.keys() == fp0.keys() and not moved and (not C.has_adapter_full(scope) or bool(changed_inside)) and (scope == "adapter-full" or any(float(p.detach().abs().sum()) > 0 for n, p in mine.items() if ".lora_B." in n))
    report["numbers"].update(seconds_per_update=seconds, peak_vram_gb=T.peak_gb(), optimiser_state_bytes=opt_bytes, parameters=counts)
    rec("updates", ok, {"seconds_per_update": seconds, "peak_vram_gb": T.peak_gb(), "optimiser_state_mb": round(opt_bytes / 2**20, 2), "parameters_compared_outside_the_scope": len(fp0) - len(changed_inside),
                        "moved_outside_the_scope": moved[:4]})

    state = C.trainable_state(model, scope)
    ckpt = out / f"scope-check-{scope}.safetensors"
    C.save_lora_safetensors(ckpt, state, {"run_id": f"scope-check-{scope}", "epoch": 1, "update": 2, "rank": 8, "alpha": 16, "digest": C.state_digest(state), "targets": len(targets), "scope": scope,
                                          "model_revision": D.MODEL_REVISION})
    trained = token_ce(model, probe, ids, device, dtype)
    with torch.no_grad():
        for p in mine.values():
            p.add_(torch.randn_like(p) * 1e-2)                                                  # scramble, then reload the checkpoint
    loaded, _meta = C.load_lora_safetensors(ckpt)
    C.load_trainable_state(model, loaded, scope)
    again = compare(trained, token_ce(model, probe, ids, device, dtype))
    rec("reload", again["max_abs_token_ce_diff"] == 0.0 and C.state_digest(C.trainable_state(model, scope)) == C.state_digest(state), {**again, "checkpoint_bytes": ckpt.stat().st_size, "effect_vs_base": compare(base, trained)})

    if args.export_dir:
        exp = Path(args.export_dir)
        export_id = args.export_id or f"scope-check-{scope}"
        if args.runtime_config_out:                                                              # the full CLI: manifest + runtime config the worker needs for the vLLM stage
            X.main(["--base", str(args.model), "--lora", str(ckpt), "--out", str(exp), "--id", export_id, "--base-manifest", str(args.base_manifest), "--manifest-out", str(args.manifest_out),
                    "--runtime-config-in", str(args.runtime_config_in), "--runtime-config-out", str(args.runtime_config_out)])
            manifest = json.loads(Path(args.manifest_out).read_text(encoding="utf-8"))
        else:
            manifest = X.export(args.model, ckpt, exp, export_id)
        del model, opt, params, mine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m2, p2, pr2, la2 = (load_stack or T.load_stack)(exp, device)
        ds2 = T.ClipSet([clips[i] for i in ids], p2.tokenizer, store, p2, pr2, la2)
        merged = token_ce(m2, ds2, ids, device, m2.text_decoder.model.embed_tokens.weight.dtype)
        cm = compare(trained, merged)
        rec("export", cm["mean_abs_token_ce_diff"] < 0.02 and cm["top1_agreement"] >= 0.97, {**cm, "base_vs_merged": compare(base, merged), "export_delta": manifest["derivation"]["delta"],
                                                                                              "export_files": len(manifest["files"]), "kept_for_vllm_stage": bool(args.keep_export)})
        if exp.name.endswith("-tmp") and not args.keep_export:
            shutil.rmtree(exp)
    else:
        rec("export", False, "not requested (--export-dir): the export is NOT verified", skipped=True)
    report["overall"] = verdict(report["checks"])
    (out / "scope_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"[scope-check {scope}] OVERALL {report['overall']}", flush=True)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["overall"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", choices=C.SCOPES, required=True)
    for name in ("data", "model", "batch", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--export-dir", type=Path, default=None, help="an EMPTY folder whose name ends in -tmp: the merged export is written here, loaded and compared (deleted afterwards unless --keep-export)")
    p.add_argument("--keep-export", action="store_true", help="keep the export for the vLLM stage of scope_check_job.sbatch (the job deletes it)")
    for name in ("base-manifest", "manifest-out", "runtime-config-in", "runtime-config-out"):
        p.add_argument(f"--{name}", type=Path, default=None, help="with --runtime-config-out: also write the model manifest and the worker runtime config (needed by the vLLM stage)")
    p.add_argument("--export-id", default=None, help="model manifest id of the export (must equal the manifest file's basename the worker looks up)")
    p.add_argument("--encoder-checkpointing", action="store_true", help="enable encoder layer checkpointing for the run (encoder-lora): measures its VRAM and speed")
    p.add_argument("--lr", type=float, default=6e-4, help="smoke-only learning rate (large enough that the export check has a real adapter effect to reproduce)")
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--architecture", choices=("5x", "3x"), default="5x", help="which model directory --model is: the pinned 5x model or the A02 3x adapter model (selects the expected parameter counts)")
    p.add_argument("--stand-in", action="store_true", help="tests only: skip the pinned-model parameter counts")
    return check(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
