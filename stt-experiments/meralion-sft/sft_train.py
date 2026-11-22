"""One shared, configurable single-GPU trainer for E01-E06 (CE or entity-weighted CE; uniform or targeted sampling). `--scope` picks what is trained, everything else stays frozen:
decoder (default: attention + MLP LoRA on the 182 decoder projections, E01-E04), adapter-lora (LoRA on every Linear of the speech adapter), adapter-full (all speech-adapter
parameters, FP32 master weights), encoder-lora (LoRA on the speech encoder's attention + MLP projections) and decoder+adapter (decoder LoRA and the full adapter together).

    python -u sft_train.py --run-id E01 --objective ce --sampling uniform --k 978 --data <sft-data-v1> --model <model dir> --batch <audio batch dir> --out-root <runs>

Matched design (spec section 8): every unique training clip once per epoch + K extra draws (uniform = from all clips, targeted = from the target pool: the clips that contain entity spans),
shuffled per epoch with seeds that depend only on (seed, sampling mode, epoch), so E01/E02 share one draw list and E03/E04 another.
One optimiser step = `accumulation` micro-batches of ONE clip; the per-epoch final partial group is flushed. The loss is normalised by the token weights of the
WHOLE accumulation group, not per clip. Resumable: state is written every --save-every optimiser steps and at every epoch end (`--stop-after-epoch` ends the process
after an epoch so a validation process can use the whole GPU, then `--resume` continues).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch

import sft_augment as A
import sft_core as C
import sft_data as D
import sft_features as F


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ data access
class WaveStore:
    """Whole-call 16 kHz mono waveforms loaded exactly as the inference pipeline loads them (soundfile -> meralion_3_asr load_audio), kept in memory."""

    def __init__(self, batch_dir, call_ids, load_audio):
        self.dir = Path(batch_dir)
        calls = json.loads((self.dir / "calls.json").read_text(encoding="utf-8"))
        if calls["revision"] != D.MODEL_REVISION:
            raise ValueError("the audio batch was prepared for a different model revision")
        self.records = {c["call_id"]: c for c in calls["calls"]}
        missing = sorted(set(call_ids) - set(self.records))
        if missing:
            raise ValueError(f"calls missing from the audio batch: {missing[:5]}")
        self.load_audio, self.cache = load_audio, {}

    def wave(self, call_id):
        if call_id not in self.cache:
            import soundfile
            rec = self.records[call_id]
            path = self.dir / rec["audio"]
            with open(path, "rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != rec["audio_sha256"]:
                    raise ValueError(f"audio hash differs for {call_id}")
            samples, rate = soundfile.read(path, dtype="float32")
            self.cache[call_id] = self.load_audio((samples, rate))
        return self.cache[call_id]

    def clip(self, clip):
        wave = self.wave(clip["call_id"])
        a, b = clip["start_sample"], min(clip["end_sample"], len(wave))
        if not 0 <= a < b or b - a > D.MAX_CLIP_SAMPLES or clip["end_sample"] - len(wave) > 1600:
            raise ValueError(f"{clip['clip_id']}: clip [{a}, {b}) does not fit the recording ({len(wave)} samples)")
        import numpy as np
        return np.ascontiguousarray(wave[a:b])


class ClipSet:
    """Clips of one split with their pre-tokenised targets; builds Examples on demand."""

    def __init__(self, clips, tokenizer, store, processor, prompt, load_audio, features=None, augmenter=None, augment_all=False):
        self.clips = {c["clip_id"]: c for c in clips}
        self.tokens = {}
        for c in clips:
            ids, flags, _ = D.tokenise_clip(tokenizer, c)
            self.tokens[c["clip_id"]] = (ids, flags)
        self.store, self.processor, self.prompt, self.load_audio, self.features = store, processor, prompt, load_audio, features
        self.augmenter, self.augment_all = augmenter, augment_all                     # a draw id `<clip>|aug<epoch>` is an augmented copy of the clip (sft_augment); everything else is clean

    def sizes(self, draw):
        ids, flags = self.tokens[A.base_id(draw)]
        return len(ids), sum(flags)

    def example(self, draw):
        clip_id, epoch = A.parse_draw(draw)
        clip = self.clips[clip_id]
        ids, flags = self.tokens[clip_id]
        audio = self.store.clip(clip)
        if self.augmenter is not None and (epoch is not None or self.augment_all):
            audio, cached = self.augmenter.apply(audio, clip_id, epoch or 0), None        # augmented audio is never in the (clean-audio) feature cache
        else:
            cached = self.features.get(clip, audio) if self.features is not None else None
        return C.build_example(clip, ids, flags, audio, self.processor, self.prompt, self.load_audio, cached=cached)


# ------------------------------------------------------------------ model stack
def load_processor(model_dir):
    """Processor, chat prompt and audio loader only (no model, no GPU): what the feature cache is built with."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    from transformers import AutoProcessor
    from meralion_3_asr._audio_io import load_audio
    from meralion_3_asr.prompts import build_messages
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = processor.tokenizer.apply_chat_template(build_messages(), tokenize=False, add_generation_prompt=True)
    return processor, prompt, load_audio


def load_stack(model_dir, device="cuda", attn="eager", gradient_checkpointing=False):
    processor, prompt, load_audio = load_processor(model_dir)
    from transformers import AutoModelForSpeechSeq2Seq
    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_dir), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation=attn).to(device)
    model.config.use_cache = False
    model.text_decoder.config.use_cache = False
    if gradient_checkpointing:
        model.text_decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, processor, prompt, load_audio


def package_versions():
    from importlib import metadata
    out = {}
    for name in ("torch", "transformers", "peft", "safetensors", "tokenizers", "numpy", "soundfile", "librosa", "meralion-3-asr", "accelerate"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# ------------------------------------------------------------------ training
def digest_of_config(config):
    """Identity of a run's resolved configuration (a resumed run must match it). Options that are off do not enter it, so a run started before the option existed still resumes with newer code."""
    ignored = ("resume", "stop_after_epoch", "max_updates", "smoke", "skip_baseline_validation")
    return sha256_bytes(json.dumps({k: v for k, v in config.items() if k not in ignored and not (k == "augment" and v == "none")}, sort_keys=True).encode())


def plan_for(args, train_ids, important_ids):
    return D.plan_epochs(train_ids, important_ids, args.k, args.epochs, args.seed, args.sampling)


def total_updates(plans, accumulation):
    return sum(len(D.group_boundaries(len(p), accumulation)) for p in plans)


def sampling_summary(draws, trainset, important):
    from collections import Counter
    counts = Counter(A.base_id(c) for c in draws)
    tokens = sum(trainset.sizes(c)[0] for c in draws)
    return {"draws": len(draws), "unique_clips": len(counts), "repeated_clips": sum(1 for v in counts.values() if v > 1), "extra_draws": len(draws) - len(counts),
            "important_draw_share": round(sum(1 for c in draws if A.base_id(c) in important) / len(draws), 4), "supervised_tokens": tokens,
            "entity_tokens": sum(trainset.sizes(c)[1] for c in draws), "audio_hours": round(sum(trainset.clips[A.base_id(c)]["duration_seconds"] for c in draws) / 3600, 3),
            "augmented_draws": sum(1 for c in draws if A.is_augmented(c))}


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def fmt(x, digits=4):
    return "N/A" if x is None else f"{x:.{digits}f}"


def evaluate_ce(model, valset, device, dtype, limit=None, scope="decoder"):
    """Teacher-forced validation: eval mode, no grad, token-pooled CE / WCE@2 / EntCE / NonEntCE, overall and per language."""
    C.set_eval_mode(model)
    C.check_mode_policy(model, training=False, scope=scope)
    total, by_lang = C.Sums(), {}
    ids = sorted(valset.clips)[:limit]
    with torch.no_grad():
        for clip_id in ids:
            ex = valset.example(clip_id).to(device, dtype)
            loss = C.token_losses(model, ex)
            total.add(loss, ex.entity)
            by_lang.setdefault(ex.language, C.Sums()).add(loss, ex.entity)
    return {"overall": total.metrics(), "by_language": {k: v.metrics() for k, v in sorted(by_lang.items())}, "clips": len(ids)}


def train_group(model, examples_fn, group, trainset, args, device, dtype, params, update, total, optimiser, sums):
    """One optimiser step over an accumulation group. Returns (pre-clip grad norm, lr, microstep seconds)."""
    denominator = C.group_denominator([trainset.sizes(c) for c in group], args.objective, args.entity_weight)
    started = time.monotonic()
    for clip_id in group:
        ex = examples_fn(clip_id).to(device, dtype)
        loss = C.token_losses(model, ex)
        objective = C.micro_objective(loss, ex.entity, args.objective, args.entity_weight, denominator)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite loss on {clip_id}")
        objective.backward()
        sums.add(loss, ex.entity, float(objective.detach()) * denominator)
        del ex, loss, objective
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("non-finite gradient norm")
    lr = args.lr * C.lr_factor(update, total, args.warmup)
    for g in optimiser.param_groups:
        g["lr"] = lr
    C.check_mode_policy(model, training=True, scope=args.scope)
    optimiser.step()
    if C.has_adapter_full(args.scope):
        C.sync_adapter_master(model)                                    # round the updated FP32 masters into the BF16 working weights
    optimiser.zero_grad(set_to_none=True)
    return grad_norm, lr, time.monotonic() - started


def trim_logs(run_dir, progress):
    """Drop log rows written after the state being resumed from (an interrupted epoch or a crash between checkpoint and state): the structured logs describe exactly one trajectory."""
    keep = {"train_metrics.jsonl": lambda r: r["update"] <= progress["update"],
            "val_metrics.jsonl": lambda r: r["epoch"] < progress["epoch"], "epoch_summary.jsonl": lambda r: r["epoch"] < progress["epoch"],
            "val_midpoints.jsonl": lambda r: r["update"] <= progress["update"],
            "sampling_summary.jsonl": lambda r: r["epoch"] < progress["epoch"] or (progress["group"] > 0 and r["epoch"] == progress["epoch"])}
    removed = {}
    for name, ok in keep.items():
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        kept = [r for r in rows if ok(r)]
        if len(kept) != len(rows):
            path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8", newline="\n")
            removed[name] = len(rows) - len(kept)
    for p in (Path(run_dir) / "checkpoints").glob("epoch-*.complete.json"):
        if int(p.name.split("-")[1].split(".")[0]) >= progress["epoch"]:
            p.unlink()
            removed[p.name] = 1
    return removed


def complete_marker(run_dir, epoch):
    return Path(run_dir) / "checkpoints" / f"epoch-{epoch}.complete.json"


def run(args, stack=None):
    run_dir = Path(args.out_root) / args.run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_bytes((data / name).read_bytes()) != digest:
            raise ValueError(f"{name} differs from the frozen manifest")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_digest = digest_of_config(config)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists() and not args.resume:
        raise SystemExit(f"{run_dir} already holds a run: use a new --run-id, or --resume")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, prompt, load_audio = stack or load_stack(args.model, device, gradient_checkpointing=args.gradient_checkpointing)
    dtype = model.text_decoder.model.embed_tokens.weight.dtype
    tokenizer = processor.tokenizer
    train_clips = [json.loads(l) for l in (data / "clips_train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_clips = [json.loads(l) for l in (data / "clips_validation.jsonl").read_text(encoding="utf-8").splitlines()]
    store = WaveStore(args.batch, {c["call_id"] for c in train_clips + val_clips}, load_audio)
    features = F.FeatureCache.open(args.feature_cache, sha256_bytes((data / "manifest.json").read_bytes()), prompt, F.processor_signature(processor)) if args.feature_cache else None
    trainset = ClipSet(train_clips, tokenizer, store, processor, prompt, load_audio, features, augmenter=A.Augmenter(args.seed) if args.augment == "extras" else None)
    valset = ClipSet(val_clips, tokenizer, store, processor, prompt, load_audio, features)
    for name, ds, clips in (("TRAIN", trainset, train_clips), ("VALIDATION", valset, val_clips)):
        if D.tokens_fingerprint([ds.tokens[c["clip_id"]][0] for c in clips]) != manifest["token_fingerprints"][name]:
            raise ValueError(f"{name} target token ids differ from the frozen manifest: the tokenizer differs from the one used to prepare the data")
    # the targeted pool: "entity" (default, corrected) = clips that contain at least one entity span of the eval engine's entity scoring; "important" = the original broader rule (adds compliance-evidence-only clips)
    important = {c["clip_id"] for c in train_clips if (bool(c["entity_spans"]) if args.target_pool == "entity" else c["important"])}
    plans = plan_for(args, [c["clip_id"] for c in train_clips], important)
    total = total_updates(plans, args.accumulation)
    if args.max_updates:
        total = min(total, args.max_updates)

    scope = args.scope
    if scope == "adapter-full":
        targets = C.inject_adapter_full(model)
    elif scope in C.JOINT_SCOPES:
        targets = C.inject_joint(model, args.rank, args.alpha, args.dropout, args.seed, encoder="encoder" in scope)
    else:
        targets = C.inject_lora(model, scope, args.rank, args.alpha, args.dropout, args.seed)
    counts = C.check_trainable(model, targets, scope)
    if args.encoder_checkpointing:
        if scope != "encoder-lora" and scope != "decoder+adapter+encoder":
            raise SystemExit("--encoder-checkpointing applies to the encoder-lora and decoder+adapter+encoder scopes only")
        print(f"[{args.run_id}] gradient checkpointing on {C.enable_encoder_checkpointing(model)} encoder layers", flush=True)
    params = list(C.trainable_parameters(model, scope).values())
    optimiser = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay, fused=(device == "cuda"))
    get_state = lambda: C.trainable_state(model, scope)      # noqa: E731  the checkpoint payload: LoRA A/B, or the adapter's own parameters
    init_digest = C.state_digest(get_state())
    progress = {"epoch": 1, "group": 0, "update": 0, "elapsed": 0.0}
    state_path = run_dir / "checkpoints" / "latest-state.pt"
    manifest_sha = sha256_bytes((data / "manifest.json").read_bytes())
    if args.resume:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["config_digest"] != config_digest:
            raise ValueError("the resolved configuration differs from the run being resumed")
        original = json.loads(cfg_path.read_text(encoding="utf-8"))
        if original["data_manifest_sha256"] != manifest_sha or saved.get("data_manifest_sha256") != manifest_sha or original["model_revision"] != D.MODEL_REVISION:
            raise ValueError("the data manifest (or model revision) differs from the one this run started with: refusing to resume")
        if C.state_digest(get_state()) != original["init_lora_digest"]:
            raise ValueError("the fresh trainable-parameter initialisation differs from the original run's: refusing to resume")
        removed = trim_logs(run_dir, saved["progress"])
        if removed:
            print(f"[{args.run_id}] resume: removed log rows written after the resumed state: {removed}", flush=True)
        if args.stop_after_epoch and saved["progress"]["epoch"] > args.stop_after_epoch:           # the epoch is already finished: nothing to train, only (re)write its completion marker if all of it exists
            ckpt = run_dir / "checkpoints" / f"epoch-{args.stop_after_epoch}.safetensors"
            rows = [json.loads(l) for l in (run_dir / "epoch_summary.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (run_dir / "epoch_summary.jsonl").is_file() else []
            if ckpt.is_file() and any(r["epoch"] == args.stop_after_epoch for r in rows):
                complete_marker(run_dir, args.stop_after_epoch).write_text(json.dumps({"epoch": args.stop_after_epoch, "update": saved["progress"]["update"], "note": "re-derived on resume"}), encoding="utf-8")
                print(f"EPOCH_DONE {args.stop_after_epoch}", flush=True)
                return saved["progress"]
            raise ValueError(f"state is past epoch {args.stop_after_epoch} but its checkpoint or summary is missing: inspect {run_dir}")
        C.load_trainable_state(model, saved["lora"], scope)
        optimiser.load_state_dict(saved["optimiser"])
        progress = saved["progress"]
        C.restore_rng(saved["rng"])
    else:
        cfg_path.write_text(json.dumps({"config": config, "config_digest": config_digest, "init_lora_digest": init_digest, "targets": len(targets), "parameters": counts,
                                        "plan_updates": total, "augmentation": A.describe() if args.augment != "none" else None, "sampling_summaries": [sampling_summary(p, trainset, important) for p in plans],
                                        "data_manifest_sha256": sha256_bytes((data / "manifest.json").read_bytes()), "model_revision": D.MODEL_REVISION,
                                        "versions": package_versions(), "host": platform.node(), "gpu": gpu_name(), "python": sys.version.split()[0],
                                        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, indent=1), encoding="utf-8")
        torch.manual_seed(args.seed)
    print(f"[{args.run_id}] scope {scope}: trainable parameters {counts['trainable']:,} ({counts['trainable_percent']}% of {counts['total']:,}) in {len(targets)} {'tensors' if scope == 'adapter-full' else 'LoRA modules'}; objective {args.objective}; "
          f"plan {total} optimiser steps; GPU {gpu_name()}", flush=True)

    def log(name, row):
        with open(run_dir / name, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def save_state(epoch, group, update, elapsed, tag="latest"):
        C.atomic_save({"lora": get_state(), "optimiser": optimiser.state_dict(), "progress": {"epoch": epoch, "group": group, "update": update, "elapsed": elapsed},
                       "rng": C.rng_state(), "config_digest": config_digest, "data_manifest_sha256": manifest_sha}, state_path)

    def mid_validate(epoch, step_in_epoch, steps_in_epoch, update):
        """Mid-epoch teacher-forced validation: adapter checkpoint step-<update>.safetensors, then the same evaluate_ce as at epoch ends, logged to val_midpoints.jsonl only (never to epoch_summary)."""
        lora = get_state()
        ckpt = run_dir / "checkpoints" / f"step-{update:04d}.safetensors"
        C.save_lora_safetensors(ckpt, lora, {"run_id": args.run_id, "epoch": epoch, "update": update, "midpoint": True, "rank": args.rank, "alpha": args.alpha, "digest": C.state_digest(lora),
                                              "targets": len(targets), "scope": scope, "model_revision": D.MODEL_REVISION})
        t0 = time.monotonic()
        val = evaluate_ce(model, valset, device, dtype, args.validate_clips, scope)
        seconds = time.monotonic() - t0
        C.set_train_mode(model, scope)
        log("val_midpoints.jsonl", {"run_id": args.run_id, "epoch": epoch, "epoch_fraction": round(step_in_epoch / steps_in_epoch, 4), "step_in_epoch": step_in_epoch, "update": update,
                                    "validation": val["overall"], "validation_by_language": val["by_language"], "validation_clips": val["clips"], "validation_seconds": round(seconds, 1),
                                    "checkpoint": ckpt.name, "checkpoint_digest": C.state_digest(lora)})
        wce = (lambda d: f" WCE@2 {fmt(d['wce2'])}") if args.objective == "wce" else ""
        print(f"[{args.run_id} | Epoch {epoch}/{args.epochs} | Step {step_in_epoch:03d}/{steps_in_epoch}] mid-epoch validation CE {fmt(val['overall']['ce'])}{wce(val['overall']) if args.objective == 'wce' else ''} "
              f"EntCE {fmt(val['overall']['ent_ce'])} NonEntCE {fmt(val['overall']['non_ent_ce'])} | {seconds:.0f}s validation | saved {ckpt.name}", flush=True)

    started = time.monotonic() - progress["elapsed"]
    if not args.resume and not args.skip_baseline_validation:      # epoch 0: teacher-forced validation of the untouched base model (LoRA B = 0), the reference of every later curve
        t0 = time.monotonic()
        base_val = evaluate_ce(model, valset, device, dtype, args.validate_clips, scope)
        row0 = {"run_id": args.run_id, "epoch": 0, "update": 0, "train": None, "validation": base_val["overall"], "validation_by_language": base_val["by_language"], "validation_clips": base_val["clips"],
                "validation_seconds": round(time.monotonic() - t0, 1), "note": "base model before any update"}
        disagreement = None
        if args.epoch0_reference:                                   # the untouched-model validation must reproduce the reference run's epoch 0 (E00 numbers) within a declared tolerance
            ref = next(r for r in (json.loads(l) for l in Path(args.epoch0_reference).read_text(encoding="utf-8").splitlines() if l.strip()) if r["epoch"] == 0)["validation"]
            diffs = {k: abs(base_val["overall"][k] - ref[k]) for k in ("ce", "ent_ce", "non_ent_ce")}
            row0["epoch0_agreement"] = {"reference": str(args.epoch0_reference), "absolute_difference_nats": diffs, "tolerance_nats": args.epoch0_tolerance, "passed": all(d <= args.epoch0_tolerance for d in diffs.values())}
            disagreement = None if row0["epoch0_agreement"]["passed"] else diffs
        log("val_metrics.jsonl", row0)
        log("epoch_summary.jsonl", row0)
        if disagreement is not None:
            raise SystemExit(f"epoch-0 validation differs from the reference beyond {args.epoch0_tolerance} nats: {disagreement}")
        print(f"[{args.run_id} | Epoch 0/{args.epochs}] baseline validation CE {fmt(base_val['overall']['ce'])} EntCE {fmt(base_val['overall']['ent_ce'])} NonEntCE {fmt(base_val['overall']['non_ent_ce'])} | {row0['validation_seconds']:.0f}s", flush=True)
        started = time.monotonic()
    window, epoch_sums = C.Sums(), C.Sums()
    window_seconds, window_updates, since_save = 0.0, 0, 0
    done_epoch = progress["epoch"] - 1
    C.set_train_mode(model, scope)
    for epoch in range(progress["epoch"], args.epochs + 1):
        draws = plans[epoch - 1]
        if args.augment == "extras":
            draws = A.mark_extras(draws, epoch)                    # first occurrence of every clip stays clean, each extra draw is an augmented copy
        groups = D.group_boundaries(len(draws), args.accumulation)
        first = progress["group"] if epoch == progress["epoch"] else 0
        if first == 0:
            log("sampling_summary.jsonl", {"epoch": epoch, **sampling_summary(draws, trainset, important)})
        position = sum(groups[:first])
        mids = {round(len(groups) * k / args.validate_per_epoch) for k in range(1, args.validate_per_epoch)} - {0, len(groups)} if args.validate_per_epoch > 1 else set()
        for gi in range(first, len(groups)):
            update = progress["update"] + 1
            if update > total:
                break
            group = draws[position: position + groups[gi]]
            position += groups[gi]
            sums = C.Sums()
            grad_norm, lr, seconds = train_group(model, trainset.example, group, trainset, args, device, dtype, params, update, total, optimiser, sums)
            progress["update"] = update
            window.merge(sums), epoch_sums.merge(sums)
            window_seconds += seconds
            window_updates += 1
            since_save += 1
            step_in_epoch = gi + 1
            if step_in_epoch % args.log_every == 0 or step_in_epoch == len(groups) or update == total:
                m = window.metrics()
                elapsed = time.monotonic() - started
# the optimised quantity, token-pooled over the window: CE -> CE; WCE@w -> sum(w*loss)/sum(w)
                objective_loss = m["ce"] if args.objective == "ce" else (window.v["loss_all"] + (args.entity_weight - 1) * window.v["loss_ent"]) / (m["tokens"] + (args.entity_weight - 1) * m["entity_tokens"])
                row = {"run_id": args.run_id, "epoch": epoch, "step_in_epoch": step_in_epoch, "steps_in_epoch": len(groups), "update": update,
                       "objective_type": "ce" if args.objective == "ce" else f"wce@{args.entity_weight:g}", "objective_loss": objective_loss, **m, "lr": lr, "grad_norm": grad_norm,
                       "elapsed_seconds": round(elapsed, 1), "seconds_per_update": round(window_seconds / window_updates, 3), "peak_vram_gb": peak_gb()}
                log("train_metrics.jsonl", row)
                head = f"[{args.run_id} | Epoch {epoch}/{args.epochs} | Step {step_in_epoch:03d}/{len(groups)}]"
                tail = f"EntCE: {fmt(m['ent_ce'])} | NonEntCE: {fmt(m['non_ent_ce'])} | LR: {lr:.3e} | GradNorm: {grad_norm:.3f} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
                if args.objective == "ce":
                    print(f"{head} CE (Loss): {fmt(m['ce'])} | {tail}", flush=True)
                else:
                    print(f"{head} CE: {fmt(m['ce'])} | WCE@{args.entity_weight:g} (Loss): {fmt(objective_loss)} | {tail}", flush=True)
                window, window_seconds, window_updates = C.Sums(), 0.0, 0
            if since_save >= args.save_every:
                save_state(epoch, gi + 1, update, time.monotonic() - started)
                since_save = 0
            if step_in_epoch in mids and update < total:
                mid_validate(epoch, step_in_epoch, len(groups), update)
            if update == total:
                break
        progress["group"] = 0
        done_epoch = epoch
        # ---- epoch end: lightweight LoRA checkpoint, teacher-forced validation, structured summary
        lora = get_state()
        ckpt = run_dir / "checkpoints" / f"epoch-{epoch}.safetensors"
        C.save_lora_safetensors(ckpt, lora, {"run_id": args.run_id, "epoch": epoch, "update": progress["update"], "rank": args.rank, "alpha": args.alpha, "digest": C.state_digest(lora),
                                              "targets": len(targets), "scope": scope, "model_revision": D.MODEL_REVISION})
        t0 = time.monotonic()
        val = evaluate_ce(model, valset, device, dtype, args.validate_clips, scope)
        val_seconds = time.monotonic() - t0
        C.set_train_mode(model, scope)
        tm = epoch_sums.metrics()
        summary = {"run_id": args.run_id, "epoch": epoch, "update": progress["update"], "train": tm, "validation": val["overall"], "validation_by_language": val["by_language"],
                   "validation_clips": val["clips"], "validation_seconds": round(val_seconds, 1), "elapsed_seconds": round(time.monotonic() - started, 1),
                   "peak_vram_gb": peak_gb(), "checkpoint": ckpt.name, "checkpoint_digest": C.state_digest(lora), "gpu": gpu_name()}
        log("val_metrics.jsonl", summary)
        log("epoch_summary.jsonl", summary)
        wce_part = (lambda d: f" WCE@2 {fmt(d['wce2'])}") if args.objective == "wce" else (lambda d: "")     # the fixed diagnostic is stored for every run but printed only when it is optimised
        print(f"[{args.run_id} | Epoch {epoch}/{args.epochs}] train CE {fmt(tm['ce'])}{wce_part(tm)} EntCE {fmt(tm['ent_ce'])} NonEntCE {fmt(tm['non_ent_ce'])} | "
              f"validation CE {fmt(val['overall']['ce'])}{wce_part(val['overall'])} EntCE {fmt(val['overall']['ent_ce'])} NonEntCE {fmt(val['overall']['non_ent_ce'])} | "
              f"{val_seconds:.0f}s validation | peak {summary['peak_vram_gb']} GB | saved {ckpt.name}", flush=True)
        epoch_sums = C.Sums()
        progress["epoch"] = epoch + 1
        save_state(epoch + 1, 0, progress["update"], time.monotonic() - started)
        since_save = 0
        complete_marker(run_dir, epoch).write_text(json.dumps({"epoch": epoch, "update": progress["update"], "checkpoint": ckpt.name, "checkpoint_digest": summary["checkpoint_digest"],
                                                                "data_manifest_sha256": manifest_sha, "written_after": ["checkpoint", "validation", "logs", "resumable state"]}), encoding="utf-8")
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(f"EPOCH_DONE {epoch}", flush=True)
            break
        if progress["update"] >= total:
            break
    (run_dir / "resource.json").write_text(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 1), "peak_vram_gb": peak_gb(),
                                                       "gpu": gpu_name(), "updates": progress["update"], "last_epoch": done_epoch}, indent=1), encoding="utf-8")
    return progress


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--objective", choices=("ce", "wce"), required=True)
    p.add_argument("--scope", choices=C.SCOPES, default="decoder", help="what is trained (the rest stays frozen): decoder = attention + MLP LoRA on the decoder (E01-E04); adapter-lora = LoRA on every Linear of the speech adapter; "
                   "adapter-full = every speech-adapter parameter (BF16 computation, FP32 master weights that AdamW updates; lr applies to the masters; native adapter dropout off); encoder-lora = LoRA on the speech encoder's q/k/v/out_proj + fc1/fc2")
    p.add_argument("--entity-weight", type=float, default=2.0)
    p.add_argument("--sampling", choices=("uniform", "targeted"), required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--target-pool", choices=("entity", "important"), default="entity", help="clips the targeted extra draws come from: entity = clips with entity spans (the spans the entity WER is computed on); important = the original broader rule (also compliance-evidence-only clips)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-5, help="peak learning rate. Default 2.5e-5: the best of the 5e-5 / 2.5e-5 / 1e-5 sweep (E01/E02 used 5e-5, E0x-LR25 2.5e-5, E0x-LR10 1e-5)")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--validate-clips", type=int, default=None, help="limit teacher-forced validation to the first N clips (smoke only)")
    p.add_argument("--max-updates", type=int, default=None, help="stop the plan after N optimiser steps (smoke only; also shortens the LR schedule)")
    p.add_argument("--feature-cache", type=Path, default=None, help="folder built by sft_features.py: the identical per-clip processor output (verified bit-for-bit) instead of recomputing it every step")
    p.add_argument("--validate-per-epoch", type=int, default=2, help="teacher-forced validation this many times per epoch: always at the epoch end, plus evenly spaced mid-epoch points (2 = every half epoch). Each mid-epoch point also saves checkpoints/step-<update>.safetensors and logs to val_midpoints.jsonl. Validation never updates weights or consumes random numbers, so training is unchanged by it")
    p.add_argument("--epoch0-reference", type=Path, default=None, help="epoch_summary.jsonl of a run whose epoch-0 row (the untouched model, E00) this run's epoch-0 validation must reproduce; the run stops before training if it does not")
    p.add_argument("--epoch0-tolerance", type=float, default=1e-3, help="absolute tolerance (nats) on the overall CE, entity CE and non-entity CE of the epoch-0 agreement check")
    p.add_argument("--stop-after-epoch", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-baseline-validation", action="store_true", help="smoke/tests only: skip the epoch-0 validation of the base model")
    p.add_argument("--gradient-checkpointing", action="store_true", help="decoder layers (only effective while the decoder is in train mode, i.e. --scope decoder)")
    p.add_argument("--augment", choices=("none", "extras"), default="none", help="extras: the extra draws of the matched budget (second occurrences of a clip in an epoch) are replaced by augmented copies "
                   "(synthetic noise at 0-20 dB SNR, telephone, muffled; sft_augment.py); every clip is still seen clean once per epoch and the budget, schedule and sampling are unchanged")
    p.add_argument("--encoder-checkpointing", action="store_true", help="recompute each speech-encoder layer in the backward pass (--scope encoder-lora only): less memory, more compute; the encoder stays in eval mode")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
