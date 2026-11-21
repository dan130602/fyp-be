"""Merge a lightweight LoRA checkpoint into the pinned base weights and write a standalone checkpoint + model manifest + runtime config for the existing worker.

    python sft_export.py --base <pinned model dir> --lora <epoch-N.safetensors> --out <export dir> --id <manifest id> [--base-manifest <json>] [--runtime-config-in <json> --runtime-config-out <json>]

CPU only (safetensors streaming, no model instantiation): W' = bf16(fp32(W) + (alpha/r) * B @ A) for exactly the projections the LoRA covers (182 decoder projections, the 2 speech-adapter
Linears or the 192 encoder projections, whatever the checkpoint names); an adapter-full checkpoint (no lora_A/B keys) replaces the named adapter tensors by bf16(fp32 master weight).
Every other tensor is copied byte for byte. The base directory is never modified. Files not needed to load or hash-check the model (README, assets) are not copied. The manifest follows the worker's
`model-manifest-v1` schema (kind complete_checkpoint, status complete, per-file sha256 and size) and records its derivation; it has no `upstream_verification` because it is not upstream.
The merge is rounded to bf16: adapters whose deltas are below bf16 resolution of a weight lose that part, so teacher-forced CE of the merged model is checked against the LoRA model in the smoke.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import sft_core as C

COPY_FILES = ("config.json", "configuration_meralion3.py", "modeling_meralion3.py", "processing_meralion3.py", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "model.safetensors.index.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_tensor(weight, a, b, scale):
    return (weight.float() + scale * (b.float() @ a.float())).to(weight.dtype)


def expected_modules(scope, weight_names):
    """The Linear modules a LoRA checkpoint of this scope must cover, taken from the base model's own tensor names."""
    mods = {k[:-len(".weight")] for k in weight_names if k.endswith(".weight")}
    parts = {"decoder": ("decoder",), "decoder+adapter": ("decoder",), "decoder+adapter+encoder": ("decoder", "encoder"), "encoder-lora": ("encoder",), "adapter-lora": ("adapter",)}[scope]
    patterns = {"decoder": C.DECODER_TARGET_PATTERN, "encoder": C.ENCODER_TARGET_PATTERN}
    out = set()
    for part in parts:
        out |= {m for m in mods if m.startswith("speech_audio_adapter.")} if part == "adapter" else {m for m in mods if re.fullmatch(patterns[part], m)}
    return out


def validate_payload(lora, meta, weight_map):
    """Enforce the DECLARED scope (metadata `scope`; checkpoints written before scopes existed are decoder LoRA): the checkpoint must hold exactly the complete tensor set of that scope,
    in the right form (adapter-full: whole tensors only; LoRA scopes: paired A/B matrices only), finite and FP32. Missing, extra (out-of-scope) or mixed tensors are refused before anything is written."""
    scope = meta.get("scope", "decoder")
    if scope not in C.SCOPES:
        raise SystemExit(f"unknown scope {scope!r} (expected one of {C.SCOPES})")
    lora_keys = [k for k in lora if ".lora_" in k]
    plain_keys = [k for k in lora if ".lora_" not in k]
    want_modules = set() if scope == "adapter-full" else expected_modules(scope, weight_map)       # LoRA part
    want_full = {k for k in weight_map if k.startswith("speech_audio_adapter.")} if C.has_adapter_full(scope) else set()                      # whole-tensor part
    if not want_modules and not want_full:
        raise SystemExit(f"the base model has no tensors for scope {scope}")
    if lora_keys and not want_modules:
        raise SystemExit(f"mixed payload: a {scope} checkpoint must not contain LoRA matrices (found {lora_keys[:2]})")
    if plain_keys and not want_full:
        raise SystemExit(f"mixed payload: a {scope} checkpoint must hold LoRA A/B matrices only (found {plain_keys[:2]})")
    malformed = [k for k in lora_keys if not (k.endswith(".lora_A") or k.endswith(".lora_B"))]
    if malformed:
        raise SystemExit(f"mixed payload: malformed LoRA tensor names {malformed[:2]}")
    got_modules = {k.rsplit(".lora_", 1)[0] for k in lora_keys}
    unpaired = sorted(m for m in got_modules if f"{m}.lora_A" not in lora or f"{m}.lora_B" not in lora)
    if unpaired:
        raise SystemExit(f"LoRA checkpoint has an unpaired A or B matrix: {unpaired[:2]}")
    want, got = want_modules | want_full, got_modules | set(plain_keys)
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        raise SystemExit(f"the checkpoint does not match its declared scope {scope}: {len(missing)} tensors missing {missing[:3]}, {len(extra)} outside the scope {extra[:3]}")
    bad = sorted(k for k, t in lora.items() if t.dtype != torch.float32 or not bool(torch.isfinite(t).all()))
    if bad:
        raise SystemExit(f"checkpoint tensors must be finite FP32: {bad[:3]}")
    return scope


def export(base, lora_path, out, manifest_id, base_manifest=None, source_note=""):
    base, out = Path(base), Path(out)
    if (base / "adapter_config.json").exists() or (out / "adapter_config.json").exists():
        raise SystemExit("adapter_config.json present: complete checkpoints only")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: exports are written once")
    with safe_open(str(lora_path), framework="pt") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}
        meta = dict(f.metadata() or {})
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    scope = validate_payload(lora, meta, index["weight_map"])
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})              # LoRA modules: merged into the base weight
    full_keys = sorted(k for k in lora if ".lora_" not in k)                                   # whole FP32 tensors (the adapter): replace the base ones, rounded to the base dtype
    if modules:
        rank, alpha = int(json.loads(meta["rank"])), float(json.loads(meta["alpha"]))
        scale = alpha / rank
    else:
        rank = alpha = scale = None
    wanted = {**{f"{m}.weight": ("lora", m) for m in modules}, **{k: ("full", k) for k in full_keys}}
    unknown = sorted(set(wanted) - set(index["weight_map"]))
    if unknown:
        raise SystemExit(f"checkpoint tensors without a base weight: {unknown[:4]}")
    for shard in sorted({index["weight_map"][k] for k in wanted}):   # shapes are checked against the base BEFORE any output is written
        with safe_open(str(base / shard), framework="pt") as f:
            for key in (k for k in wanted if index["weight_map"][k] == shard):
                shape = tuple(f.get_slice(key).get_shape())
                kind, m = wanted[key]
                if kind == "full":
                    ok = tuple(lora[m].shape) == shape
                else:
                    a, b = lora[f"{m}.lora_A"], lora[f"{m}.lora_B"]
                    ok = a.dim() == 2 and b.dim() == 2 and a.shape[0] == b.shape[1] == rank and (b.shape[0], a.shape[1]) == shape
                if not ok:
                    raise SystemExit(f"{m}: checkpoint shapes do not fit the base weight {shape}")
    out.mkdir(parents=True, exist_ok=True)
    merged, delta_stats = 0, {"max_abs_delta": 0.0, "sum_sq_delta": 0.0, "sum_sq_weight": 0.0, "changed_after_rounding": 0, "elements": 0}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(str(base / shard), framework="pt") as f:
            for key in f.keys():
                w = f.get_tensor(key)
                if key in wanted:
                    kind, m = wanted[key]
                    if kind == "full":
                        new = lora[m].to(w.dtype)
                    else:
                        new = merge_tensor(w, lora[f"{m}.lora_A"], lora[f"{m}.lora_B"], scale)
                    d = (new.float() - w.float())
                    delta_stats["max_abs_delta"] = max(delta_stats["max_abs_delta"], float(d.abs().max()))
                    delta_stats["sum_sq_delta"] += float((d ** 2).sum())
                    delta_stats["sum_sq_weight"] += float((w.float() ** 2).sum())
                    delta_stats["changed_after_rounding"] += int((d != 0).sum())
                    delta_stats["elements"] += d.numel()
                    w, merged = new, merged + 1
                tensors[key] = w.contiguous()
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        del tensors
    if merged != len(wanted):
        raise SystemExit(f"merged {merged} of {len(wanted)} weights")
    for name in COPY_FILES:
        shutil.copy2(base / name, out / name)
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size, "source": "sft export (merged)" if p.suffix == ".safetensors" else f"copied unchanged from {base.name}"}
             for p in sorted(out.iterdir()) if p.is_file()}
    base_doc = json.loads(Path(base_manifest).read_text(encoding="utf-8")) if base_manifest else {}
    manifest = {"schema": "model-manifest-v1", "id": manifest_id, "repo": base_doc.get("repo", "MERaLiON/MERaLiON-3-3B-ASR"),
                "revision": f"{base_doc.get('revision', '9f226d738dee62c60a2f43b0ab7b9a2585a132c3')}+sft:{meta.get('digest', '')[:12]}", "kind": "complete_checkpoint", "status": "complete",
                "files_listing_complete": True, "files": files,
                "derivation": {"base_manifest": base_doc.get("id"), "base_revision": base_doc.get("revision"), "lora_file_sha256": sha256_file(lora_path), "lora_digest": meta.get("digest"),
                               "run_id": meta.get("run_id"), "epoch": meta.get("epoch"), "scope": meta.get("scope", "decoder"), "rank": rank, "alpha": alpha, "merged_tensors": merged,
                               "merge_dtype": ("bfloat16 (FP32 master weights rounded)" if not modules else "bfloat16 (fp32 accumulate; adapter FP32 masters rounded)" if full_keys else "bfloat16 (fp32 accumulate)"),
                               "delta": {"max_abs": delta_stats["max_abs_delta"], "relative_rms": (delta_stats["sum_sq_delta"] / delta_stats["sum_sq_weight"]) ** 0.5,
                                         "elements_changed_by_rounding_share": delta_stats["changed_after_rounding"] / delta_stats["elements"]}, "note": source_note}}
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--id", required=True, help="model manifest id, e.g. meralion-3-3b-asr-sft-E01-ep1")
    p.add_argument("--base-manifest", type=Path)
    p.add_argument("--manifest-out", type=Path, help="default: <out>.manifest.json next to the export folder")
    p.add_argument("--runtime-config-in", type=Path, help="the frozen runtime config to copy with the new model_manifest id")
    p.add_argument("--runtime-config-out", type=Path)
    args = p.parse_args(argv)
    manifest = export(args.base, args.lora, args.out, args.id, args.base_manifest)
    target = args.manifest_out or args.out.with_name(args.out.name + ".manifest.json")
    target.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.runtime_config_in:
        cfg = json.loads(args.runtime_config_in.read_text(encoding="utf-8"))
        cfg["model_manifest"] = args.id
        cfg["config_id"] = cfg["config_id"] + "-" + args.id
        args.runtime_config_out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"export": str(args.out), "manifest": str(target), "files": len(manifest["files"]), "delta": manifest["derivation"]["delta"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
