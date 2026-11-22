"""Label SFT predictions as SFT before they are imported (the worker-document converter always writes adapter=None, training=False, which would make a fine-tuned model look like the base model).

    python stamp_sft_predictions.py --in <predictions folder from to_predictions.py> --out <new folder> --run-dir <fetched runs/RUN_ID> --epoch 2

Reads the training run's config.json and epoch_summary.jsonl (checkpoint digest) and writes, into settings of every CALL_x.json: adapter = {method (lora | full-adapter), scope, merged, run_id, epoch, checkpoint, checkpoint_digest, rank, alpha,
learning_rate, objective, entity_weight, sampling, k, seed}, training = true. Nothing else in a file changes. Import with: import_predictions.py --allow-adapter ...
"""
import argparse
import json
from pathlib import Path


def stamp(src, dst, run_dir, epoch):
    run_dir, src, dst = Path(run_dir), Path(src), Path(dst)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    rows = [json.loads(l) for l in (run_dir / "epoch_summary.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    row = next((r for r in rows if r["epoch"] == epoch), None)
    if row is None:
        raise SystemExit(f"{run_dir.name} has no epoch {epoch} in epoch_summary.jsonl")
    scope = config.get("scope", "decoder")                       # runs from before scopes existed are decoder LoRA
    full = scope == "adapter-full"
    adapter = {"method": {"adapter-full": "full-adapter", "decoder+adapter": "lora+full-adapter", "decoder+adapter+encoder": "lora+full-adapter"}.get(scope, "lora"), "scope": scope, "merged_into_bf16_weights": True, "run_id": config["run_id"], "epoch": epoch, "checkpoint": row["checkpoint"],
               "checkpoint_digest": row["checkpoint_digest"], "rank": None if full else config["rank"], "alpha": None if full else config["alpha"], "learning_rate": config["lr"], "objective": config["objective"],
               "entity_weight": config["entity_weight"] if config["objective"] == "wce" else None, "sampling": config["sampling"], "k": config["k"], "seed": config["seed"]}
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("CALL_*.json"))
    if not files:
        raise SystemExit(f"no CALL_*.json in {src}")
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["settings"] = {**doc["settings"], "adapter": adapter, "training": True}
        (dst / path.name).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(files), adapter


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, required=True)
    args = p.parse_args(argv)
    n, adapter = stamp(args.src, args.out, args.run_dir, args.epoch)
    print(json.dumps({"stamped": n, "adapter": adapter}, ensure_ascii=False))


if __name__ == "__main__":
    main()
