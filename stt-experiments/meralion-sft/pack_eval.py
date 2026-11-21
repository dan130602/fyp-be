"""Pack one finished evaluation into a few files (the scratch area has a FILE-count quota of 100,000).

    python pack_eval.py --out <eval dir> --engine-runs <evaluation-engine/runs>

Needs <out>/metrics.json (the evaluation completed). Writes <out>/engine_runs.tar.gz (the scorer's run folders named in engine_runs.json) and <out>/worker.tar.gz (worker documents, state, logs),
verifies both archives, then removes the packed folders and the prediction copies (exact paths only: names must start with sft- and contain -prelim-). Nothing is lost: unpack and re-score anywhere.
"""
import argparse
import json
import shutil
import tarfile
from pathlib import Path


def pack(out, engine_runs):
    out, engine_runs = Path(out), Path(engine_runs)
    if not (out / "metrics.json").is_file():
        raise SystemExit("no metrics.json: the evaluation is not complete, nothing packed")
    runs = [Path(v).name for v in json.loads((out / "engine_runs.json").read_text(encoding="utf-8")).values()]
    for name in runs:
        if not (name.startswith("sft-") and "-prelim-" in name and (engine_runs / name).is_dir()):
            raise SystemExit(f"unexpected engine run name {name!r}")
    with tarfile.open(out / "engine_runs.tar.gz", "w:gz") as tar:
        for name in runs:
            tar.add(engine_runs / name, arcname=name)
    with tarfile.open(out / "worker.tar.gz", "w:gz") as tar:
        tar.add(out / "worker", arcname="worker")
    for archive, expected in (("engine_runs.tar.gz", runs), ("worker.tar.gz", ["worker"])):
        with tarfile.open(out / archive) as tar:
            top = {m.name.split("/")[0] for m in tar.getmembers()}
        if top != set(expected):
            raise SystemExit(f"{archive} does not contain exactly {expected}")
    for name in runs:
        shutil.rmtree(engine_runs / name)
    shutil.rmtree(out / "worker")
    for folder in ("predictions_raw", "predictions_percentage", "predictions_final"):
        if (out / folder).is_dir():
            shutil.rmtree(out / folder)
    print(json.dumps({"packed": runs + ["worker"], "out": str(out)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--engine-runs", type=Path, required=True)
    a = p.parse_args()
    pack(a.out, a.engine_runs)
