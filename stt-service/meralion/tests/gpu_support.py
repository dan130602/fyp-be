"""Test helper for REV 4.1 A3: write a sampler file, a run-meta and a bound gpu-scope.json into a run folder, as run_worker does on the cluster."""
import importlib.util
import json
import types
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
spec = importlib.util.spec_from_file_location("gpu_scope_support", TOOLS / "gpu_scope.py")
gpu_scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_scope)

HEADER = "timestamp, name, uuid, memory.used [MiB], memory.total [MiB], utilization.gpu [%]"


def no_nvidia_smi(command, **kwargs):
    return types.SimpleNamespace(returncode=1, stdout="", stderr="no nvidia-smi on this machine")


def write_scope(folder, *, used=40000, total=46068, allocated="GPU-1", parent=None, sampler_uuid=None, record=True, rows=None):
    """gpu.csv + run-meta.json (+ gpu-scope.json) in `folder`. `sampler_uuid` is the uuid the sampler's rows carry (default: the allocated one); `parent` is the parent card's uuid (default: the allocated one)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    uuid = sampler_uuid if sampler_uuid is not None else allocated
    lines = rows if rows is not None else [f"2026/11/17 10:00:00.0, L40S, {uuid}, {used} MiB, {total} MiB, 0 %"]
    (folder / "gpu.csv").write_text(HEADER + chr(10) + chr(10).join(lines) + chr(10), encoding="utf-8")
    meta_path = folder / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.update(gpu_uuid=allocated, parent_gpu_uuid=parent or allocated)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    if record:
        (folder / gpu_scope.SCOPE_FILE).write_text(json.dumps(gpu_scope.build_record(meta, folder / "gpu.csv", run=no_nvidia_smi, env={})), encoding="utf-8")
    return folder / "gpu.csv"
