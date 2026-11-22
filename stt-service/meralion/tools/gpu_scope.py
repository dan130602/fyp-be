"""Which device did a GPU-memory measurement measure, and may it be called a slice measurement? (REV 4.1 A3.) Standard library only (pynvml is optional and only for the NVML method).

    python gpu_scope.py --run-meta run-meta.json --csv gpu.csv --out gpu-scope.json [--torch-total-mib N]

Why: on a MIG node `nvidia-smi` reports memory.used per slice but memory.total of the PARENT card, and nothing says that an `-i MIG-...` query measured the slice at all. Nearby numbers cannot establish which
device was measured, so the order is: (1) IDENTITY: the device UUID the measurement RETURNS must equal the UUID the scheduler ALLOCATED (the first CUDA_VISIBLE_DEVICES entry, recorded in run-meta.json as gpu_uuid);
a missing, empty, [N/A] or PARENT UUID cannot give a verified scope; (2) CAPACITY: the reported memory.total must be valid and, for a MIG slice, clearly below the parent card's (and within 5 % of torch's figure when
one is given); (3) NUMBERS (smoke only): used memory agrees with torch.cuda.mem_get_info within max(256 MiB, 3 %): a sanity check, never a substitute for 1 and 2.
Result `scope`: slice_verified (a MIG slice) | device_verified (a whole GPU) | scope_unverified (+ reason) | unavailable. CUDA_VISIBLE_DEVICES is never read for writing here and never changed.

The record (gpu-scope.json) is bound to ONE run: the allocated device, the method, the sampler file's sha256. memory_report() returns a number as `peak_mib` ONLY for a verified, correctly bound record; otherwise
peak_mib is None and the raw sampler maximum is reported separately as `device_reported_peak_mib`, labelled as not a slice measurement. An unavailable or unverified scope never changes a functional verdict.

REV 4.2: the record's scope LABEL is never copied. memory_report() derives the scope again from the sampler bytes the record is bound to, the allocated identity in run-meta and the capacities the record holds, and
the record must agree (else `record_contradicts_sampler`, no number). A row that reports more memory used than its total is impossible and makes the stream unverified. A malformed record or run-meta is unavailable
evidence, never an exception. LIMIT: this is a consistency check, not tamper-proofing: the parent card's total is read live at run time and cannot be re-derived offline, so a record edited together with its own
parent capacity and sampler hash is not detected.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

SCOPE_FILE = "gpu-scope.json"
CSV_METHOD = "nvidia-smi -i"
NVML_METHOD = "nvml-by-uuid"
PARENT_FRACTION = 0.9                  # a total at or above this fraction of the parent card's is "parent-sized"
TORCH_TOLERANCE = 0.05
NUMBERS_ABSOLUTE_MIB, NUMBERS_RELATIVE = 256, 0.03
VERIFIED = ("slice_verified", "device_verified")
PREAMBLE = ("#", "GPU ", "  MIG", "host:", "CUDA_", "SLURM_")
UNVERIFIED_LABEL = "device-reported (nvidia-smi), scope NOT verified: not a slice measurement"
VERIFIED_LABEL = {"slice_verified": "slice memory.used (nvidia-smi, driver level: a reservation for vLLM, not a minimum)",
                  "device_verified": "whole-GPU memory.used (nvidia-smi, driver level: a reservation for vLLM, not a minimum)"}


def uuid_kind(value):
    if not isinstance(value, str):                                                                # REV 4.2: a list, number or null from a malformed file is never an identity
        return None
    value = value.strip()
    return "mig" if re.fullmatch(r"MIG-[0-9A-Za-z-]+", value) else "gpu" if re.fullmatch(r"GPU-[0-9A-Za-z-]+", value) else None


# ---- the sampler file ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mib(cell):
    """`254 MiB` -> 254; None for [N/A], [Not Supported], text or an empty cell."""
    match = re.fullmatch(r"\s*(\d+)\s*MiB\s*", cell or "")
    return int(match.group(1)) if match else None


def parse_sampler(text):
    """-> {"rows": [{"timestamp", "name", "uuid", "used_mib", "total_mib"}], "data_lines", "malformed", "not_available", "impossible"}. Only well-formed, POSSIBLE rows are rows; error text, truncated lines and
    [N/A] values are counted, never turned into numbers; a row that reports more memory USED than its own TOTAL (or a total of 0) is counted as `impossible` (REV 4.2) and is never a reading."""
    rows, malformed, not_available, data_lines, impossible = [], 0, 0, 0, 0
    columns = None
    for row in csv.reader(line for line in text.splitlines() if line.strip() and not line.startswith(PREAMBLE)):
        cells = [cell.strip() for cell in row]
        if any("memory.used" in cell for cell in cells):
            names = [re.sub(r"\s*\[.*\]\s*$", "", cell) for cell in cells]
            columns = {name: names.index(name) for name in ("timestamp", "name", "uuid", "memory.used", "memory.total") if name in names}
            continue
        data_lines += 1
        if columns is None or not {"memory.used", "memory.total"} <= set(columns) or len(cells) <= max(columns.values()):
            malformed += 1
            continue
        used, total = mib(cells[columns["memory.used"]]), mib(cells[columns["memory.total"]])
        if used is None or total is None:
            raw = [cells[columns["memory.used"]], cells[columns["memory.total"]]]
            if any(item.startswith("[") for item in raw):
                not_available += 1
            else:
                malformed += 1
            continue
        if total <= 0 or used > total:
            impossible += 1
            continue
        uuid = cells[columns["uuid"]] if "uuid" in columns else ""
        rows.append({"timestamp": cells[columns["timestamp"]] if "timestamp" in columns else None, "name": cells[columns["name"]] if "name" in columns else None,
                     "uuid": uuid if uuid_kind(uuid) else "", "used_mib": used, "total_mib": total})
    return {"rows": rows, "data_lines": data_lines, "malformed": malformed, "not_available": not_available, "impossible": impossible}


def raw_peak_mib(csv_path):
    """The largest memory.used in a sampler file, whatever device it describes (None when there is no readable row). NOT a slice measurement by itself: see memory_report().
    An impossible reading (used above total) is never the maximum, and a file that has one is never re-read in the legacy layout."""
    if not csv_path or not Path(csv_path).is_file():
        return None
    parsed = parse_sampler(Path(csv_path).read_text(encoding="utf-8", errors="replace"))
    values = [row["used_mib"] for row in parsed["rows"]]
    if values:
        return max(values)
    if parsed["impossible"]:
        return None
    peak, column = None, 2                                                           # the oldest layout (timestamp, name, memory.used, no uuid column)
    for row in csv.reader(line for line in Path(csv_path).read_text(encoding="utf-8", errors="replace").splitlines() if not line.startswith(PREAMBLE)):
        if any("memory.used" in cell for cell in row):
            column = next(i for i, cell in enumerate(row) if "memory.used" in cell)
        elif len(row) > column and mib(row[column]) is not None:
            peak = mib(row[column]) if peak is None else max(peak, mib(row[column]))
    return peak


# ---- the verdict on one measurement ---------------------------------------------------------------------------------------------------------------------------------------------------------

def evaluate(*, allocated, parent, parsed, parent_total_mib=None, torch_total_mib=None, numbers=None):
    """IDENTITY first, then capacity, then (when given) the numbers. -> {"scope", "reason", "returned_uuids", "memory_total_mib", "parent_total_mib", "device_kind"}.
    `parsed` is parse_sampler()'s result; `numbers` is {"measured_used_mib", "torch_used_mib"}."""
    kind = uuid_kind(allocated)
    totals = sorted({row["total_mib"] for row in parsed["rows"]})
    result = {"scope": None, "reason": None, "returned_uuids": sorted({row["uuid"] for row in parsed["rows"] if row["uuid"]}), "memory_total_mib": totals[-1] if totals else None,
              "parent_total_mib": parent_total_mib, "device_kind": {"mig": "mig_slice", "gpu": "whole_gpu"}.get(kind)}

    def decide(scope, reason=None):
        result.update(scope=scope, reason=reason)
        return result
    if parsed.get("impossible"):
        return decide("scope_unverified", "memory_used_exceeds_total")                          # REV 4.2: a stream that reports more memory used than exists cannot be verified, however the identity looks
    if not parsed["rows"]:
        return decide("unavailable", "no_valid_rows" if not parsed["not_available"] else "memory_not_available")
    if kind is None:
        return decide("scope_unverified", "identity_missing")                                   # the allocated device's own identity is unknown
    returned = set(result["returned_uuids"])
    if not returned or any(not row["uuid"] for row in parsed["rows"]):
        return decide("scope_unverified", "identity_missing")                                   # a row without a UUID: the device is unknown
    if returned != {allocated}:
        return decide("scope_unverified", "parent_identity" if parent and returned == {parent} and parent != allocated else "identity_mismatch")
    if len(totals) != 1 or totals[0] <= 0:
        return decide("scope_unverified", "capacity_inconsistent")
    total = totals[0]
    if kind == "mig":
        if not parent_total_mib:
            return decide("scope_unverified", "parent_capacity_unknown")
        if total >= PARENT_FRACTION * parent_total_mib:
            return decide("scope_unverified", "capacity_parent_sized")
    if torch_total_mib and abs(total - torch_total_mib) > TORCH_TOLERANCE * torch_total_mib:
        return decide("scope_unverified", "capacity_inconsistent")
    if numbers:
        measured, expected = numbers["measured_used_mib"], numbers["torch_used_mib"]
        if abs(measured - expected) > max(NUMBERS_ABSOLUTE_MIB, NUMBERS_RELATIVE * expected):
            return decide("scope_unverified", "numbers_disagree")
    return decide("slice_verified" if kind == "mig" else "device_verified")


def nvml_probe(allocated, nvml=None):
    """The NVML method: a handle taken BY the allocated UUID, then its own UUID, memory and (for a MIG handle) its parent. Failure is recorded, never raised. `nvml` is an injectable pynvml-like module."""
    try:
        if nvml is None:
            import pynvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByUUID(allocated)
        text = lambda value: value.decode() if isinstance(value, bytes) else str(value)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        found = {"available": True, "error": None, "returned_uuid": text(nvml.nvmlDeviceGetUUID(handle)), "memory_total_mib": int(info.total) // 2 ** 20, "memory_used_mib": int(info.used) // 2 ** 20,
                 "is_mig_handle": bool(nvml.nvmlDeviceIsMigDeviceHandle(handle)) if hasattr(nvml, "nvmlDeviceIsMigDeviceHandle") else None, "parent_uuid": None, "parent_total_mib": None}
        if found["is_mig_handle"] and hasattr(nvml, "nvmlDeviceGetDeviceHandleFromMigDeviceHandle"):
            parent = nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
            found["parent_uuid"] = text(nvml.nvmlDeviceGetUUID(parent))
            found["parent_total_mib"] = int(nvml.nvmlDeviceGetMemoryInfo(parent).total) // 2 ** 20
        return found
    except Exception as error:                                                                 # an unsupported call, a missing library, a permission error: a recorded fact
        return {"available": False, "error": f"{type(error).__name__}: {str(error)[:160]}", "returned_uuid": None, "memory_total_mib": None, "memory_used_mib": None,
                "is_mig_handle": None, "parent_uuid": None, "parent_total_mib": None}


def evaluate_nvml(allocated, probe, parent=None):
    """The same identity -> capacity verdict for the NVML reading (a MIG handle must say it is one)."""
    if not probe.get("available"):
        return {"scope": "unavailable", "reason": "nvml_unavailable"}
    if uuid_kind(allocated) == "mig" and probe.get("is_mig_handle") is not True:
        return {"scope": "scope_unverified", "reason": "not_a_mig_handle"}                  # the handle taken by a MIG uuid must itself be a MIG handle
    used, total = probe.get("memory_used_mib"), probe.get("memory_total_mib")
    if not (isinstance(used, int) and isinstance(total, int)) or isinstance(used, bool) or isinstance(total, bool):
        return {"scope": "unavailable", "reason": "nvml_memory_not_available"}
    possible = total > 0 and used <= total
    rows = [{"timestamp": None, "name": None, "uuid": probe["returned_uuid"] if uuid_kind(probe["returned_uuid"]) else "", "used_mib": used, "total_mib": total}] if possible else []
    return evaluate(allocated=allocated, parent=probe.get("parent_uuid") or parent, parsed={"rows": rows, "data_lines": 1, "malformed": 0, "not_available": 0, "impossible": 0 if possible else 1},
                    parent_total_mib=probe.get("parent_total_mib"))


# ---- the per-run record and how it is used -------------------------------------------------------------------------------------------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query(run, arguments):
    try:
        done = run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else None


def build_record(meta, csv_path, *, run=subprocess.run, nvml=None, torch_total_mib=None, env=None):
    """The gpu-scope.json of ONE run, from its run-meta and its sampler file."""
    env = os.environ if env is None else env
    allocated, parent = meta.get("gpu_uuid"), meta.get("parent_gpu_uuid")
    text = Path(csv_path).read_text(encoding="utf-8", errors="replace") if Path(csv_path).is_file() else ""
    parsed = parse_sampler(text)
    parent_total = None
    if uuid_kind(allocated) == "mig" and parent:
        found = query(run, ["-i", parent, "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        parent_total = int(found.splitlines()[0]) if found and found.splitlines()[0].strip().isdigit() else None
    verdict = evaluate(allocated=allocated, parent=parent, parsed=parsed, parent_total_mib=parent_total, torch_total_mib=torch_total_mib)
    probe = nvml_probe(allocated, nvml) if uuid_kind(allocated) else {"available": False, "error": "no allocated UUID", "returned_uuid": None, "memory_total_mib": None, "memory_used_mib": None,
                                                                    "is_mig_handle": None, "parent_uuid": None, "parent_total_mib": None}
    return {"schema": "gpu-scope-v1", "allocated_uuid": allocated, "parent_uuid": parent, "method": CSV_METHOD, "scope": verdict["scope"], "reason": verdict["reason"], "device_kind": verdict["device_kind"],
            "returned_uuids": verdict["returned_uuids"], "memory_total_mib": verdict["memory_total_mib"], "parent_total_mib": parent_total, "torch_total_mib": torch_total_mib,
            "sampler_csv_sha256": sha256_file(csv_path) if Path(csv_path).is_file() else None,
            "sampler_format": {"valid_rows": len(parsed["rows"]), "data_lines": parsed["data_lines"], "malformed": parsed["malformed"], "not_available": parsed["not_available"]},
            "nvml": {"method": NVML_METHOD, "probe": probe, "verdict": evaluate_nvml(allocated, probe, parent)},
            "driver": query(run, ["--query-gpu=driver_version", "--format=csv,noheader"]), "node": platform.node(), "slurm_job_id": env.get("SLURM_JOB_ID") or meta.get("slurm_job_id"),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def record_problems(record):
    """Why a gpu-scope.json is not a well-formed record ([] when it is). Types only: whether it is TRUE is decided by re-deriving it (memory_report)."""
    if not isinstance(record, dict):
        return [f"the record is a {type(record).__name__}, not an object"]
    problems = []
    for key in ("allocated_uuid", "parent_uuid", "method", "sampler_csv_sha256", "reason"):
        if key not in record or not (record[key] is None or isinstance(record[key], str)):
            problems.append(f"{key} is missing or not a string/null")
    for key in ("method", "sampler_csv_sha256"):
        if key in record and record[key] is None:
            problems.append(f"{key} is null")
    if not isinstance(record.get("scope"), str) or record.get("scope") not in ("slice_verified", "device_verified", "scope_unverified", "unavailable"):
        problems.append("scope is missing or not one of the vocabulary")
    for key in ("parent_total_mib", "torch_total_mib", "memory_total_mib"):
        value = record.get(key)
        if (key not in record and key != "torch_total_mib") or not (value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)):
            problems.append(f"{key} is missing or not a non-negative integer/null")
    if not isinstance(record.get("returned_uuids"), list) or not all(isinstance(item, str) for item in record.get("returned_uuids", [])):
        problems.append("returned_uuids is missing or not a list of strings")
    return problems


def memory_report(csv_path):
    """The ONLY way a report may state GPU memory: {"peak_mib", "scope", "reason", "method", "device_uuid", "label", "device_reported_peak_mib"}.
    peak_mib is a number only for a verified scope whose record is bound to THIS run (same allocated device as run-meta, same sampler file bytes, same method); otherwise None."""
    folder = Path(csv_path).parent if csv_path else None
    raw = raw_peak_mib(csv_path)
    out = {"peak_mib": None, "scope": "unavailable", "reason": None, "method": None, "device_uuid": None, "label": UNVERIFIED_LABEL if raw is not None else "no GPU measurement", "device_reported_peak_mib": raw}
    if not csv_path or not Path(csv_path).is_file():
        out["reason"] = "no sampler file"
        return out
    record_path, meta_path = folder / SCOPE_FILE, folder / "run-meta.json"
    if not record_path.is_file():
        out["reason"] = "no scope record (a run made before REV 4.1 has none)"
        return out
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    except (OSError, ValueError) as error:
        out["reason"] = f"the scope record cannot be read ({error})"
        return out
    shape = record_problems(record)
    if shape:                                                                                  # REV 4.2: a malformed record is unavailable evidence, never an exception
        out["reason"] = "scope record malformed: " + "; ".join(shape)
        return out
    out.update(method=record["method"], device_uuid=record["allocated_uuid"])
    problems = []
    allocated = meta.get("gpu_uuid") if isinstance(meta, dict) else None
    parent = meta.get("parent_gpu_uuid") if isinstance(meta, dict) else None
    if not isinstance(allocated, str) or not allocated:
        problems.append("run-meta records no allocated device")
    elif record["allocated_uuid"] != allocated:
        problems.append(f"the record is for device {record['allocated_uuid']!r}, this run was allocated {allocated!r}")
    if parent is not None and not isinstance(parent, str):
        problems.append("run-meta's parent device is not a string")
    if record["sampler_csv_sha256"] != sha256_file(csv_path):
        problems.append("the sampler file changed after the scope record was made")
    if record["method"] != CSV_METHOD:
        problems.append(f"the record is for method {record['method']!r}, the sampler used {CSV_METHOD!r}")
    if problems:
        out["reason"] = "scope record not bound to this run: " + "; ".join(problems)
        return out
    # REV 4.2: the label is never copied. The scope is DERIVED again from the bound sampler bytes, the allocated identity and the capacities the record holds, and the record must agree with it.
    verdict = evaluate(allocated=allocated, parent=parent, parsed=parse_sampler(Path(csv_path).read_text(encoding="utf-8", errors="replace")), parent_total_mib=record["parent_total_mib"],
                       torch_total_mib=record.get("torch_total_mib"))
    contradictions = [f"{name}: record {record[key]!r}, derived {verdict[key]!r}" for name, key in (("scope", "scope"), ("reason", "reason"), ("returned device uuids", "returned_uuids"), ("memory total", "memory_total_mib"))
                      if record[key] != verdict[key]]
    if contradictions:
        out.update(scope="scope_unverified", reason="record_contradicts_sampler (" + "; ".join(contradictions) + ")")
        return out
    out.update(scope=verdict["scope"], reason=verdict["reason"])
    if verdict["scope"] in VERIFIED:
        out.update(peak_mib=raw, label=VERIFIED_LABEL[verdict["scope"]])
    return out


def smoke_scope(*, allocated, parent, one_shot, torch_total_mib, torch_used_mib, parent_total_mib=None):
    """The smoke's cross-check (device_smoke.post_facts): identity and capacity from a one-shot nvidia-smi sample, then the numbers against torch.cuda.mem_get_info. Reported, never gating."""
    parsed = parse_sampler(one_shot or "")
    numbers = {"measured_used_mib": parsed["rows"][0]["used_mib"], "torch_used_mib": torch_used_mib} if parsed["rows"] and torch_used_mib is not None else None
    verdict = evaluate(allocated=allocated, parent=parent, parsed=parsed, parent_total_mib=parent_total_mib, torch_total_mib=torch_total_mib, numbers=numbers)
    return {"gpu_memory_scope": verdict["scope"], "reason": verdict["reason"], "returned_uuids": verdict["returned_uuids"], "memory_total_mib": verdict["memory_total_mib"], "parent_total_mib": parent_total_mib}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-meta", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--torch-total-mib", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        meta = json.loads(Path(args.run_meta).read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"run-meta is a {type(meta).__name__}, not an object")
    except (OSError, ValueError) as error:
        print(f"gpu_scope: run-meta unreadable ({error}); no record written", file=sys.stderr)
        return 0                                                                              # a measurement note never fails the run
    record = build_record(meta, args.csv, torch_total_mib=args.torch_total_mib)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"scope": record["scope"], "reason": record["reason"], "nvml_scope": record["nvml"]["verdict"]["scope"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
