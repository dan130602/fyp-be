"""The verdict on a measured EC2 run (REV 3.0 B5, REV 3.2 Amendment 8, REV 3.3 Amendment 9, REV 3.4). Standard library only; pure functions, so the verdict is unit-tested without Docker or AWS.

    python resource_summary.py declaration --out ec2-declaration.json [--primary-mib 24576] [--downsize-mib 12288]
    python resource_summary.py declaration-env <ec2-declaration.json> <primary|downsize>        (KEY=VALUE lines for measure_resources.sh)
    python resource_summary.py summarize --declaration D --limit <primary|downsize> --run NAME=DIR [--run NAME=DIR ...] --out summary.json

A run directory (written by measure_resources.sh, ONE fresh container per measured run) holds
  samples.tsv      ts_ms mem_current anon file shmem usage_usec nr_throttled nr_procs swap_current        (every sample interval; NA when a value could not be read)
  procs.tsv        ts_ms pid ppid rss_kb comm                                                                   (the processes of the container's cgroup, so compile children are included)
  final/           memory.peak memory.events memory.pressure memory.swap.events docker-inspect.json status.env   (read while the container was still ALIVE)
  completion.json  exit_code documents_expected documents_written all_end_token failed_markers               (what the workload produced; written by the caller; every key is REQUIRED, nothing is defaulted)
  limits/          start/{memory.max,memory.swap.max,cpu.max,cpuset.cpus.effective} and docker-inspect-start.json   (the limits ACTUALLY enforced, read before the workload; the final read repeats them in final/)
  context.json     optional: the measured context an image-contents run is valid FOR (image digest, profile, checkpoint, limits, GPU, driver, instance, workload), written by the caller

REV 4.1: a sample counts toward the anonymous total only when BOTH anon and shmem were read (never 0), the maximum is compared in BYTES, memory.events and the completion record are validated rather than
defaulted, and the declared limits (memory, swap = 0, cpu quota, cpuset) must be SHOWN enforced by the cgroup files and docker's HostConfig (a readable zero swap USAGE is not that). A flagged run
(PASS_FLAGGED) never becomes `validated` by itself: judge_t2 needs a review disposition bound to that very record, and the claim is only for the measured context.

What the numbers mean (never one of them alone): `sampled_max_anon_shmem_mib` is a SAMPLED maximum, a lower bound of the true anonymous peak (a burst shorter than the sampling gap can fall between two
samples). memory.peak is kept by the kernel without a gap but counts everything charged to the cgroup including reclaimable page cache, so it is an upper bound. The true anonymous peak lies between them.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MIB = 1024 * 1024
BUILD_PROCESSES = ("ninja", "nvcc", "cc1plus", "cc1", "cicc", "ptxas", "fatbinary", "gcc", "g++", "c++")
BRACKET = "sampled maximum; the true anonymous peak lies between this value and memory.peak"
SAMPLE_COLUMNS = ("ts_ms", "mem_current", "anon", "file", "shmem", "usage_usec", "nr_throttled", "nr_procs", "swap_current")
VERDICTS = ("PASS", "PASS_FLAGGED", "FAIL", "INCONSISTENT")
LIMIT_FILES = ("memory.max", "memory.swap.max", "cpu.max", "cpuset.cpus.effective")
CPU_PERIOD = 100000
COMPLETION_INTEGERS = ("exit_code", "documents_expected", "documents_written", "failed_markers")


# ---- the declaration: fixed BEFORE the first run, hashed into every run record ---------------------------------------------------------------------------------------------------------------

def make_declaration(primary_mib=24576, downsize_mib=12288):
    return {"schema": "ec2-declaration-v1", "memory_limit_mib_primary": primary_mib, "memory_limit_mib_downsize": downsize_mib,
            "memory_swap_policy": "memory-swap equals the memory limit: no swap", "cpus": 4, "cpuset": "0-3", "build_and_thread_controls": {"MAX_JOBS": 4, "OMP_NUM_THREADS": 4, "MKL_NUM_THREADS": 4,
            "OPENBLAS_NUM_THREADS": 4, "TORCHINDUCTOR_COMPILE_THREADS": 4}, "anon_fraction_gate": 0.80, "host_margin_mib": 1024, "sample_interval_s": 2, "downsize_screening_fraction": 0.80,
            "cache_policy": "one fresh container per measured run; the cold run uses EMPTY cache volumes (FLASHINFER_WORKSPACE_BASE, VLLM_CACHE_ROOT, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, XDG_CACHE_HOME), the warm run the populated ones",
            "terminate_grace_seconds": 20, "workload_timeout_seconds": 21600}


def declaration_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limit_of(declaration, key):
    if key not in ("primary", "downsize"):
        raise ValueError(f"limit key must be primary or downsize, not {key!r}")
    return int(declaration[f"memory_limit_mib_{key}"])


def headroom_ok(available_mib, limit_mib, margin_mib):
    """A PRECONDITION, not a redefinition of the limit: the host must have the declared limit plus the margin available before the container starts."""
    return available_mib is not None and available_mib >= limit_mib + margin_mib


def downsize_screening(overall_sampled_max_mib, declaration):
    """The launch rule for the optional g4dn.xlarge test: a SCREENING rule on a LOWER bound. It avoids paying for an instance the measurements already exclude; it never establishes that the instance fits."""
    ceiling = declaration["downsize_screening_fraction"] * declaration["memory_limit_mib_downsize"]
    return {"launch_allowed": overall_sampled_max_mib is not None and overall_sampled_max_mib <= ceiling, "ceiling_mib": round(ceiling, 1), "overall_sampled_max_anon_shmem_mib": overall_sampled_max_mib,
            "nature": "screening on a sampled lower bound; the actual g4dn.xlarge run, at the declared downsize limit, is the evidence"}


# ---- parsing ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _number(token):
    return None if token in ("NA", "", None) else int(token)


def parse_samples(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != len(SAMPLE_COLUMNS):
            continue
        rows.append({name: _number(token) for name, token in zip(SAMPLE_COLUMNS, parts)})
    return rows


def parse_procs(text):
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[0].isdigit():
            rows.append({"ts_ms": int(parts[0]), "pid": int(parts[1]), "ppid": int(parts[2]), "rss_kb": int(parts[3]), "comm": parts[4].strip()})
    return rows


def parse_counters(text):
    """memory.events, memory.swap.events, cpu.stat: `key value` per line."""
    counters = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"-?\d+", parts[1]):
            counters[parts[0]] = int(parts[1])
    return counters


def parse_pressure(text):
    """memory.pressure: `some avg10=0.00 avg60=0.00 avg300=0.00 total=0` and `full ...`."""
    result = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in ("some", "full"):
            values = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
            result[parts[0]] = {"avg10": float(values.get("avg10", 0)), "total": int(values.get("total", 0))}
    return result or None


def parse_status(text):
    status = {}
    for line in (text or "").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            status[key.strip()] = value.strip()
    return status


def host_config(text):
    """The enforced limits Docker reports (HostConfig), or None when the inspect output is missing or unreadable."""
    try:
        data = json.loads(text) if text else None
    except ValueError:
        return None
    data = data[0] if isinstance(data, list) and data else data
    config = data.get("HostConfig") if isinstance(data, dict) else None
    return {key: config.get(key) for key in ("Memory", "MemorySwap", "NanoCpus", "CpusetCpus")} if isinstance(config, dict) else None


def load_run(directory):
    directory = Path(directory)

    def read(name):
        path = directory / name
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    peak_text = read("final/memory.peak")
    inspect_text = read("final/docker-inspect.json")
    try:
        docker = json.loads(inspect_text) if inspect_text else None
        docker = docker[0] if isinstance(docker, list) and docker else docker
    except ValueError:
        docker = None
    malformed_files = []

    def read_json(name):
        """The parsed JSON of a record file, or None when it is absent OR not valid JSON (the latter is listed in malformed_files and judged INCONSISTENT; load_run itself never raises on file content)."""
        text = read(name)
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            malformed_files.append(name)
            return None
    completion, context = read_json("completion.json"), read_json("context.json")
    last = {name: read(f"last_sampled/{name}") for name in ("memory.peak", "memory.events", "memory.pressure")}
    swap_events_text = read("final/memory.swap.events")
    limits = {"docker_start": host_config(read("limits/docker-inspect-start.json")), "docker_final": host_config(inspect_text),
              "cgroup_start": {name: ((read(f"limits/start/{name}") or "").strip() or None) for name in LIMIT_FILES},
              "cgroup_final": {name: ((read(f"final/{name}") or "").strip() or None) for name in LIMIT_FILES}}
    last_sampled = {"memory_peak": int(last["memory.peak"].strip()) if last["memory.peak"] and last["memory.peak"].strip().isdigit() else None,
                    "memory_events": parse_counters(last["memory.events"]) if last["memory.events"] else None, "memory_pressure": parse_pressure(last["memory.pressure"])}
    return {"last_sampled": last_sampled, "samples": parse_samples(read("samples.tsv") or ""), "procs": parse_procs(read("procs.tsv") or ""), "status": parse_status(read("final/status.env")),
            "memory_peak": int(peak_text.strip()) if peak_text and peak_text.strip().isdigit() else None, "memory_events": parse_counters(read("final/memory.events")) if read("final/memory.events") else None,
            "memory_pressure": parse_pressure(read("final/memory.pressure")), "memory_swap_events": parse_counters(swap_events_text) if swap_events_text is not None else None,
            "docker": (docker or {}).get("State") if isinstance(docker, dict) else None, "completion": completion, "limits": limits,
            "context": context, "run_id": directory.name, "malformed_files": malformed_files}


# ---- statistics --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sample_stats(samples, procs, nominal_interval_s):
    if not samples:
        return {"samples": 0, "samples_complete": 0, "samples_incomplete": 0, "complete_first_ts_ms": None, "complete_last_ts_ms": None, "max_gap_between_complete_s": None, "sampled_max_anon_shmem_bytes": None,
                "sampled_max_memory_current_bytes": None, "nominal_interval_s": nominal_interval_s, "max_gap_s": None, "sampled_max_anon_shmem_mib": None, "sampled_max_memory_current_mib": None, "average_cores": None,
                "throttled_periods": None, "swap_current_max_bytes": None, "largest_processes_at_peak": [], "peak_concurrent_build_processes": 0}
    stamps = [row["ts_ms"] for row in samples]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    complete = [row for row in samples if row["anon"] is not None and row["shmem"] is not None]       # REV 4.1 B1: an unreadable component is never treated as 0
    complete_stamps = [row["ts_ms"] for row in complete]
    complete_gaps = [b - a for a, b in zip(complete_stamps, complete_stamps[1:])]
    anon_shmem = [row["anon"] + row["shmem"] for row in complete]
    current = [row["mem_current"] for row in samples if row["mem_current"] is not None]
    usage = [row["usage_usec"] for row in samples if row["usage_usec"] is not None]
    throttled = [row["nr_throttled"] for row in samples if row["nr_throttled"] is not None]
    swap = [row["swap_current"] for row in samples if row["swap_current"] is not None]
    wall = (stamps[-1] - stamps[0]) / 1000 if len(stamps) > 1 else 0
    peak_ts = max((row for row in samples if row["mem_current"] is not None), key=lambda row: row["mem_current"], default=None)
    largest = []
    if peak_ts is not None and procs:
        nearest = min({p["ts_ms"] for p in procs}, key=lambda t: abs(t - peak_ts["ts_ms"]))
        largest = sorted((p for p in procs if p["ts_ms"] == nearest), key=lambda p: -p["rss_kb"])[:5]
        largest = [{"pid": p["pid"], "comm": p["comm"], "rss_mib": round(p["rss_kb"] / 1024, 1)} for p in largest]
    concurrent = {}
    for p in procs:
        if p["comm"].split("/")[-1] in BUILD_PROCESSES:
            concurrent[p["ts_ms"]] = concurrent.get(p["ts_ms"], 0) + 1
    return {"samples": len(samples), "samples_complete": len(complete), "samples_incomplete": len(samples) - len(complete), "complete_first_ts_ms": complete_stamps[0] if complete_stamps else None,
            "complete_last_ts_ms": complete_stamps[-1] if complete_stamps else None, "max_gap_between_complete_s": round(max(complete_gaps) / 1000, 3) if complete_gaps else (0.0 if complete_stamps else None),
            "nominal_interval_s": nominal_interval_s, "max_gap_s": round(max(gaps) / 1000, 3) if gaps else 0.0,
            "sampled_max_anon_shmem_bytes": max(anon_shmem) if anon_shmem else None, "sampled_max_memory_current_bytes": max(current) if current else None,
            "sampled_max_anon_shmem_mib": round(max(anon_shmem) / MIB, 1) if anon_shmem else None, "sampled_max_memory_current_mib": round(max(current) / MIB, 1) if current else None,
            "average_cores": round((usage[-1] - usage[0]) / 1e6 / wall, 2) if len(usage) > 1 and wall > 0 else None, "throttled_periods": (throttled[-1] - throttled[0]) if len(throttled) > 1 else None,
            "swap_current_max_bytes": max(swap) if swap else None, "largest_processes_at_peak": largest, "peak_concurrent_build_processes": max(concurrent.values(), default=0)}


def is_count(value):
    """A SIGNED integer (never a bool): a process exit status may legitimately be negative (a signal)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_nonneg_count(value):
    """A count of things (documents, markers, events): an integer that cannot be negative. A negative number is malformed evidence, never a quiet zero."""
    return is_count(value) and value >= 0


NONNEGATIVE_COMPLETION = ("documents_written", "failed_markers")


def completion_problems(completion):
    """Why a completion record is not a complete record of what the workload produced ([] when it is). Nothing is defaulted: a missing key is a problem, never `None == None`.
    exit_code is a signed integer (any non-zero value, negative signal statuses included, is a FAILURE judged elsewhere); the document and marker counts cannot be negative."""
    if completion is None:
        return ["no completion record"]
    if not isinstance(completion, dict):
        return [f"the completion record is a {type(completion).__name__}, not an object"]
    problems = [f"{key} is missing or not an integer" for key in COMPLETION_INTEGERS if not is_count(completion.get(key))]
    problems += [f"{key} is negative" for key in NONNEGATIVE_COMPLETION if is_count(completion.get(key)) and completion[key] < 0]
    if is_count(completion.get("documents_expected")) and completion["documents_expected"] < 1:
        problems.append("documents_expected is below 1")
    if not isinstance(completion.get("all_end_token"), bool):
        problems.append("all_end_token is missing or not a boolean")
    return problems


def limits_problems(limits, declaration, limit_mib):
    """REV 4.1 B3 and the ONE swap rule: the DECLARED limits must be shown ENFORCED, by the HostConfig docker reports (read before the workload and again in the final read) and by the cgroup files at the start
    and in the final read: memory.max equals the limit, memory.swap.max equals 0 (a readable zero swap USAGE proves nothing about what was prohibited), cpu.max is the quota, the cpuset is the declared one."""
    if not isinstance(limits, dict):
        return ["no limits record: the declared limits are not shown to be enforced"]
    memory = limit_mib * MIB
    expected_cgroup = {"memory.max": str(memory), "memory.swap.max": "0", "cpu.max": f"{int(declaration['cpus']) * CPU_PERIOD} {CPU_PERIOD}", "cpuset.cpus.effective": str(declaration["cpuset"])}
    expected_docker = {"Memory": memory, "MemorySwap": memory, "NanoCpus": int(declaration["cpus"]) * 10 ** 9, "CpusetCpus": str(declaration["cpuset"])}
    problems = []
    for phase in ("start", "final"):
        found = limits.get(f"cgroup_{phase}") or {}
        for name, want in expected_cgroup.items():
            got = found.get(name)
            if got is None:
                problems.append(f"{name} was not read at the {phase}" + (" (the no-swap limit is not shown to be enforced)" if name == "memory.swap.max" else ""))
            elif got != want:
                problems.append(f"{name} is {got!r} at the {phase}, declared {want!r}" + (" (swap is not prohibited)" if name == "memory.swap.max" else ""))
        docker = limits.get(f"docker_{phase}")
        if not isinstance(docker, dict):
            problems.append(f"the container's HostConfig was not read at the {phase}")
            continue
        for name, want in expected_docker.items():
            if docker.get(name) != want:
                problems.append(f"HostConfig {name} is {docker.get(name)!r} at the {phase}, declared {want!r}" + (" (MemorySwap must equal Memory: no swap)" if name == "MemorySwap" else ""))
    return problems


def judge_run(run, declaration, limit_key, *, expect_declaration_sha256=None):
    """The verdict of ONE T1 gating run: PASS | PASS_FLAGGED | FAIL | INCONSISTENT, with every reason. FAIL evidence wins (a missing final metric never turns a failed run into a pass); INCONSISTENT means
    the evidence is not usable (nothing is decided); PASS_FLAGGED never auto-passes: the numbers are reported and judged."""
    limit_mib = limit_of(declaration, limit_key)
    stats = sample_stats(run["samples"], run["procs"], declaration["sample_interval_s"])
    status = run["status"] if isinstance(run.get("status"), dict) else {}                  # REV 4.2: every record is an object or it is ignored (and reported); nothing below calls .get on another type
    pressure, completion = run["memory_pressure"], run["completion"]
    docker_state = run["docker"] if isinstance(run.get("docker"), dict) else {}
    events = run["memory_events"] if isinstance(run["memory_events"], dict) else {}
    fail, inconsistent, flags = [], [], []
    if run.get("docker") is not None and not isinstance(run["docker"], dict):
        inconsistent.append("docker's State record is present but not an object, so whether the container was OOM-killed is not established")
    for name in run.get("malformed_files") or []:
        inconsistent.append(f"{name} is not valid JSON, so what it records is not established")
    absent_counters = [key for key in ("oom", "oom_kill", "max") if not is_nonneg_count(events.get(key))]
    if absent_counters:
        inconsistent.append(f"memory.events is missing, or {absent_counters} is absent or not a non-negative integer, so OOM and limit events are not established")
    counter = lambda key: events[key] if is_nonneg_count(events.get(key)) else 0           # a counter that is absent, negative or not an integer is already INCONSISTENT above; it is never compared as a non-number
    if counter("oom") > 0 or counter("oom_kill") > 0:
        fail.append(f"memory.events oom={counter('oom')} oom_kill={counter('oom_kill')}")
    if docker_state.get("OOMKilled") is True:
        fail.append("docker reports the container was OOM-killed")
    if (stats["swap_current_max_bytes"] or 0) > 0:
        fail.append("swap was used (memory.swap.current)")
    swap_events = run.get("memory_swap_events")
    if swap_events is not None:                                                             # an ABSENT file is optional (recorded elsewhere); a PRESENT one must be readable
        if not isinstance(swap_events, dict):
            inconsistent.append("memory.swap.events is present but not readable, so swap events are not established")
        else:
            unreadable = [key for key in ("max", "fail") if not is_nonneg_count(swap_events.get(key))]
            if unreadable:
                inconsistent.append(f"memory.swap.events is present but {unreadable} is absent or not a non-negative integer, so swap events are not established")
            positive = {key: swap_events[key] for key in ("max", "fail") if is_nonneg_count(swap_events.get(key)) and swap_events[key] > 0}
            if positive:                                                                     # REV 4.1: with memory.swap.max == 0 any positive count means the kernel tried to swap this container's pages and was refused;
                fail.append(f"swap was refused or used: memory.swap.events {positive} (the kernel tried to swap this container's pages)")        # an established failure keeps FAIL even when the other counter is malformed
    if status.get("workload_exit") not in (None, "0"):
        fail.append(f"the workload exited with {status.get('workload_exit')}")
    if completion is None:
        inconsistent.append("no completion record: whether the run completed is not established")
    else:
        problems = completion_problems(completion)
        if problems:                                                                         # REV 4.1 B1: required keys are validated, never defaulted
            inconsistent.append("the completion record is incomplete or malformed: " + "; ".join(problems))
        if isinstance(completion, dict):                                                     # REV 4.2: a non-object record has been reported above and is not read any further
            if is_count(completion.get("exit_code")) and completion["exit_code"] != 0:
                fail.append(f"the worker exited with {completion['exit_code']}")
            if is_nonneg_count(completion.get("documents_written")) and is_nonneg_count(completion.get("documents_expected")) and completion["documents_written"] != completion["documents_expected"]:
                fail.append(f"{completion['documents_written']} documents written, {completion['documents_expected']} expected")
            if (is_nonneg_count(completion.get("failed_markers")) and completion["failed_markers"] > 0) or completion.get("all_end_token") is False:
                fail.append("the run has failed markers or a decode that did not end with an end token")
    gate = declaration["anon_fraction_gate"] * limit_mib
    if stats["sampled_max_anon_shmem_bytes"] is not None and stats["sampled_max_anon_shmem_bytes"] > gate * MIB:                                  # bytes, not rounded MiB (REV 4.1 B2)
        fail.append(f"sampled anonymous+shared memory {stats['sampled_max_anon_shmem_mib']} MiB is above {declaration['anon_fraction_gate']:.0%} of the declared limit ({gate:.0f} MiB)")
    if not run["samples"]:
        inconsistent.append("no samples")
    elif stats["samples_complete"] == 0:
        inconsistent.append(f"none of the {stats['samples']} samples has both anon and shmem readable: the anonymous total is not established")
    elif stats["samples_incomplete"] > 0:
        flags.append(f"{stats['samples_incomplete']} of {stats['samples']} samples lack anon or shmem (not counted as 0); the sampled maximum covers {stats['samples_complete']} complete samples between "
                     f"t={stats['complete_first_ts_ms']} and t={stats['complete_last_ts_ms']} ms, largest gap between complete samples {stats['max_gap_between_complete_s']} s")
    swap_problems = limits_problems(run.get("limits"), declaration, limit_mib)
    if swap_problems:
        inconsistent.append("declared limits (memory, swap, cpu, cpuset) are not shown to be enforced: " + "; ".join(swap_problems))
    if status.get("finality") != "final_confirmed":
        inconsistent.append(f"final metrics are not final_confirmed (finality {status.get('finality')!r}: {status.get('reason', 'no reason recorded')}); final_metrics_missing")
    if run["memory_peak"] is None:
        inconsistent.append("memory.peak is missing")
    elif stats["sampled_max_anon_shmem_bytes"] is not None and stats["sampled_max_anon_shmem_bytes"] > run["memory_peak"]:
        inconsistent.append(f"the sampled anonymous+shared maximum ({stats['sampled_max_anon_shmem_bytes']} bytes) exceeds memory.peak ({run['memory_peak']} bytes): another cgroup, a reset or a restart")
    if status.get("fresh_container") != "1":
        inconsistent.append("the run was not in its own fresh container (memory.peak is the high-water mark since the cgroup was created)")
    if expect_declaration_sha256 is not None and status.get("declaration_sha256") != expect_declaration_sha256:
        inconsistent.append("the declaration hash recorded with the run is not that of the declaration given (a declared value was changed between runs)")
    if status.get("limit_mib") not in (None, str(limit_mib)):
        inconsistent.append(f"the run recorded limit {status.get('limit_mib')} MiB, the declaration says {limit_mib}")
    if counter("max") > 0:
        flags.append(f"memory.events max={counter('max')}: the cgroup hit its limit and reclaimed (page-cache reclaim and real pressure cannot be told apart here)")
    if pressure and pressure.get("full", {}).get("total", 0) > 0:
        flags.append(f"memory.pressure full stall total {pressure['full']['total']} us")
    verdict = "FAIL" if fail else "INCONSISTENT" if inconsistent else "PASS_FLAGGED" if flags else "PASS"
    result = {"verdict": verdict, "fail_reasons": fail, "inconsistent_reasons": inconsistent, "flags": flags, "limit_mib": limit_mib, "stats": stats,
            "last_sampled_substitute": {"memory_peak_mib": None if (run.get("last_sampled") or {}).get("memory_peak") is None else round(run["last_sampled"]["memory_peak"] / MIB, 1),
                                        "memory_events": (run.get("last_sampled") or {}).get("memory_events"), "label": "last sampled (at most one interval old); never final, never a pass"},
            "kernel": {"memory_peak_mib": None if run["memory_peak"] is None else round(run["memory_peak"] / MIB, 1), "memory_events": events or None, "memory_pressure": pressure,
                       "swap_events": run["memory_swap_events"]}, "finality": status.get("finality"), "completion": completion, "bracket": BRACKET,
            "run_id": run.get("run_id"), "context": run.get("context"), "limits_enforced": not swap_problems}
    result["record_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode("utf-8")).hexdigest()           # what a review disposition is bound to
    return result


REQUIRED_CONTEXT = ("image_digest", "settings_id", "frozen_digest", "config_fingerprint", "checkpoint_manifest_sha256", "limits", "gpu_model", "driver", "instance_type", "workload")


def context_differences(measured, production):
    """REV 4.1 B5: where the measured context and the production context differ ([] when they agree in every required entry). An entry missing on either side is a difference, never a match."""
    measured, production = measured if isinstance(measured, dict) else {}, production if isinstance(production, dict) else {}
    differences = []
    for key in REQUIRED_CONTEXT:
        if measured.get(key) is None or production.get(key) is None:
            differences.append(f"{key} is not recorded on the {'measured' if measured.get(key) is None else 'production'} side")
        elif measured[key] != production[key]:
            differences.append(f"{key}: measured {measured[key]!r}, production {production[key]!r}")
    return differences


def disposition_accepts(disposition, image_run):
    """A review disposition counts only when it accepts THIS run's record: same run id, same record hash, accepted true. A disposition for another run or a changed record is ignored."""
    return (isinstance(disposition, dict) and disposition.get("accepted") is True and bool(image_run.get("run_id")) and disposition.get("run_id") == image_run.get("run_id")
            and bool(image_run.get("record_sha256")) and disposition.get("resource_record_sha256") == image_run.get("record_sha256"))


def judge_t2(completion, status, image_run=None, disposition=None, production_context=None):
    """T2 (the real image entrypoint). FUNCTIONAL and RESOURCE evidence are separate; missing final metrics never make a functionally successful run INCONSISTENT.
    `functional` is pass only for a COMPLETE, good completion record (never from a bare exit code). `production_image_memory` (REV 4.1 B5) is never taken from a status field:
      not_validated_by_T2             no image-contents run, or it FAILED or is INCONSISTENT;
      context_differs                 the run is usable but its measured context (image digest, profile, checkpoint, enforced limits, GPU and driver, instance, workload) is not the production one;
      pending_review                  the run is PASS_FLAGGED and no matching review disposition exists;
      validated_for_measured_context  PASS, or PASS_FLAGGED with a disposition that accepts this very run record. It says the image fits THAT context, nothing more."""
    problems = completion_problems(completion)
    exit_code_bad = isinstance(completion, dict) and is_count(completion.get("exit_code")) and completion["exit_code"] != 0
    good = not problems and not exit_code_bad and completion["documents_written"] == completion["documents_expected"] and completion["failed_markers"] == 0 and completion["all_end_token"] is True
    functional = "pass" if good else "fail" if exit_code_bad or not problems else "not_established"
    finality = (status.get("finality") if isinstance(status, dict) else None) or "unavailable"
    verdict = (image_run or {}).get("verdict")
    if verdict not in ("PASS", "PASS_FLAGGED"):
        memory, notes = "not_validated_by_T2", ([] if image_run is None else [f"the image-contents run is {verdict}"])
    else:
        notes = context_differences(image_run.get("context"), production_context)
        if notes:
            memory = "context_differs"
        elif verdict == "PASS_FLAGGED" and not disposition_accepts(disposition, image_run):
            memory, notes = "pending_review", list(image_run.get("flags") or [])
        else:
            memory = "validated_for_measured_context"
    return {"functional": functional, "resource_evidence": finality if finality in ("final_confirmed", "last_sampled") else "unavailable", "production_image_memory": memory,
            "production_image_memory_notes": notes, "validated_for": image_run.get("context") if memory == "validated_for_measured_context" else None}


def summarize_runs(runs, declaration, limit_key, declaration_hash=None):
    judged = {name: judge_run(run, declaration, limit_key, expect_declaration_sha256=declaration_hash) for name, run in runs.items()}
    sampled = [j["stats"]["sampled_max_anon_shmem_mib"] for j in judged.values() if j["stats"]["sampled_max_anon_shmem_mib"] is not None]
    overall_max = max(sampled) if sampled else None
    verdicts = {j["verdict"] for j in judged.values()}
    overall = "FAIL" if "FAIL" in verdicts else "INCONSISTENT" if "INCONSISTENT" in verdicts or not judged else "PASS_FLAGGED" if "PASS_FLAGGED" in verdicts else "PASS"
    peaks = [j["kernel"]["memory_peak_mib"] for j in judged.values() if j["kernel"]["memory_peak_mib"] is not None]
    return {"limit": limit_key, "limit_mib": limit_of(declaration, limit_key), "runs": judged, "overall": {"verdict": overall, "sampled_max_anon_shmem_mib_over_all_runs": overall_max,
            "memory_peak_mib_over_all_runs": max(peaks) if peaks else None, "which_run_is_largest": "not assumed: the maximum is taken over every measured run (cold, warm and the others)", "bracket": BRACKET},
            "downsize_screening": downsize_screening(overall_max, declaration) if limit_key == "primary" else None}


# ---- command line ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("declaration")
    make.add_argument("--out", required=True)
    make.add_argument("--primary-mib", type=int, default=24576)
    make.add_argument("--downsize-mib", type=int, default=12288)
    env = commands.add_parser("declaration-env")
    env.add_argument("declaration")
    env.add_argument("limit", choices=("primary", "downsize"))
    summary = commands.add_parser("summarize")
    summary.add_argument("--declaration", required=True)
    summary.add_argument("--limit", required=True, choices=("primary", "downsize"))
    summary.add_argument("--run", action="append", required=True, help="NAME=DIRECTORY")
    summary.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "declaration":
        Path(args.out).write_text(json.dumps(make_declaration(args.primary_mib, args.downsize_mib), indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"declaration written; sha256 {declaration_sha256(args.out)}")
        return 0
    declaration = json.loads(Path(args.declaration).read_text(encoding="utf-8"))
    if args.command == "declaration-env":
        controls = declaration["build_and_thread_controls"]
        values = {"LIMIT_MIB": limit_of(declaration, args.limit), "SWAP_MIB": limit_of(declaration, args.limit), "CPUS": declaration["cpus"], "CPUSET": declaration["cpuset"], "SAMPLE_INTERVAL_S": declaration["sample_interval_s"],
                  "HOST_MARGIN_MIB": declaration["host_margin_mib"], "GRACE_S": declaration["terminate_grace_seconds"], "TIMEOUT_S": declaration["workload_timeout_seconds"],
                  "DECLARATION_SHA256": declaration_sha256(args.declaration), **controls}
        for key, value in values.items():
            print(f"{key}={value}")
        return 0
    runs = {}
    for item in args.run:
        name, _, directory = item.partition("=")
        runs[name] = load_run(directory)
    result = summarize_runs(runs, declaration, args.limit, declaration_sha256(args.declaration))
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"overall": result["overall"], "per_run": {n: r["verdict"] for n, r in result["runs"].items()}}, indent=1))
    return 0 if result["overall"]["verdict"] in ("PASS", "PASS_FLAGGED") else 1


if __name__ == "__main__":
    sys.exit(main())
