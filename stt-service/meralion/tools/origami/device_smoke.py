"""A profile smoke test on the ALLOCATED device (REV 3.0 A1; run on Origami inside the vLLM environment; never part of the image). It replaces vllm_smoke.py for profile validation.

    python device_smoke.py --settings-id vllm-rp105-sched0-v1 --model-dir <model> --jobs jobs.json --call CALL_064 [--windows 4] --out device-smoke.json
    python device_smoke.py classify <log file> <exit code> [<report json>]

ORDER (REV 3.1 Amendment 1; enforced by the code structure and by tests): validate the profile -> pre_facts() (NO vLLM import, NO CUDA initialisation) -> build_decoder() -> post_facts().
vLLM resolves its platform class lazily on the FIRST access of `vllm.platforms.current_platform`, and the decoder sets the profile's VLLM_ENABLE_V1_MULTIPROCESSING itself BEFORE it imports vLLM;
looking at vLLM first would pick the NVML class on a MIG slice and reproduce the startup crash. post_facts() therefore begins with assert_profile_environment().

What it does: records the device facts, builds the decoder of the named settings id (--settings-id is REQUIRED: there is no default profile), decodes the first N windows of one call as ONE batch and records
native finish reasons, stop reasons, token counts and the empty-text count. It makes NO adapter-versus-package equality check (two separate generations are not an equality test; the equivalence evidence is
capture/replay). Exit code 0 only if every named check holds.

`classify` (REV 3.1 Amendment 3) reads the job log and the exit code and says WHICH kind of failure it was (memory, software, numerical), so a result on a smaller board is never mistaken for a finding
about the T4. It never changes a setting and never starts anything.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCHEDULING_VARIABLE = "VLLM_ENABLE_V1_MULTIPROCESSING"
T4_REFERENCE_GIB = 14.6                       # about the usable memory of a 16 GB T4; a reference to read the measured value against, replaced by the T1 measurement
MEMORY_BELOW_REFERENCE_GIB = 14.0
MEMORY_LIMITED_SIGNATURES = (                 # (text, where it comes from in vLLM 0.16.0 / torch)
    "CUDA out of memory", "OutOfMemoryError", "No available memory for the cache blocks",                  # vllm/v1/core/kv_cache_utils.py:617
    "To serve at least one request with the models's max seq len",                                           # kv_cache_utils.py:645-651
    "Free memory on device",                                                                                 # vllm/v1/worker/gpu_worker.py:506
)


# ---- pure helpers (importable on a CPU-only laptop) --------------------------------------------------------------------------------------------------------------------------------

def uuid_visible(env=None):
    env = os.environ if env is None else env
    return any(entry.strip().startswith(("MIG-", "GPU-")) for entry in (env.get("CUDA_VISIBLE_DEVICES") or "").split(","))


def first_visible_is_mig(env=None):
    env = os.environ if env is None else env
    return (env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip().startswith("MIG-")


def expected_platform_class(env=None):
    """NonNvmlCudaPlatform on a MIG slice WITH the tracked patch (single MIG device, in-process engine); NvmlCudaPlatform everywhere else."""
    return "NonNvmlCudaPlatform" if first_visible_is_mig(env) else "NvmlCudaPlatform"


def assert_profile_environment(settings_id, env=None):
    """Raise if the scheduling variable is not what the settings id's frozen table requires at this point: a reordered tool (vLLM looked at before the decoder was built) fails here, loudly."""
    from meralion_worker.decoders import frozen
    env = os.environ if env is None else env
    execution = frozen.execution_of(settings_id)
    found = env.get(SCHEDULING_VARIABLE)
    if execution is not None:
        if found != execution[SCHEDULING_VARIABLE]:
            raise RuntimeError(f"{SCHEDULING_VARIABLE} is {found!r} but settings {settings_id!r} require {execution[SCHEDULING_VARIABLE]!r}: vLLM was reached before the decoder set it (platform class would be wrong)")
    elif found not in (None, "1"):
        raise RuntimeError(f"{SCHEDULING_VARIABLE} is {found!r} but the default profile {settings_id!r} never runs with it")


def parse_gpu_query(csv_text):
    """`nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader` -> a list of dicts (one per device row)."""
    rows = []
    for line in csv_text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and not parts[0].lower().startswith("name"):
            rows.append({"name": parts[0], "memory_total": parts[1], "driver_version": parts[2], "compute_cap": parts[3]})
    return rows


def pre_facts(run=None, env=None):
    """What can be read WITHOUT importing vLLM or initialising CUDA: the device listing, the driver, the environment and the runtime identity (importlib.metadata and find_spec only)."""
    from meralion_worker.runtime_identity import runtime_identity
    env = os.environ if env is None else env
    run = run or subprocess.run

    def smi(*arguments):
        try:
            done = run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=60)
            return done.stdout.strip() if done.returncode == 0 else f"nvidia-smi failed ({done.returncode}): {done.stderr.strip()[:200]}"
        except (OSError, subprocess.SubprocessError) as error:
            return f"nvidia-smi unavailable: {error}"
    query = smi("--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader")
    return {"nvidia_smi_L": smi("-L").splitlines(), "gpu_query": parse_gpu_query(query), "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "mig_device_visible": first_visible_is_mig(env),
            "launcher_scheduling_variable": env.get(SCHEDULING_VARIABLE), "runtime": runtime_identity(), "t4_reference_usable_gib": T4_REFERENCE_GIB}


def build_decoder(settings_id, model_dir):
    """The ONLY place the decoder is built. The decoder validates its settings, sets the profile's scheduling variable and only then imports the package and vLLM."""
    from meralion_worker.decoders.vllm_decoder import VllmDecoder
    return VllmDecoder(model_dir, settings_id=settings_id)


def post_facts(decoder, settings_id, env=None):
    """What needs vLLM and CUDA: only after the decoder exists. Starts with assert_profile_environment."""
    assert_profile_environment(settings_id, env)
    import torch
    import vllm.platforms
    capability = torch.cuda.get_device_capability(0)
    properties = torch.cuda.get_device_properties(0)
    return {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "arch_list": list(torch.cuda.get_arch_list()), "capability": list(capability),
            "device": {"name": properties.name, "total_memory_gib": round(properties.total_memory / 2 ** 30, 2), "multi_processor_count": properties.multi_processor_count},
            "platform_class": type(vllm.platforms.current_platform).__name__, "engine_effective": decoder.describe()["engine_effective"]}


MEMORY_VOCABULARY = {      # REV 4.1 A3: what each memory figure IS (a report uses the words, never "slice memory" for anything else)
    "nvidia_smi_memory_used / torch.cuda.mem_get_info": "driver-level usage of the device: the CUDA context plus vLLM's pre-allocated pool (a RESERVATION of gpu_memory_utilization x total, not a minimum)",
    "memory_allocated / max_memory_allocated / memory_reserved": "the PyTorch allocator of THIS process only (meaningful for an in-process engine such as the sched0 profiles, not for the default multiprocess mode)",
    "vLLM log lines (Model loading took, Available KV cache memory)": "vLLM's own budget accounting",
    "memory.total": "the instance total ONLY when gpu_memory_scope is slice_verified (a MIG slice) or device_verified (a whole GPU); on a MIG node an unscoped query reports the parent card",
    "gpu_memory_scope": "identity first (the returned UUID is the allocated one), then capacity, then agreement of the numbers as a sanity check; reported, never gating",
}


def load_gpu_scope():
    """tools/gpu_scope.py by path (it is shipped next to tools/origami and has no package)."""
    spec = importlib.util.spec_from_file_location("gpu_scope", Path(__file__).resolve().parents[1] / "gpu_scope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def memory_scope_facts(*, run=None, env=None, torch_module=None, nvml=None):
    """After the decode, while the engine is alive and in the same process (sched0 is in-process): torch's device counters, the allocator figures, ONE scoped nvidia-smi sample (with its uuid column) and the
    NVML reading by UUID, judged by gpu_scope (identity -> capacity -> numbers). REPORTED and never gating: a measurement quirk must not fail the functional smoke. CUDA_VISIBLE_DEVICES is only read."""
    gs = load_gpu_scope()
    env = os.environ if env is None else env
    run = run or subprocess.run
    smi_L = ""
    try:
        done = run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
        smi_L = done.stdout if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        pass
    found = re.findall(r"UUID: ((?:GPU|MIG)-[0-9A-Za-z-]+)", smi_L)
    first = (env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    allocated = first if gs.uuid_kind(first) else (found[0] if found else None)
    parent = next((uuid for uuid in found if uuid.startswith("GPU-")), None)
    if torch_module is None:
        import torch as torch_module
    free, total = torch_module.cuda.mem_get_info()
    torch_total, torch_used = int(total) // 2 ** 20, (int(total) - int(free)) // 2 ** 20
    allocator = {name: int(getattr(torch_module.cuda, name)()) // 2 ** 20 for name in ("memory_allocated", "max_memory_allocated", "memory_reserved")}
    device = first or None
    sample = None
    try:
        done = run(["nvidia-smi", *(["-i", device] if device else []), "--query-gpu=timestamp,name,uuid,memory.used,memory.total,utilization.gpu", "--format=csv"], capture_output=True, text=True, timeout=60)
        sample = done.stdout if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    parent_total = None
    if gs.uuid_kind(allocated) == "mig" and parent:
        try:
            done = run(["nvidia-smi", "-i", parent, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
            parent_total = int(done.stdout.strip().splitlines()[0]) if done.returncode == 0 and done.stdout.strip().splitlines()[0].strip().isdigit() else None
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    verdict = gs.smoke_scope(allocated=allocated, parent=parent, one_shot=sample, torch_total_mib=torch_total, torch_used_mib=torch_used, parent_total_mib=parent_total)
    probe = gs.nvml_probe(allocated, nvml) if gs.uuid_kind(allocated) else {"available": False, "error": "no allocated UUID"}
    return {**verdict, "allocated_uuid": allocated, "parent_uuid": parent, "torch_device_counters_mib": {"total": torch_total, "used": torch_used}, "allocator_this_process_mib": allocator,
            "one_shot_nvidia_smi_sample": sample, "nvml_by_uuid": {"probe": probe, "verdict": gs.evaluate_nvml(allocated, probe, parent)}, "vocabulary": MEMORY_VOCABULARY}


def log_facts(log_text):
    """Facts a job log states about the engine (vLLM 0.16.0 formats, read from saved logs): KV cache size and concurrency, the attention backend that was selected, model memory, preemptions."""
    def first(pattern):
        match = re.search(pattern, log_text)
        return match.group(1) if match else None
    kv = first(r"GPU KV cache size: ([\d,]+) tokens")
    return {"kv_cache_tokens": int(kv.replace(",", "")) if kv else None, "maximum_concurrency": first(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
            "attention_backend_selected": first(r"Using (?:AttentionBackendEnum\.)?([A-Z_]+) backend"), "model_memory_gib": first(r"Model loading took ([\d.]+) GiB"),
            "available_kv_cache_gib": first(r"Available KV cache memory: ([\d.]+) GiB"), "preemption_lines": len(re.findall(r"preempt", log_text, flags=re.IGNORECASE))}


def plan_checks(*, settings_id, describe, execution, pre, post, rows, expect_capability=None, expect_platform=None, require_no_empty_text=False):
    """The named checks, from plain data (a truth table in the tests). A check that cannot be evaluated is False, never skipped."""
    from meralion_worker.decoders import frozen
    checks = {}
    checks["profile_digest_is_the_frozen_digest"] = describe.get("frozen_digest") == frozen.frozen_digest(settings_id) and describe.get("settings_id") == frozen.resolve_settings_id(settings_id)
    wanted_mode = "profile-in-process" if frozen.execution_of(settings_id) else "default"
    checks["execution_mode_matches_the_profile"] = (execution or {}).get("mode") == wanted_mode
    try:
        frozen.check_frozen(settings_id, (describe.get("settings") or {}).get("observed") or {})
        checks["observed_settings_pass_the_frozen_check"] = True
    except Exception:
        checks["observed_settings_pass_the_frozen_check"] = False
    if frozen.effective_engine_expectation(settings_id) is not None:
        checks["effective_engine_equals_the_profile_expectation"] = not frozen.engine_differences(settings_id, post.get("engine_effective"))
    capability = post.get("capability")
    checks["arch_list_contains_the_device_architecture"] = bool(capability) and f"sm_{capability[0]}{capability[1]}" in (post.get("arch_list") or [])
    if expect_capability is not None:
        checks["capability_is_the_expected_one"] = capability == list(expect_capability)
    checks["platform_class_is_the_expected_one"] = post.get("platform_class") == (expect_platform or expected_platform_class_from_pre(pre))
    checks["every_decode_ended_with_an_end_token_and_none_is_missing"] = bool(rows) and all(row.get("stop_reason") == "end_token" for row in rows)
    if require_no_empty_text:
        checks["no_empty_text_windows"] = not any(not (row.get("text") or "").strip() for row in rows)
    return checks


def expected_platform_class_from_pre(pre):
    return "NonNvmlCudaPlatform" if (pre or {}).get("mig_device_visible") else "NvmlCudaPlatform"


def classify_outcome(exit_code, log_text, report=None):
    """REV 3.1 Amendment 3: pass | memory_limited | software_failure | numerical_suspect | unclassified. Unknown text is never silently mapped to a known class."""
    report = report or {}
    if exit_code == 0 and report.get("passed") is True:
        return "pass"
    if report.get("completed") and (report.get("empty_text_windows", 0) > 0 or report.get("all_end_token") is False):
        return "numerical_suspect"                                                              # it ran to the end, but the output is empty or did not end properly
    if any(signature in log_text for signature in MEMORY_LIMITED_SIGNATURES):
        return "memory_limited"
    traceback = "Traceback (most recent call last)" in log_text
    build_error = "no kernel image is available" in log_text or "ninja: build stopped" in log_text or ("nvcc" in log_text and "error" in log_text.lower())
    failed_check = bool(report.get("checks")) and not all(report["checks"].values())            # a profile or device check failed
    if traceback or build_error or failed_check:
        return "software_failure"
    return "unclassified"


MEANING = {
    "pass": "the software path works on this board; EC2 compatibility is still pending T1",
    "memory_limited": "memory-driven: INCONCLUSIVE for the 16 GB T4 when this board has less memory than the T4 reference; a finding against the profile only on a board at or above it",
    "software_failure": "the cause is reported; a cluster-specific driver problem does not condemn the T4, an FP16 numerical failure probably does",
    "numerical_suspect": "the run completed but a decode is empty or did not end with an end token: FP16 numerical trouble is suspected",
    "unclassified": "the log shows no known signature: reported as it is, never mapped to a known class",
}


# ---- the run ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_jobs(path, call):
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))["jobs"]
    matches = [job for job in jobs if job["job_name"] == call]
    if len(matches) != 1:
        raise SystemExit(f"call {call!r} is not exactly once in {path}")
    return matches[0]["input"]


def run(args):
    from meralion_worker import audio, strategy_fg
    from meralion_worker.decoders import frozen
    from meralion_worker.decoders.base import DecodeRequest
    settings_id = frozen.resolve_settings_id(args.settings_id)                                  # 1. validate the profile (no vLLM here)
    frozen.execution_of(settings_id)
    pre = pre_facts()                                                                           # 2. no vLLM, no CUDA
    started = time.monotonic()
    decoder = build_decoder(settings_id, args.model_dir)                                        # 3. the decoder sets the scheduling variable, then imports vLLM
    load_seconds = round(time.monotonic() - started, 1)
    post = post_facts(decoder, settings_id)                                                     # 4. only now look at vLLM and CUDA
    recording = audio.load_recording(args.call, read_jobs(args.jobs, args.call))                # 5. decode
    bounds = strategy_fg.windows_for(recording.num_samples)[: args.windows]
    results = decoder.decode_many([DecodeRequest(args.call, a, b, recording.get(a, b)) for a, b in bounds])
    rows = [{"window": [a, b], "text_length": len(r.text), "text": r.text, "tokens": r.tokens, "stop_reason": r.stop_reason, "native": r.native} for (a, b), r in zip(bounds, results)]
    try:
        memory = memory_scope_facts()                                                           # REPORTED, never part of `checks`
    except Exception as error:
        memory = {"gpu_memory_scope": "unavailable", "reason": f"the measurement failed: {type(error).__name__}: {str(error)[:160]}", "vocabulary": MEMORY_VOCABULARY}
    describe = decoder.describe()
    checks = plan_checks(settings_id=settings_id, describe=describe, execution=decoder.execution, pre=pre, post=post, rows=rows,
                         expect_capability=tuple(int(x) for x in args.expect_capability.split(".")) if args.expect_capability else None, expect_platform=args.expect_platform,
                         require_no_empty_text=args.require_no_empty_text)
    empty = sum(1 for row in rows if not row["text"].strip())
    report = {"tool": "device_smoke", "settings_id": settings_id, "call": args.call, "windows": rows, "model_load_s": load_seconds, "pre_facts": pre, "post_facts": post,
              "execution": decoder.execution, "constructor_overrides": describe["constructor_overrides"], "checks": checks, "passed": all(checks.values()), "completed": True,
              "gpu_memory": memory, "empty_text_windows": empty, "all_end_token": all(row["stop_reason"] == "end_token" for row in rows), "worker_git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT"),
              "memory_reference": {"t4_usable_gib": T4_REFERENCE_GIB, "measured_total_memory_gib": post["device"]["total_memory_gib"]}}
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": report["passed"], "empty_text_windows": empty, "platform_class": post["platform_class"], "device": post["device"], "gpu_memory_scope": memory.get("gpu_memory_scope")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


def classify_command(argv):
    log_file, exit_code = argv[0], int(argv[1])
    report = json.loads(Path(argv[2]).read_text(encoding="utf-8")) if len(argv) > 2 and Path(argv[2]).is_file() else None
    text = Path(log_file).read_text(encoding="utf-8", errors="replace") if Path(log_file).is_file() else ""
    outcome = classify_outcome(exit_code, text, report)
    memory = (report or {}).get("memory_reference") or {}
    measured = memory.get("measured_total_memory_gib")
    note = MEANING[outcome]
    if outcome == "memory_limited" and measured is not None:
        note += f"; measured total memory {measured} GiB against the T4 reference {T4_REFERENCE_GIB} GiB: " + ("INCONCLUSIVE for the T4" if measured < MEMORY_BELOW_REFERENCE_GIB else "a finding against the profile on 16 GB")
    print(json.dumps({"outcome": outcome, "meaning": note, "log_facts": log_facts(text)}, indent=1))
    return 0 if outcome == "pass" else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings-id", required=True)                  # no default: a smoke never silently validates the stock profile
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--call", required=True)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-capability", default=None, help="MAJOR.MINOR, e.g. 12.0 (Blackwell) or 7.5 (T4)")
    parser.add_argument("--expect-platform", default=None, help="override the expected vLLM platform class name")
    parser.add_argument("--require-no-empty-text", action="store_true")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "classify":
        return classify_command(argv[1:])
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
