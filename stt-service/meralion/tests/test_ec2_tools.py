"""The EC2 measurement and reporting tools (REV 3.0 B5, REV 3.2 Amendments 7-8, REV 3.3 Amendment 9, REV 3.4): the verdict is pure and tested here; the measuring script runs against a FAKE docker and a fake
cgroup folder. Nothing contacts AWS or Docker.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
EC2 = TOOLS / "ec2"
MIB = 1024 * 1024
GIB = 1024 * MIB


def load(name, folder=EC2):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = load("resource_summary")


def working_bash():
    bash = shutil.which("bash")
    try:
        if bash and subprocess.run([bash, "-c", "true"], capture_output=True, timeout=20).returncode == 0:
            return bash
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


BASH = working_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no working bash")


def unix(path):
    posix = Path(path).as_posix()
    return f"/{posix[0].lower()}{posix[2:]}" if len(posix) > 2 and posix[1] == ":" else posix


DECLARATION = rs.make_declaration()
LIMIT = 24576


def sample(ts, current=3 * GIB, anon=1 * GIB, file=2 * GIB, shmem=0, usage=0, throttled=0, procs=1, swap=0):
    return {"ts_ms": ts, "mem_current": current, "anon": anon, "file": file, "shmem": shmem, "usage_usec": usage, "nr_throttled": throttled, "nr_procs": procs, "swap_current": swap}


def good_limits(limit_mib=LIMIT, **changes):
    """The limits as enforced and read at the start and in the final read: cgroup files and docker's HostConfig (REV 4.1 B3)."""
    cgroup = {"memory.max": str(limit_mib * MIB), "memory.swap.max": "0", "cpu.max": "400000 100000", "cpuset.cpus.effective": "0-3"}
    docker = {"Memory": limit_mib * MIB, "MemorySwap": limit_mib * MIB, "NanoCpus": 4 * 10 ** 9, "CpusetCpus": "0-3"}
    limits = {"cgroup_start": dict(cgroup), "cgroup_final": dict(cgroup), "docker_start": dict(docker), "docker_final": dict(docker)}
    limits.update(changes)
    return limits


def good_run(**over):
    run = {"limits": good_limits(), "samples": [sample(0), sample(2000, anon=4 * GIB), sample(4000, anon=3 * GIB)], "procs": [], "status": {"finality": "final_confirmed", "reason": "ok", "workload_exit": "0", "fresh_container": "1",
           "declaration_sha256": "d" * 64, "limit_mib": str(LIMIT)}, "memory_peak": 12 * GIB, "memory_events": {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0},
           "memory_pressure": {"some": {"avg10": 0.0, "total": 0}, "full": {"avg10": 0.0, "total": 0}}, "memory_swap_events": {"high": 0, "max": 0, "fail": 0}, "docker": {"OOMKilled": False, "Running": True},
           "completion": {"exit_code": 0, "documents_expected": 22, "documents_written": 22, "all_end_token": True, "failed_markers": 0}, "last_sampled": {}}
    run.update(over)
    return run


def judge(run, key="primary"):
    return rs.judge_run(run, DECLARATION, key, expect_declaration_sha256="d" * 64)


# ---- the declaration -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_declaration_fixes_the_limits_in_advance_and_is_hashable(tmp_path):
    assert DECLARATION["memory_limit_mib_primary"] == 24576 and DECLARATION["memory_limit_mib_downsize"] == 12288 and DECLARATION["cpus"] == 4 and DECLARATION["cpuset"] == "0-3"
    assert DECLARATION["build_and_thread_controls"] == {"MAX_JOBS": 4, "OMP_NUM_THREADS": 4, "MKL_NUM_THREADS": 4, "OPENBLAS_NUM_THREADS": 4, "TORCHINDUCTOR_COMPILE_THREADS": 4}
    assert DECLARATION["anon_fraction_gate"] == 0.80 and DECLARATION["host_margin_mib"] == 1024 and DECLARATION["sample_interval_s"] == 2 and "no swap" in DECLARATION["memory_swap_policy"]
    path = tmp_path / "decl.json"
    assert rs.main(["declaration", "--out", str(path)]) == 0
    first = rs.declaration_sha256(path)
    assert json.loads(path.read_text(encoding="utf-8")) == DECLARATION and first == rs.declaration_sha256(path)
    path.write_text(path.read_text(encoding="utf-8").replace("24576", "20000"), encoding="utf-8")
    assert rs.declaration_sha256(path) != first                                                     # a changed declared value is detectable from the recorded hash
    assert rs.limit_of(DECLARATION, "primary") == 24576 and rs.limit_of(DECLARATION, "downsize") == 12288
    with pytest.raises(ValueError):
        rs.limit_of(DECLARATION, "other")


def test_headroom_is_a_precondition_and_not_a_new_limit():
    assert rs.headroom_ok(25600, 24576, 1024) is True and rs.headroom_ok(25599, 24576, 1024) is False and rs.headroom_ok(None, 24576, 1024) is False


# ---- parsing and statistics --------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_parsers_read_the_tools_formats_and_ignore_damaged_lines():
    rows = rs.parse_samples("# header\n1000 3000 100 200 0 5000000 2 1 0\n2000 NA NA NA NA NA NA 1 NA\nshort line\n")
    assert len(rows) == 2 and rows[0]["anon"] == 100 and rows[1]["mem_current"] is None
    assert rs.parse_counters("low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n") == {"low": 0, "high": 0, "max": 3, "oom": 1, "oom_kill": 1}
    assert rs.parse_pressure("some avg10=0.50 avg60=0.10 avg300=0.00 total=1234\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n") == {"some": {"avg10": 0.5, "total": 1234}, "full": {"avg10": 0.0, "total": 0}}
    assert rs.parse_pressure("") is None and rs.parse_status("a=1\n# x\nb=two\n") == {"a": "1", "b": "two"}
    procs = rs.parse_procs("1000 12 1 2048 python\n1000 13 12 512000 cc1plus\nbad\n")
    assert [p["comm"] for p in procs] == ["python", "cc1plus"] and procs[1]["rss_kb"] == 512000


def test_a_spike_between_two_samples_is_not_seen_and_the_report_says_the_true_peak_lies_between_the_sampled_maximum_and_memory_peak():
    run = good_run(samples=[sample(0, anon=1 * GIB), sample(2000, anon=2 * GIB), sample(4000, anon=1 * GIB)], memory_peak=20 * GIB)             # a 15 GiB spike at t=3 s fell between the samples
    result = judge(run)
    assert result["stats"]["sampled_max_anon_shmem_mib"] == 2048.0 and result["kernel"]["memory_peak_mib"] == 20480.0 and result["verdict"] == "PASS"
    assert result["bracket"] == "sampled maximum; the true anonymous peak lies between this value and memory.peak" and rs.BRACKET in result["bracket"]


def test_the_actual_largest_gap_is_computed_from_the_timestamps_not_assumed():
    stats = rs.sample_stats([sample(0), sample(2000), sample(9000), sample(11000)], [], 2)
    assert stats["max_gap_s"] == 7.0 and stats["nominal_interval_s"] == 2 and stats["samples"] == 4


def test_cores_throttling_processes_and_concurrent_build_processes_are_reported():
    samples = [sample(0, usage=0, throttled=0), sample(10000, current=9 * GIB, usage=20_000_000, throttled=7)]
    procs = rs.parse_procs("0 1 0 1000 sleep\n10000 1 0 1000 sleep\n10000 20 1 6000000 python\n10000 21 20 900000 ninja\n10000 22 21 800000 cc1plus\n10000 23 21 700000 nvcc\n")
    stats = rs.sample_stats(samples, procs, 2)
    assert stats["average_cores"] == 2.0 and stats["throttled_periods"] == 7 and stats["peak_concurrent_build_processes"] == 3
    assert stats["largest_processes_at_peak"][0] == {"pid": 20, "comm": "python", "rss_mib": 5859.4}


# ---- the verdict per run ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_a_complete_clean_run_passes():
    result = judge(good_run())
    assert result["verdict"] == "PASS" and result["fail_reasons"] == result["inconsistent_reasons"] == result["flags"] == [] and result["finality"] == "final_confirmed"


def test_page_cache_alone_does_not_fail_a_run_but_anonymous_memory_above_the_fraction_does():
    cache_heavy = good_run(samples=[sample(0, current=24 * GIB, anon=1 * GIB, file=22 * GIB)], memory_peak=24 * GIB)
    assert judge(cache_heavy)["verdict"] == "PASS"                                                    # memory.current near the limit, but it is reclaimable cache
    heavy = good_run(samples=[sample(0, anon=int(0.81 * LIMIT * MIB))], memory_peak=24 * GIB)
    result = judge(heavy)
    assert result["verdict"] == "FAIL" and any("above 80%" in r for r in result["fail_reasons"])
    assert judge(good_run(samples=[sample(0, anon=int(0.79 * LIMIT * MIB))], memory_peak=24 * GIB))["verdict"] == "PASS"
    small = {**good_run()["status"], "limit_mib": "12288"}                                                       # the downsize run records its own declared limit
    assert judge(good_run(samples=[sample(0, anon=int(0.30 * 12288 * MIB))], memory_peak=24 * GIB, status=small, limits=good_limits(12288)), key="downsize")["verdict"] == "PASS"
    assert judge(good_run(samples=[sample(0, anon=int(0.85 * 12288 * MIB))], memory_peak=12 * GIB, status=small, limits=good_limits(12288)), key="downsize")["verdict"] == "FAIL"


@pytest.mark.parametrize("change, fragment", [
    (dict(memory_events={"max": 0, "oom": 1, "oom_kill": 0}), "oom=1"), (dict(memory_events={"max": 0, "oom": 0, "oom_kill": 1}), "oom_kill=1"),
    (dict(docker={"OOMKilled": True}), "OOM-killed"), (dict(samples=[sample(0, swap=1024)]), "swap"), (dict(memory_swap_events={"fail": 2}), "swap"),
    (dict(status={**good_run()["status"], "workload_exit": "3"}), "exited with 3"),
    (dict(completion={"exit_code": 1, "documents_expected": 22, "documents_written": 22, "all_end_token": True, "failed_markers": 0}), "worker exited with 1"),
    (dict(completion={"exit_code": 0, "documents_expected": 22, "documents_written": 21, "all_end_token": True, "failed_markers": 0}), "21 documents written"),
    (dict(completion={"exit_code": 0, "documents_expected": 22, "documents_written": 22, "all_end_token": False, "failed_markers": 0}), "did not end"),
    (dict(completion={"exit_code": 0, "documents_expected": 22, "documents_written": 22, "all_end_token": True, "failed_markers": 1}), "failed markers"),
])
def test_each_failure_condition_fails_the_run(change, fragment):
    result = judge(good_run(**change))
    assert result["verdict"] == "FAIL" and any(fragment in r for r in result["fail_reasons"]), result["fail_reasons"]


@pytest.mark.parametrize("change, fragment", [
    (dict(memory_peak=None), "memory.peak is missing"), (dict(samples=[sample(0, anon=10 * GIB)], memory_peak=4 * GIB), "exceeds memory.peak"),
    (dict(samples=[]), "no samples"), (dict(completion=None), "no completion record"),
    (dict(status={**good_run()["status"], "finality": "last_sampled", "reason": "final_metrics_missing: x"}), "final_metrics_missing"),
    (dict(status={**good_run()["status"], "finality": "unavailable", "reason": "never read"}), "not final_confirmed"),
    (dict(status={**good_run()["status"], "fresh_container": "0"}), "own fresh container"),
    (dict(status={**good_run()["status"], "declaration_sha256": "e" * 64}), "declaration hash"),
    (dict(status={**good_run()["status"], "limit_mib": "12288"}), "recorded limit 12288"),
])
def test_unusable_evidence_is_inconsistent_never_a_pass(change, fragment):
    result = judge(good_run(**change))
    assert result["verdict"] == "INCONSISTENT" and any(fragment in r for r in result["inconsistent_reasons"]), result["inconsistent_reasons"]


def test_events_at_the_limit_and_a_pressure_stall_are_flagged_not_auto_passed():
    flagged = judge(good_run(memory_events={"low": 0, "high": 0, "max": 5, "oom": 0, "oom_kill": 0}))
    assert flagged["verdict"] == "PASS_FLAGGED" and "max=5" in flagged["flags"][0]
    stalled = judge(good_run(memory_pressure={"some": {"avg10": 1.0, "total": 9}, "full": {"avg10": 0.2, "total": 4000}}))
    assert stalled["verdict"] == "PASS_FLAGGED" and "full stall total 4000" in stalled["flags"][0]
    assert judge(good_run(memory_pressure={"some": {"avg10": 1.0, "total": 9}, "full": {"avg10": 0.0, "total": 0}}))["verdict"] == "PASS"     # `some` is reported, only `full` flags


def test_a_failure_wins_over_missing_final_metrics_and_the_last_sampled_values_never_turn_into_a_pass():
    status = {**good_run()["status"], "finality": "last_sampled", "reason": "final_metrics_missing: the holder died"}
    killed = judge(good_run(status=status, docker={"OOMKilled": True}, memory_peak=None, last_sampled={"memory_peak": 10 * GIB, "memory_events": {"oom_kill": 0}}))
    assert killed["verdict"] == "FAIL" and killed["inconsistent_reasons"] and killed["last_sampled_substitute"]["memory_peak_mib"] == 10240.0
    assert "never final, never a pass" in killed["last_sampled_substitute"]["label"]
    missing = judge(good_run(status=status, memory_peak=None, last_sampled={"memory_peak": 10 * GIB}))
    assert missing["verdict"] == "INCONSISTENT"


# ---- over all runs: the maximum, not an assumption about which is largest ------------------------------------------------------------------------------------------------------------

def test_the_overall_maximum_is_taken_over_every_run_even_when_the_warm_run_is_larger():
    cold = good_run(samples=[sample(0, anon=3 * GIB)], memory_peak=8 * GIB)
    warm = good_run(samples=[sample(0, anon=7 * GIB)], memory_peak=12 * GIB)
    summary = rs.summarize_runs({"cold": cold, "warm": warm, "call103": good_run(samples=[sample(0, anon=5 * GIB)], memory_peak=9 * GIB)}, DECLARATION, "primary", "d" * 64)
    assert summary["overall"]["sampled_max_anon_shmem_mib_over_all_runs"] == 7168.0 and summary["overall"]["memory_peak_mib_over_all_runs"] == 12288.0
    assert "not assumed" in summary["overall"]["which_run_is_largest"] and summary["overall"]["verdict"] == "PASS"
    worse = rs.summarize_runs({"cold": cold, "warm": good_run(status={**good_run()["status"], "workload_exit": "1"})}, DECLARATION, "primary", "d" * 64)
    assert worse["overall"]["verdict"] == "FAIL"
    assert rs.summarize_runs({}, DECLARATION, "primary")["overall"]["verdict"] == "INCONSISTENT"                 # no run is not a pass


def test_the_downsize_launch_rule_is_a_screening_rule_on_a_lower_bound():
    allowed = rs.downsize_screening(9830.0, DECLARATION)
    assert allowed["launch_allowed"] is True and allowed["ceiling_mib"] == 9830.4 and "screening" in allowed["nature"]
    assert rs.downsize_screening(9831.0, DECLARATION)["launch_allowed"] is False and rs.downsize_screening(None, DECLARATION)["launch_allowed"] is False
    one = rs.summarize_runs({"cold": good_run(samples=[sample(0, anon=9 * GIB)], memory_peak=12 * GIB)}, DECLARATION, "primary", "d" * 64)
    assert one["downsize_screening"]["launch_allowed"] is True
    assert rs.summarize_runs({"x": good_run()}, DECLARATION, "downsize", "d" * 64)["downsize_screening"] is None


# ---- T2: functional and resource evidence are separate -------------------------------------------------------------------------------------------------------------------------------

def test_t2_functional_success_stands_when_only_last_sampled_metrics_survive_and_the_production_image_memory_is_not_validated():
    completion = {"exit_code": 0, "documents_expected": 1, "documents_written": 1, "all_end_token": True, "failed_markers": 0}
    result = rs.judge_t2(completion, {"finality": "last_sampled"})
    assert (result["functional"], result["resource_evidence"], result["production_image_memory"]) == ("pass", "last_sampled", "not_validated_by_T2")
    assert rs.judge_t2(completion, {})["resource_evidence"] == "unavailable" and rs.judge_t2(completion, None)["functional"] == "pass"
    assert rs.judge_t2(completion, {"finality": "final_confirmed"})["production_image_memory"] == "not_validated_by_T2"          # a status field alone NEVER validates the image (REV 4.1 B5)
    assert rs.judge_t2({**completion, "exit_code": 1}, {"finality": "final_confirmed"})["functional"] == "fail" and rs.judge_t2(None, {})["functional"] == "not_established"


# ---- REV 4.1 B1-B6: nothing required is defaulted, bytes not rounded MiB, one swap rule, flagged runs wait for review -----------------------------------------------------------------------------

def test_codexs_probes_are_no_longer_a_pass():
    assert judge(good_run(memory_events=None))["verdict"] == "INCONSISTENT"                                                                  # was PASS
    assert judge(good_run(samples=[{**sample(0), "anon": None, "shmem": None}, {**sample(2000), "anon": None, "shmem": None}]))["verdict"] == "INCONSISTENT"      # was PASS
    assert judge(good_run(completion={}))["verdict"] == "INCONSISTENT"                                                                       # was PASS (None == None)
    assert rs.judge_t2({"exit_code": 0}, {"finality": "final_confirmed"}) == {**rs.judge_t2({"exit_code": 0}, {"finality": "final_confirmed"}), "functional": "not_established", "production_image_memory": "not_validated_by_T2"}


@pytest.mark.parametrize("events", [{}, {"oom": 0, "oom_kill": 0}, {"oom": 0, "oom_kill": 0, "max": "0"}, {"oom": None, "oom_kill": 0, "max": 0}, [], "oom 0"])
def test_a_memory_events_record_without_integer_oom_oom_kill_and_max_is_inconsistent_never_a_pass(events):
    result = judge(good_run(memory_events=events))
    assert result["verdict"] == "INCONSISTENT" and any("memory.events" in r for r in result["inconsistent_reasons"])


@pytest.mark.parametrize("completion, fragment", [
    ({"exit_code": 0, "documents_expected": 22, "documents_written": 22, "all_end_token": True}, "failed_markers"),
    ({"exit_code": 0, "documents_expected": 22, "documents_written": 22, "failed_markers": 0}, "all_end_token"),
    ({"exit_code": 0, "documents_expected": 22, "all_end_token": True, "failed_markers": 0}, "documents_written"),
    ({"documents_expected": 22, "documents_written": 22, "all_end_token": True, "failed_markers": 0}, "exit_code"),
    ({"exit_code": 0, "documents_expected": 0, "documents_written": 0, "all_end_token": True, "failed_markers": 0}, "below 1"),
    ({"exit_code": 0, "documents_expected": 22, "documents_written": "22", "all_end_token": True, "failed_markers": 0}, "documents_written"),
    ({"exit_code": 0, "documents_expected": 22, "documents_written": 22, "all_end_token": "yes", "failed_markers": 0}, "all_end_token"),
])
def test_an_incomplete_or_malformed_completion_record_is_inconsistent_and_a_bad_value_in_a_complete_one_still_fails(completion, fragment):
    result = judge(good_run(completion=completion))
    assert result["verdict"] == "INCONSISTENT" and any(fragment in r for r in result["inconsistent_reasons"]), result["inconsistent_reasons"]
    bad = judge(good_run(completion={"exit_code": 1, "documents_expected": 22, "documents_written": 21, "all_end_token": False, "failed_markers": 2}))
    assert bad["verdict"] == "FAIL" and len(bad["fail_reasons"]) == 3                                                                  # present-but-bad values remain FAIL


def test_the_sampled_maximum_is_compared_in_bytes_so_rounding_cannot_invent_an_inconsistency_or_hide_a_failure():
    peak = int(1024.06 * MIB)                                                                                  # 1024.06 MiB displays as 1024.1 MiB: the rounded figure times MIB is ABOVE the peak
    result = judge(good_run(samples=[sample(0, anon=peak)], memory_peak=peak))
    assert result["verdict"] == "PASS" and result["stats"]["sampled_max_anon_shmem_bytes"] == peak and result["stats"]["sampled_max_anon_shmem_mib"] == 1024.1
    assert judge(good_run(samples=[sample(0, anon=peak)], memory_peak=peak - 1))["verdict"] == "INCONSISTENT"                    # one byte above the kernel peak is a real inconsistency
    gate = int(0.80 * LIMIT * MIB)
    assert judge(good_run(samples=[sample(0, anon=gate)], memory_peak=24 * GIB))["verdict"] == "PASS"                                # exactly at the gate
    assert judge(good_run(samples=[sample(0, anon=gate + 1)], memory_peak=24 * GIB))["verdict"] == "FAIL"                            # one byte above it (a rounded comparison would pass)


def test_a_sample_missing_anon_or_shmem_is_incomplete_not_zero_and_coverage_is_reported():
    only_shmem = {**sample(2000), "anon": None, "shmem": 0}                                                    # shmem readable (0) with anon unreadable: the total is NOT 0
    only_anon = {**sample(4000), "anon": 5 * GIB, "shmem": None}
    stats = rs.sample_stats([sample(0, anon=GIB), only_shmem, only_anon, sample(6000, anon=2 * GIB)], [], 2)
    assert stats["samples"] == 4 and stats["samples_complete"] == 2 and stats["samples_incomplete"] == 2
    assert stats["sampled_max_anon_shmem_bytes"] == 2 * GIB and stats["complete_first_ts_ms"] == 0 and stats["complete_last_ts_ms"] == 6000 and stats["max_gap_between_complete_s"] == 6.0
    result = judge(good_run(samples=[sample(0, anon=GIB), only_shmem, only_anon, sample(6000, anon=2 * GIB)]))
    assert result["verdict"] == "PASS_FLAGGED" and "2 of 4 samples lack anon or shmem" in result["flags"][0] and "not counted as 0" in result["flags"][0]
    none_complete = judge(good_run(samples=[only_shmem, only_anon]))
    assert none_complete["verdict"] == "INCONSISTENT" and any("both anon and shmem" in r for r in none_complete["inconsistent_reasons"])
    assert judge(good_run(samples=[sample(0), sample(2000)]))["verdict"] == "PASS"                              # complete coverage: no flag


@pytest.mark.parametrize("limits_change, fragment", [
    (dict(cgroup_start={"memory.max": str(LIMIT * MIB), "cpu.max": "400000 100000", "cpuset.cpus.effective": "0-3"}), "memory.swap.max was not read at the start"),
    (dict(cgroup_final={"memory.max": str(LIMIT * MIB), "memory.swap.max": None, "cpu.max": "400000 100000", "cpuset.cpus.effective": "0-3"}), "memory.swap.max was not read at the final"),
    (dict(cgroup_start={**good_limits()["cgroup_start"], "memory.swap.max": "max"}), "swap is not prohibited"),
    (dict(cgroup_final={**good_limits()["cgroup_final"], "memory.swap.max": "1048576"}), "swap is not prohibited"),
    (dict(docker_start={**good_limits()["docker_start"], "MemorySwap": 2 * LIMIT * MIB}), "MemorySwap must equal Memory"),
    (dict(docker_final={**good_limits()["docker_final"], "MemorySwap": -1}), "MemorySwap must equal Memory"),
    (dict(cgroup_final={**good_limits()["cgroup_final"], "memory.max": str(1 * GIB)}), "memory.max is"),
    (dict(cgroup_start={**good_limits()["cgroup_start"], "cpu.max": "800000 100000"}), "cpu.max is"),
    (dict(cgroup_start={**good_limits()["cgroup_start"], "cpuset.cpus.effective": "0-7"}), "cpuset.cpus.effective is"),
    (dict(docker_start={**good_limits()["docker_start"], "NanoCpus": 8 * 10 ** 9}), "NanoCpus"),
    (dict(docker_final=None), "HostConfig was not read at the final"),
])
def test_declared_limits_not_shown_to_be_enforced_are_inconsistent_and_the_swap_rule_is_one_rule(limits_change, fragment):
    result = judge(good_run(limits=good_limits(**limits_change)))
    assert result["verdict"] == "INCONSISTENT" and any(fragment in r for r in result["inconsistent_reasons"]), result["inconsistent_reasons"]
    assert result["limits_enforced"] is False


def test_a_readable_zero_swap_usage_does_not_stand_in_for_the_swap_limit_and_missing_limits_are_inconsistent():
    absent = judge(good_run(limits=None))
    assert absent["verdict"] == "INCONSISTENT" and "no limits record" in " ".join(absent["inconsistent_reasons"])
    no_swap_max = judge(good_run(limits=good_limits(cgroup_start={**good_limits()["cgroup_start"], "memory.swap.max": None}, cgroup_final={**good_limits()["cgroup_final"], "memory.swap.max": None}),
                                 samples=[sample(0, swap=0), sample(2000, swap=0)]))
    assert no_swap_max["verdict"] == "INCONSISTENT"                                                                      # swap_current == 0 in every sample proves nothing about what was prohibited
    assert judge(good_run())["limits_enforced"] is True and judge(good_run())["verdict"] == "PASS"
    assert judge(good_run(samples=[sample(0, swap=4096)]))["verdict"] == "FAIL"                                          # observed swap use is still a failure on top


# ---- B5: no automatic validated; the claim is bound to the measured context ------------------------------------------------------------------------------------------------------------------

CONTEXT = {"image_digest": "sha256:" + "a" * 64, "settings_id": "vllm-rp105-sched0-fp16-v1", "frozen_digest": "f" * 64, "config_fingerprint": "c" * 64, "checkpoint_manifest_sha256": "m" * 64,
           "limits": {"memory_mib": 24576, "memory_swap_mib": 24576, "cpus": 4, "cpuset": "0-3"}, "gpu_model": "Tesla T4", "driver": "570.172.08", "instance_type": "g4dn.2xlarge",
           "workload": {"jobs_sha256": "j" * 64, "recordings": 22, "largest_recording_windows": 125}}
COMPLETION = {"exit_code": 0, "documents_expected": 1, "documents_written": 1, "all_end_token": True, "failed_markers": 0}


def image_run(**over):
    result = judge(good_run(context=dict(CONTEXT), run_id="image-run-1", **over))
    return result


def flagged_run():
    return image_run(memory_events={"low": 0, "high": 0, "max": 5, "oom": 0, "oom_kill": 0})


def t2(image, disposition=None, context=CONTEXT):
    return rs.judge_t2(COMPLETION, {"finality": "last_sampled"}, image_run=image, disposition=disposition, production_context=dict(context))


def disposition_for(image, **over):
    data = {"run_id": image["run_id"], "resource_record_sha256": image["record_sha256"], "flags_reviewed": image["flags"], "accepted": True, "reviewer": "owner", "date": "2026-11-18", "reason": "page-cache reclaim"}
    data.update(over)
    return data


def test_a_clean_image_run_validates_for_exactly_its_measured_context_and_says_so():
    result = t2(image_run())
    assert result["production_image_memory"] == "validated_for_measured_context" and result["validated_for"] == CONTEXT and result["functional"] == "pass"


def test_a_flagged_image_run_stays_pending_review_until_a_matching_disposition_accepts_that_record():
    flagged = flagged_run()
    assert flagged["verdict"] == "PASS_FLAGGED" and flagged["record_sha256"]
    pending = t2(flagged)
    assert pending["production_image_memory"] == "pending_review" and "max=5" in pending["production_image_memory_notes"][0] and pending["validated_for"] is None
    assert t2(flagged, disposition_for(flagged))["production_image_memory"] == "validated_for_measured_context"
    for label, bad in (("another run", disposition_for(flagged, run_id="image-run-2")), ("another record hash", disposition_for(flagged, resource_record_sha256="0" * 64)),
                       ("not accepted", disposition_for(flagged, accepted=False)), ("accepted as a string", disposition_for(flagged, accepted="true")), ("no hash", {"run_id": "image-run-1", "accepted": True})):
        assert t2(flagged, bad)["production_image_memory"] == "pending_review", label
    changed = image_run(memory_events={"low": 0, "high": 0, "max": 9, "oom": 0, "oom_kill": 0})                              # the record changed after the review
    assert changed["record_sha256"] != flagged["record_sha256"] and t2(changed, disposition_for(flagged))["production_image_memory"] == "pending_review"


def test_incomplete_sample_coverage_also_waits_for_a_review():
    partial = image_run(samples=[sample(0), {**sample(2000), "anon": None}, sample(4000)])
    assert partial["verdict"] == "PASS_FLAGGED" and t2(partial)["production_image_memory"] == "pending_review"


@pytest.mark.parametrize("key, value", [("image_digest", "sha256:" + "b" * 64), ("settings_id", "vllm-rp105-sched0-v1"), ("frozen_digest", "e" * 64), ("config_fingerprint", "d" * 64),
                                        ("checkpoint_manifest_sha256", "n" * 64), ("limits", {"memory_mib": 32768, "memory_swap_mib": 32768, "cpus": 4, "cpuset": "0-3"}), ("gpu_model", "L40S"),
                                        ("driver", "590.48.01"), ("instance_type", "g4dn.xlarge"), ("workload", {"jobs_sha256": "j" * 64, "recordings": 175, "largest_recording_windows": 125})])
def test_the_same_image_digest_is_not_enough_each_context_entry_must_match(key, value):
    result = t2(image_run(), context={**CONTEXT, key: value})
    assert result["production_image_memory"] == "context_differs" and any(note.startswith(key) for note in result["production_image_memory_notes"])
    assert result["validated_for"] is None


def test_a_context_entry_missing_on_either_side_is_a_difference_never_a_match():
    without = {k: v for k, v in CONTEXT.items() if k != "driver"}
    assert t2(image_run(), context=without)["production_image_memory"] == "context_differs"
    unrecorded = judge(good_run(context=without, run_id="x"))
    assert t2(unrecorded)["production_image_memory"] == "context_differs" and rs.context_differences(None, CONTEXT) and rs.context_differences(CONTEXT, None)


@pytest.mark.parametrize("change", [dict(memory_events={"low": 0, "high": 0, "max": 0, "oom": 1, "oom_kill": 1}), dict(memory_events=None), dict(limits=None)])
def test_a_failed_or_inconsistent_image_run_never_validates(change):
    assert t2(image_run(**change))["production_image_memory"] == "not_validated_by_T2"
    assert rs.judge_t2(COMPLETION, {"finality": "final_confirmed"}, image_run=None, production_context=CONTEXT)["production_image_memory"] == "not_validated_by_T2"


def test_t2_functional_needs_a_complete_good_completion_record_and_is_independent_of_the_resource_evidence():
    assert rs.judge_t2({"exit_code": 0}, {"finality": "final_confirmed"})["functional"] == "not_established"
    assert rs.judge_t2({"exit_code": 2}, {})["functional"] == "fail" and rs.judge_t2({**COMPLETION, "documents_written": 0}, {})["functional"] == "fail"
    assert rs.judge_t2({**COMPLETION, "all_end_token": False}, {})["functional"] == "fail" and rs.judge_t2(None, None)["functional"] == "not_established"
    assert rs.judge_t2(COMPLETION, {"finality": "unavailable"})["functional"] == "pass"                                  # unavailable resource metrics never fail a functionally successful run


def test_the_summary_records_each_runs_context_and_record_hash_for_the_review():
    summary = rs.summarize_runs({"image": good_run(context=dict(CONTEXT), run_id="image-run-1")}, DECLARATION, "primary", "d" * 64)
    record = summary["runs"]["image"]
    assert record["run_id"] == "image-run-1" and record["context"] == CONTEXT and len(record["record_sha256"]) == 64 and record["limits_enforced"] is True


# ---- the measuring script against a fake docker -----------------------------------------------------------------------------------------------------------------------------------

FAKE_DOCKER = r'''#!/bin/sh
echo "docker $*" >> "$FAKE_STATE/calls.log"
cmd=$1; shift
cg="$CGROUP_BASE/system.slice/docker-cid123.scope"
case "$cmd" in
  run)
    if [ "$FAKE_SCENARIO" != no_cgroup ]; then
      mkdir -p "$cg"
      echo 3000000000 > "$cg/memory.current"
      printf 'anon 1073741824\nfile 5368709120\nshmem 1048576\n' > "$cg/memory.stat"
      printf 'usage_usec 1000000\nnr_throttled 0\n' > "$cg/cpu.stat"
      echo 1 > "$cg/cgroup.procs"
      echo 8589934592 > "$cg/memory.peak"
      mem=24576; prev=""
      for a in "$@"; do if [ "$prev" = "--memory" ]; then mem=${a%m}; fi; prev=$a; done
      membytes=$((mem * 1048576))
      echo "$membytes" > "$FAKE_STATE/membytes"
      echo "$membytes" > "$cg/memory.max"
      if [ "$FAKE_SCENARIO" = bad_swap ]; then echo max > "$cg/memory.swap.max"; elif [ "$FAKE_SCENARIO" != no_swap_file ]; then echo 0 > "$cg/memory.swap.max"; fi
      echo 0 > "$cg/memory.swap.current"
      echo "400000 100000" > "$cg/cpu.max"
      echo "0-3" > "$cg/cpuset.cpus.effective"
      printf 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n' > "$cg/memory.events"
      printf 'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n' > "$cg/memory.pressure"
    fi
    echo running > "$FAKE_STATE/holder"
    echo cid123 ;;
  inspect)
    if [ "$1" = "-f" ]; then if [ "$(cat "$FAKE_STATE/holder")" = running ]; then echo true; else echo false; fi
    else
      mb=$(cat "$FAKE_STATE/membytes" 2>/dev/null || echo 25769803776); ms=$mb
      if [ "$FAKE_SCENARIO" = swap_allowed ]; then ms=$((mb * 2)); fi
      echo '[{"State":{"Running":true,"OOMKilled":false,"ExitCode":0},"HostConfig":{"Memory":'"$mb"',"MemorySwap":'"$ms"',"NanoCpus":4000000000,"CpusetCpus":"0-3"}}]'
    fi ;;
  exec)
    shift
    all="$*"
    case "$all" in
      *"kill -0"*) [ -f "$FAKE_STATE/alive" ]; exit $? ;;
      *"kill -TERM"*) if [ "$FAKE_SCENARIO" = hang_term_ok ]; then rm -f "$FAKE_STATE/alive"; echo 1 > "$cg/cgroup.procs"; fi; exit 0 ;;
      *"kill -KILL"*) exit 0 ;;
      *"workload.pid"*)
        case "$FAKE_SCENARIO" in
          ok|no_cgroup|bad_swap|no_swap_file|swap_allowed) sleep 1.5; exit 0 ;;
          fail) sleep 1.5; exit 3 ;;
          hang_term_ok|hang_never_dies) touch "$FAKE_STATE/alive"; printf '1\n2\n' > "$cg/cgroup.procs"; exec sleep 30 ;;
          holder_dies) sleep 1.2; rm -rf "$cg"; echo exited > "$FAKE_STATE/holder"; exit 137 ;;
        esac ;;
    esac ;;
  rm) echo removed >> "$FAKE_STATE/removed"; rm -rf "$cg" ;;
esac
exit 0
'''


class Fake:
    def __init__(self, tmp_path, scenario, available=40000):
        self.tmp = tmp_path
        self.state = tmp_path / "state"
        self.state.mkdir()
        self.cgroups = tmp_path / "cgroups"
        self.cgroups.mkdir()
        self.out = tmp_path / "run-out"
        self.docker = tmp_path / "docker"
        self.docker.write_text(FAKE_DOCKER, encoding="utf-8", newline="\n")
        self.docker.chmod(0o755)
        self.free = tmp_path / "free.txt"
        self.free.write_text(f"              total        used        free      shared  buff/cache   available\nMem:          65000        2000       30000         100       30000       {available}\nSwap:             0           0           0\n", encoding="utf-8")
        self.declaration = tmp_path / "decl.json"
        data = dict(DECLARATION, sample_interval_s=1, workload_timeout_seconds=2, terminate_grace_seconds=2)
        self.declaration.write_text(json.dumps(data), encoding="utf-8")
        self.env = {**os.environ, "DOCKER": unix(self.docker), "CGROUP_BASE": unix(self.cgroups), "FAKE_STATE": unix(self.state), "FAKE_SCENARIO": scenario, "FREE_CMD": f"cat {unix(self.free)}",
                    "PYTHON": unix(sys.executable), "DOCKER_EXTRA_ARGS": "--gpus all"}

    def run(self, key="primary", workload=("echo", "hello")):
        return subprocess.run([BASH, unix(EC2 / "measure_resources.sh"), "run1", unix(self.out), unix(self.declaration), key, "image:tag", "--", *workload], capture_output=True, text=True, env=self.env, timeout=120)

    def calls(self):
        path = self.state / "calls.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def status(self):
        return rs.parse_status((self.out / "final" / "status.env").read_text(encoding="utf-8"))

    def run_dir(self):
        return rs.load_run(self.out)


@needs_bash
def test_a_normal_run_reads_the_final_metrics_while_the_container_is_alive_then_removes_it(tmp_path):
    fake = Fake(tmp_path, "ok")
    done = fake.run()
    assert done.returncode == 0, done.stdout + done.stderr
    status = fake.status()
    assert status["finality"] == "final_confirmed" and status["terminated_confirmed"] == "1" and status["holder_alive"] == "1" and status["fresh_container"] == "1" and status["workload_exit"] == "0"
    assert status["declaration_sha256"] == rs.declaration_sha256(fake.declaration) and status["limit_mib"] == "24576"
    assert (fake.out / "final" / "memory.peak").read_text(encoding="utf-8").strip() == "8589934592" and (fake.out / "final" / "docker-inspect.json").is_file()
    calls = fake.calls()
    run_call = next(c for c in calls if c.startswith("docker run"))
    for fragment in ("-d", "--name run1", "--cpus 4", "--cpuset-cpus 0-3", "--memory 24576m", "--memory-swap 24576m", "-e MAX_JOBS=4", "-e OMP_NUM_THREADS=4", "-e TORCHINDUCTOR_COMPILE_THREADS=4", "--gpus all", "image:tag sleep infinity"):
        assert fragment in run_call, fragment
    assert "--rm" not in run_call.split() and not [c for c in calls if c.startswith("docker run") and " --rm " in c]
    remove = next(i for i, c in enumerate(calls) if c.startswith("docker rm"))
    workload = next(i for i, c in enumerate(calls) if "workload.pid" in c)
    inspect = [i for i, c in enumerate(calls) if c.startswith("docker inspect") and "-f" not in c][-1]
    assert workload < inspect < remove == len(calls) - 1                                                   # the container is removed LAST, after the final read
    run = fake.run_dir()
    assert run["samples"] and run["memory_peak"] == 8 * GIB
    run["completion"] = {"exit_code": 0, "documents_expected": 1, "documents_written": 1, "all_end_token": True, "failed_markers": 0}
    assert rs.judge_run(run, json.loads(fake.declaration.read_text(encoding="utf-8")), "primary", expect_declaration_sha256=rs.declaration_sha256(fake.declaration))["verdict"] == "PASS"


@needs_bash
def test_a_failing_workload_still_yields_the_final_metrics_before_removal(tmp_path):
    fake = Fake(tmp_path, "fail")
    done = fake.run()
    assert done.returncode == 3
    status = fake.status()
    assert status["workload_exit"] == "3" and status["finality"] == "final_confirmed" and (fake.out / "final" / "memory.events").is_file()
    calls = fake.calls()
    assert [i for i, c in enumerate(calls) if c.startswith("docker rm")][0] > max(i for i, c in enumerate(calls) if c.startswith("docker inspect"))
    run = fake.run_dir()
    run["completion"] = {"exit_code": 3, "documents_expected": 1, "documents_written": 0, "all_end_token": True, "failed_markers": 0}
    assert rs.judge_run(run, json.loads(fake.declaration.read_text(encoding="utf-8")), "primary")["verdict"] == "FAIL"


@needs_bash
def test_an_exec_client_that_times_out_while_the_workload_runs_is_not_taken_for_a_finished_workload(tmp_path):
    fake = Fake(tmp_path, "hang_term_ok")
    done = fake.run()
    status = fake.status()
    assert done.returncode == 124 and status["client_timed_out"] == "1"
    calls = fake.calls()
    assert any("kill -TERM" in c for c in calls)                                                           # the workload was terminated, not assumed gone
    assert status["terminated_confirmed"] == "1" and status["finality"] == "final_confirmed"


@needs_bash
def test_a_workload_that_cannot_be_confirmed_terminated_is_never_final(tmp_path):
    fake = Fake(tmp_path, "hang_never_dies")
    fake.run()
    status = fake.status()
    calls = fake.calls()
    assert any("kill -TERM" in c for c in calls) and any("kill -KILL" in c for c in calls)
    assert status["terminated_confirmed"] == "0" and status["finality"] == "last_sampled" and "could not be confirmed terminated" in status["reason"]
    run = fake.run_dir()
    run["completion"] = {"exit_code": 0, "documents_expected": 1, "documents_written": 1, "all_end_token": True, "failed_markers": 0}
    verdict = rs.judge_run(run, json.loads(fake.declaration.read_text(encoding="utf-8")), "primary")
    assert verdict["verdict"] == "FAIL" and any("exited with 124" in r for r in verdict["fail_reasons"])            # the timed-out client is a failure; and the missing final values are recorded too
    assert any("final_metrics_missing" in r for r in verdict["inconsistent_reasons"])


@needs_bash
def test_a_holder_that_dies_before_the_final_read_leaves_an_explicit_marker_and_only_last_sampled_values(tmp_path):
    fake = Fake(tmp_path, "holder_dies")
    done = fake.run()
    status = fake.status()
    assert done.returncode == 137 and status["holder_alive"] == "0" and status["finality"] == "last_sampled" and "final_metrics_missing" in status["reason"]
    assert not (fake.out / "final" / "memory.peak").exists()                                               # nothing stale is presented as final
    assert (fake.out / "last_sampled" / "memory.peak").is_file()                                           # the labelled substitute, read while the container lived
    run = fake.run_dir()
    verdict = rs.judge_run({**run, "completion": {"exit_code": 137, "documents_expected": 1, "documents_written": 0, "all_end_token": True, "failed_markers": 0}}, json.loads(fake.declaration.read_text(encoding="utf-8")), "primary")
    assert verdict["verdict"] in ("FAIL", "INCONSISTENT") and verdict["verdict"] != "PASS" and verdict["last_sampled_substitute"]["memory_peak_mib"] == 8192.0


@needs_bash
def test_a_missing_cgroup_is_recorded_with_its_reason(tmp_path):
    fake = Fake(tmp_path, "no_cgroup")
    fake.run()
    status = fake.status()
    assert status["finality"] == "last_sampled" and status["cgroup"] == "" and "final_metrics_missing" in status["reason"]


@needs_bash
def test_insufficient_host_headroom_means_the_container_never_starts(tmp_path):
    fake = Fake(tmp_path, "ok", available=24576 + 1023)
    done = fake.run()
    assert done.returncode == 4 and "headroom precondition not met" in done.stdout and not [c for c in fake.calls() if c.startswith("docker run")]
    assert "headroom" in fake.status()["reason"]
    ok = Fake(tmp_path / "again", "ok", available=24576 + 1024) if (tmp_path / "again").mkdir() is None else None
    assert ok.run().returncode == 0


@needs_bash
def test_the_downsize_limit_is_used_for_the_smaller_instance_run(tmp_path):
    fake = Fake(tmp_path, "ok", available=20000)
    done = fake.run(key="downsize")
    assert done.returncode == 0 and "--memory 12288m --memory-swap 12288m" in next(c for c in fake.calls() if c.startswith("docker run")) and fake.status()["limit_mib"] == "12288"


def judged(fake):
    run = fake.run_dir()
    run["completion"] = {"exit_code": 0, "documents_expected": 1, "documents_written": 1, "all_end_token": True, "failed_markers": 0}
    return rs.judge_run(run, json.loads(fake.declaration.read_text(encoding="utf-8")), "primary", expect_declaration_sha256=rs.declaration_sha256(fake.declaration))


@needs_bash
def test_the_enforced_limits_are_read_before_the_workload_and_again_in_the_final_read_and_a_normal_run_shows_them_enforced(tmp_path):
    fake = Fake(tmp_path, "ok")
    assert fake.run().returncode == 0
    for name, value in (("memory.max", str(24576 * MIB)), ("memory.swap.max", "0"), ("cpu.max", "400000 100000"), ("cpuset.cpus.effective", "0-3")):
        assert (fake.out / "limits" / "start" / name).read_text(encoding="utf-8").strip() == value and (fake.out / "final" / name).read_text(encoding="utf-8").strip() == value
    assert (fake.out / "limits" / "docker-inspect-start.json").is_file()
    status = fake.status()
    assert status["read_failures"] == "" and "optional_absent" in status
    calls = fake.calls()
    start = next(i for i, c in enumerate(calls) if c.startswith("docker inspect") and "-f" not in c)
    assert start < next(i for i, c in enumerate(calls) if "workload.pid" in c)                                               # the start read comes BEFORE the workload
    result = judged(fake)
    assert result["verdict"] == "PASS" and result["limits_enforced"] is True


@needs_bash
def test_a_cgroup_without_memory_swap_max_is_a_named_read_failure_and_the_run_cannot_pass(tmp_path):
    fake = Fake(tmp_path, "no_swap_file")
    fake.run()
    status = fake.status()
    assert status["read_failures"] == "memory.swap.max" and status["finality"] == "last_sampled" and "final_metrics_missing: required reads failed: memory.swap.max" in status["reason"]
    result = judged(fake)
    assert result["verdict"] == "INCONSISTENT" and any("the no-swap limit is not shown to be enforced" in r for r in result["inconsistent_reasons"])


@needs_bash
def test_a_container_whose_swap_is_not_prohibited_is_inconsistent_by_the_cgroup_and_by_docker(tmp_path):
    bad_cgroup = Fake(tmp_path / "a", "bad_swap") if (tmp_path / "a").mkdir() is None else None
    bad_cgroup.run()
    assert any("swap is not prohibited" in r for r in judged(bad_cgroup)["inconsistent_reasons"])
    bad_docker = Fake(tmp_path / "b", "swap_allowed") if (tmp_path / "b").mkdir() is None else None
    bad_docker.run()
    assert any("MemorySwap must equal Memory" in r for r in judged(bad_docker)["inconsistent_reasons"]) and judged(bad_docker)["verdict"] == "INCONSISTENT"


@needs_bash
def test_the_downsize_run_records_its_own_enforced_limit(tmp_path):
    fake = Fake(tmp_path, "ok", available=20000)
    fake.run(key="downsize")
    assert (fake.out / "limits" / "start" / "memory.max").read_text(encoding="utf-8").strip() == str(12288 * MIB)
    run = fake.run_dir()
    run["completion"] = {"exit_code": 0, "documents_expected": 1, "documents_written": 1, "all_end_token": True, "failed_markers": 0}
    assert rs.judge_run(run, json.loads(fake.declaration.read_text(encoding="utf-8")), "downsize")["limits_enforced"] is True


def test_the_script_reads_the_required_and_optional_files_by_the_agreed_lists():
    body = (EC2 / "measure_resources.sh").read_text(encoding="utf-8")
    assert "for f in memory.peak memory.events memory.max memory.swap.max cpu.max cpuset.cpus.effective; do" in body
    assert "for f in memory.pressure memory.swap.events memory.swap.current; do" in body and "read_failures=$READ_FAILURES" in body and "optional_absent=$OPTIONAL_ABSENT" in body
    assert body.index("limits/start") < body.index("timeout \"$TIMEOUT_S\"")


def test_the_script_structure_trap_order_and_no_auto_remove():
    body = (EC2 / "measure_resources.sh").read_text(encoding="utf-8")
    assert "trap cleanup EXIT INT TERM HUP" in body and "sleep infinity" in body and "--rm" not in body.replace("`--rm`", "")
    cleanup = body[body.index("cleanup() {"):body.index("trap cleanup")]
    assert cleanup.index("collect_final") < cleanup.index('"$DOCKER" rm -f')
    assert body.index("confirm_termination") < body.index("write_status\n}") and "workload.pid" in body and "kill -KILL" in body and "cgroup.procs" in body
    assert body.count('"$DOCKER" rm -f') == 1 and "docker stats" not in body
    assert b"\r" not in (EC2 / "measure_resources.sh").read_bytes()


# ---- record_instance.sh ------------------------------------------------------------------------------------------------------------------------------------------------------------------

@needs_bash
@pytest.mark.parametrize("arch, cgfs, ok", [("x86_64", "cgroup2fs", True), ("aarch64", "cgroup2fs", False), ("x86_64", "tmpfs", False)])
def test_the_instance_record_requires_x86_64_and_cgroup_v2(tmp_path, arch, cgfs, ok):
    fakes = tmp_path / "bin"
    fakes.mkdir()
    for name in ("curl", "nvidia-smi", "docker", "nvidia-ctk"):
        script = fakes / name
        body = "case \"$*\" in *'image inspect'*) echo amd64;; *) echo 'fake " + name + "';; esac\n" if name == "docker" else "echo 'fake " + name + "'\n"
        script.write_text("#!/bin/sh\n" + body, encoding="utf-8", newline="\n")
        script.chmod(0o755)
    env = {**os.environ, "PATH": f"{unix(fakes)}:{os.environ['PATH']}", "UNAME_M": arch, "CGROUP_FS_TYPE": cgfs}
    done = subprocess.run([BASH, unix(EC2 / "record_instance.sh"), unix(tmp_path / "out"), "img:1"], capture_output=True, text=True, env=env, timeout=120)
    record = (tmp_path / "out" / "instance.txt").read_text(encoding="utf-8")
    assert (done.returncode == 0) is ok and f"architecture={arch}" in record and f"cgroup_fs_type={cgfs}" in record
    assert f"required_checks_ok={'yes' if ok else 'NO'}" in record and "image_architecture" in record
    assert "mem_available_mib=" in record or shutil.which("free") is None                                    # `free` is a Linux tool (absent in a Windows Git Bash)
    if not ok:
        assert "NO (" in record


def test_the_instance_record_reads_imds_v2_and_changes_nothing():
    body = (EC2 / "record_instance.sh").read_text(encoding="utf-8")
    assert "X-aws-ec2-metadata-token-ttl-seconds" in body and "api/token" in body and "instance-type" in body and "ami-id" in body
    assert "rm -rf" not in body and "docker run" not in body and "pip install" not in body and b"\r" not in (EC2 / "record_instance.sh").read_bytes()


# ---- REV 4.2 Fix 3: malformed evidence is INCONSISTENT (never an exception), an established failure keeps FAIL, swap events and negative counts are validated ----------------------------------------

def completion_with(**changes):
    return {**good_run()["completion"], **changes}


@pytest.mark.parametrize("value", [[], "", 0, "x", 5, True, [1, 2], {"exit_code": "0"}])
def test_a_completion_record_that_is_not_a_complete_object_is_inconsistent_and_never_an_exception(value):
    """Codex 4: judge_run(completion=[]) raised AttributeError after completion_problems had already named the problem."""
    result = judge(good_run(completion=value))
    assert result["verdict"] == "INCONSISTENT" and any("completion record" in r for r in result["inconsistent_reasons"]) and result["fail_reasons"] == []
    assert rs.judge_t2(value, {"finality": "final_confirmed"})["functional"] == "not_established"


def test_a_malformed_completion_record_does_not_hide_an_independently_established_failure():
    for change in (dict(memory_events={"max": 0, "oom": 1, "oom_kill": 0}), dict(docker={"OOMKilled": True}), dict(status={**good_run()["status"], "workload_exit": "3"}), dict(memory_swap_events={"max": 1, "fail": 0})):
        result = judge(good_run(completion=[], **change))
        assert result["verdict"] == "FAIL" and result["fail_reasons"] and any("completion record" in r for r in result["inconsistent_reasons"]), change


@pytest.mark.parametrize("changes, text", [(dict(failed_markers=-1), "failed_markers is negative"), (dict(documents_written=-1), "documents_written is negative"), (dict(documents_expected=-3), "documents_expected is below 1")])
def test_a_negative_count_in_the_completion_record_is_malformed_not_a_pass(changes, text):
    result = judge(good_run(completion=completion_with(**changes)))
    assert result["verdict"] == "INCONSISTENT" and any(text in r for r in result["inconsistent_reasons"]) and result["fail_reasons"] == []
    assert rs.judge_t2(completion_with(**changes), {"finality": "final_confirmed"})["functional"] == "not_established"


def test_a_negative_count_does_not_hide_a_real_failure():
    result = judge(good_run(completion=completion_with(failed_markers=-1, exit_code=1)))
    assert result["verdict"] == "FAIL" and any("worker exited with 1" in r for r in result["fail_reasons"]) and any("negative" in r for r in result["inconsistent_reasons"])


@pytest.mark.parametrize("code", [-9, -1, 1, 3, 137, 255])
def test_every_non_zero_exit_status_including_a_negative_signal_status_is_a_failure(code):
    result = judge(good_run(completion=completion_with(exit_code=code)))
    assert result["verdict"] == "FAIL" and any(f"worker exited with {code}" in r for r in result["fail_reasons"]) and not any("negative" in r for r in result["inconsistent_reasons"])
    assert rs.judge_t2(completion_with(exit_code=code), {"finality": "final_confirmed"})["functional"] == "fail"


@pytest.mark.parametrize("code", [True, "0", 0.0, None])
def test_an_exit_code_that_is_not_an_integer_is_malformed(code):
    result = judge(good_run(completion=completion_with(exit_code=code)))
    assert result["verdict"] == "INCONSISTENT" and any("exit_code is missing or not an integer" in r for r in result["inconsistent_reasons"])


@pytest.mark.parametrize("events", [{"high": 0, "max": 1, "fail": 0}, {"max": 0, "fail": 1}, {"max": 2, "fail": 2}, {"high": 5, "max": 3, "fail": 0}])
def test_any_positive_swap_events_max_or_fail_fails_the_run(events):
    """Codex 5: memory.swap.events max=1 fail=0 received PASS; the plan says any positive max or fail count is a FAIL."""
    result = judge(good_run(memory_swap_events=events))
    assert result["verdict"] == "FAIL" and any("memory.swap.events" in r for r in result["fail_reasons"])


def test_zero_swap_events_with_a_high_count_are_not_a_failure_and_an_absent_file_is_optional():
    assert judge(good_run(memory_swap_events={"high": 7, "max": 0, "fail": 0}))["verdict"] == "PASS"
    assert judge(good_run(memory_swap_events=None))["verdict"] == "PASS"


@pytest.mark.parametrize("events", [{}, {"fail": 0}, {"max": 0}, {"max": -1, "fail": 0}, {"max": 0, "fail": -2}, {"max": "1", "fail": 0}, {"max": True, "fail": 0}, [], "x", [1]])
def test_a_present_but_unreadable_swap_events_file_is_inconsistent_not_a_pass_and_not_an_exception(events):
    result = judge(good_run(memory_swap_events=events))
    assert result["verdict"] == "INCONSISTENT" and any("memory.swap.events" in r for r in result["inconsistent_reasons"])


def test_a_swap_event_failure_is_not_hidden_by_the_other_counter_being_malformed():
    result = judge(good_run(memory_swap_events={"fail": 4, "max": "?"}))
    assert result["verdict"] == "FAIL" and any("memory.swap.events" in r for r in result["inconsistent_reasons"])


@pytest.mark.parametrize("events", [{"max": 0, "oom": -5, "oom_kill": 0}, {"max": -1, "oom": 0, "oom_kill": 0}, {"max": 0, "oom": 0, "oom_kill": -1}])
def test_a_negative_memory_event_counter_is_malformed_never_a_quiet_zero(events):
    result = judge(good_run(memory_events=events))
    assert result["verdict"] == "INCONSISTENT" and any("non-negative integer" in r for r in result["inconsistent_reasons"])


@pytest.mark.parametrize("changes", [dict(docker=[1]), dict(docker="x"), dict(docker=7), dict(status=[]), dict(status="x"), dict(memory_swap_events=[1]), dict(memory_events=[1]), dict(memory_events="x")])
def test_records_that_are_not_objects_never_raise(changes):
    result = judge(good_run(**changes))
    assert result["verdict"] in ("INCONSISTENT", "FAIL")


def write_run_files(folder, **files):
    folder.mkdir(parents=True)
    for name, text in files.items():
        path = folder / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return folder


@pytest.mark.parametrize("name", ["completion.json", "context.json"])
def test_load_run_never_raises_on_invalid_json_and_the_verdict_names_the_file(tmp_path, name):
    run = rs.load_run(write_run_files(tmp_path / "run1", **{name: "{not json"}))
    assert run["malformed_files"] == [name] and run[name.split(".")[0]] is None
    result = judge({**good_run(), "malformed_files": run["malformed_files"]})
    assert result["verdict"] == "INCONSISTENT" and any(f"{name} is not valid JSON" in r for r in result["inconsistent_reasons"])


@pytest.mark.parametrize("text", ["[]", '"x"', "7", "null"])
def test_a_completion_file_that_is_valid_json_but_not_an_object_is_inconsistent(tmp_path, text):
    run = rs.load_run(write_run_files(tmp_path / "run1", **{"completion.json": text}))
    result = judge({**good_run(), "completion": run["completion"]})
    assert result["verdict"] == "INCONSISTENT" and any("completion" in r for r in result["inconsistent_reasons"])


def test_an_empty_swap_events_file_is_present_and_unreadable_while_a_missing_one_is_optional(tmp_path):
    present = rs.load_run(write_run_files(tmp_path / "a", **{"final__memory.swap.events": ""}))
    absent = rs.load_run(write_run_files(tmp_path / "b", **{"samples.tsv": ""}))
    assert present["memory_swap_events"] == {} and absent["memory_swap_events"] is None
    assert judge(good_run(memory_swap_events=present["memory_swap_events"]))["verdict"] == "INCONSISTENT" and judge(good_run(memory_swap_events=absent["memory_swap_events"]))["verdict"] == "PASS"


def test_a_clean_run_with_zero_swap_events_still_passes_after_the_stricter_validation():
    assert judge(good_run())["verdict"] == "PASS" and judge(good_run(memory_swap_events={"high": 0, "max": 0, "fail": 0}))["verdict"] == "PASS"
