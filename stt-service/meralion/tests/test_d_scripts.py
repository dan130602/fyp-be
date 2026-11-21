"""Stage 3 follow-up D scripts: isolation of the diagnostic mode, the D1 and D2 jobs, clean-commit rules, the windows file and the diagnostic n-gram audit. No cluster is contacted."""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import stub_guard

ORIGAMI =Path(__file__).resolve().parent.parent / "tools" / "origami"
TOOLS = ORIGAMI.parent
sys.path.insert(0, str(TOOLS))


def text(name):
    return (ORIGAMI / name).read_text(encoding="utf-8")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def working_bash():
    bash = shutil.which("bash")
    try:
        if bash and subprocess.run([bash, "-c", "true"], capture_output=True, timeout=20).returncode == 0:
            return bash
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


# ---- the diagnostic mode never leaks into an acceptance run --------------------------------------------------------------------------------------------------------------------

def test_the_single_process_variable_is_set_only_on_the_two_probe_command_lines():
    for path in sorted(ORIGAMI.glob("*.sbatch")) + [ORIGAMI / "common.sh", ORIGAMI / "submit.sh", ORIGAMI / "sync_code.sh"]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "VLLM_ENABLE_V1_MULTIPROCESSING" in line and not line.lstrip().startswith("#"):
                if path.name == "common.sh":                                              # the only mention: run-meta READS the variable to record it; it never sets it
                    assert 'os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "unset")' in line and "export" not in line and "=0" not in line
                    continue
                assert path.name in ("d1_diagnostics.sbatch", "e2_acceptance.sbatch", "bw_matched.sbatch") and "vllm_plugin_probe.py" in line and " env VLLM_ENABLE_V1_MULTIPROCESSING=0 " in f" {line} ", f"{path.name}:{number}: {line}"
                assert "export" not in line
    for name in ("d1_diagnostics.sbatch", "e2_acceptance.sbatch"):
        code = [line for line in text(name).splitlines() if not line.lstrip().startswith("#")]
        assert sum(line.count("VLLM_ENABLE_V1_MULTIPROCESSING=0") for line in code) == 2          # the stock probe and one more probe, each on its own command line (E2: the promoted default-profile probe)


def test_the_two_probe_tools_refuse_the_wrong_mode_before_importing_anything_heavy():
    env = {k: v for k, v in os.environ.items() if k != "VLLM_ENABLE_V1_MULTIPROCESSING"}
    env = {**env, "PYTHONPATH": str(TOOLS.parent)}
    probe = subprocess.run([sys.executable, str(ORIGAMI / "vllm_plugin_probe.py"), "--settings-id", "stock-vllm-package-v1", "--jobs", "j", "--call", "c", "--start", "0", "--end", "1", "--out", "o"],
                           capture_output=True, text=True, env=env)
    assert probe.returncode == 2 and "refuses to run a default-profile probe without VLLM_ENABLE_V1_MULTIPROCESSING=0" in probe.stderr
    profile = subprocess.run([sys.executable, str(ORIGAMI / "vllm_plugin_probe.py"), "--settings-id", "vllm-rp105-sched0-v1", "--jobs", "j", "--call", "c", "--start", "0", "--end", "1", "--out", "o"],
                             capture_output=True, text=True, env={**env, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    assert profile.returncode == 2 and "the decoder sets it itself" in profile.stderr
    windows = subprocess.run([sys.executable, str(ORIGAMI / "vllm_windows.py"), "--settings-id", "x", "--jobs", "j", "--windows", "w", "--out", "o"], capture_output=True, text=True, env={**env, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    assert windows.returncode == 2 and "refuses to run in the diagnostic single-process mode" in windows.stderr


def test_the_acceptance_runs_record_the_mode_and_the_decision_tool_rejects_the_diagnostic_one():
    common = text("common.sh")
    assert '"vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "unset")' in common
    decision_source = (TOOLS / "decision.py").read_text(encoding="utf-8")
    assert "which its execution profile does not allow" in decision_source and "execution_problem" in decision_source and "probe_problem" in decision_source


# ---- the D1 job ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_d1_runs_every_step_as_its_own_process_in_the_planned_order_and_loses_nothing_when_one_fails():
    body = text("d1_diagnostics.sbatch")
    steps = re.findall(r'^step (\w+) ', body, re.M)
    assert steps == ["1a_construct_transformers_stock", "1b_construct_vllm_stock", "2_controlled_logits_test", "3_probe_stock", "4_probe_candidate", "5_windows_stock_normal_mode",
                     "6_windows_candidate_R1_normal_mode", "7_windows_candidate_R2_fresh_engine"]
    assert "set +e" in body and 'echo "$name $rc" >> "$OUT/steps.txt"' in body and 'test -z "$FAILED"' in body
    step_lines = [line for line in body.splitlines() if line.startswith("step ")]
    assert len(step_lines) == len(steps) and all(" --out " in line for line in step_lines)    # each step writes its own result file
    assert 'CANDIDATE=vllm-rp105-v1' in body and "problem_windows.json" in body and "require_clean_commit" in body


def test_d1_decodes_the_runaway_window_in_both_probes_and_repeats_only_the_candidate():
    body = text("d1_diagnostics.sbatch")
    assert body.count("--call CALL_134 --start 24960000 --end 25440000") == 2
    assert "--passes batch,batch,single --package-parity" in body and body.count("--passes") == 3
    assert "windows-candidate-R1.json" in body and "windows-candidate-R2.json" in body and "windows-stock.json" in body


def test_the_problem_windows_file_is_valid_and_covers_the_named_failures():
    data = json.loads((ORIGAMI / "problem_windows.json").read_text(encoding="utf-8"))["windows"]
    assert len(data) == 9 and len({w["label"] for w in data}) == 9
    for window in data:
        assert re.fullmatch(r"CALL_\d{3}", window["call"]) and 0 <= window["start"] < window["end"] and window["end"] - window["start"] <= 480000
    labels = {w["label"]: w for w in data}
    assert (labels["runaway-134"]["start"], labels["runaway-134"]["end"]) == (24960000, 25440000)
    assert (labels["early-stop-217-last"]["start"], labels["early-stop-217-last"]["end"]) == (4560000, 4935680)
    assert {w["call"] for w in data} == {"CALL_064", "CALL_023", "CALL_134", "CALL_217", "CALL_103"}
    assert sum(1 for w in data if w["kind"] == "ordinary") == 2


# ---- the D2 job ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_d2_always_reruns_the_transformers_baseline_in_the_same_allocation_before_the_candidate():
    body = text("compare_candidate.sbatch")
    order = [body.index(marker) for marker in ('transformers "$TF_PILOT"', 'transformers "$TF_C4"', 'vllm "$VL_PILOT"', 'vllm "$VL_C4"')]
    assert order == sorted(order) and 'vllm "$VL_PILOT" "$JOBS/jobs-pilot22.json" --config "$CANDIDATE_CONFIG"' in body
    assert 'TF_PILOT="tf-$GPU_TAG-r2-pilot22-1"' in body and 'VL_PILOT="vllm-$GPU_TAG-rp105-pilot22-1"' in body
    assert body.count("nothing after it runs") == 3 and "refusing to reuse its id" in body and "require_clean_commit" in body and "VLLM_ENABLE_V1_MULTIPROCESSING" not in body.replace("# ", "")
    assert "start_gpu_sampler" not in body                                                    # one sampler PER RUN is started inside run_worker


def test_every_worker_run_has_its_own_sampler_run_meta_and_a_clean_commit_requirement():
    common = text("common.sh")
    body = common[common.index("run_worker()"):]
    assert body.index("require_clean_commit") < body.index("write_run_meta") < body.index("start_gpu_sampler") < body.index("meralion_worker run") < body.index("stop_gpu_sampler")
    assert '"$dir/gpu.csv"' in body and "sleep 3" in body                                      # the first row of each file is the idle sample before the model loads
    for key in ("worker_git_commit", "slurm_job_id", "gpu_uuid", "pip_freeze_sha256", "worker_arguments"):
        assert key in common


def test_the_new_jobs_send_the_gpu_type_and_time_through_submit_sh_only():
    for name in ("d1_diagnostics.sbatch", "compare_candidate.sbatch", "e1_profiles.sbatch", "e2_acceptance.sbatch", "dev175_validation.sbatch"):
        body = text(name)
        assert "submit.sh" in body[:900] and ': "${GPU_TAG:?' in body
        for line in body.splitlines():
            if line.startswith("#SBATCH"):
                assert "$" not in line
        assert "#SBATCH --time=00:10:00" in body                                               # a direct sbatch cannot spend real GPU time


@pytest.mark.skipif(working_bash() is None, reason="no working bash")
@pytest.mark.parametrize("commit, ok", [("0123456789abcdef", True), ("unknown", False), ("", False), ("0123456789abcdef-dirty", False)])
def test_require_clean_commit_refuses_unknown_empty_and_dirty_commits(commit, ok):
    common = text("common.sh")
    helper = common[common.index("require_clean_commit()"):common.index("# write_run_meta")]
    result = subprocess.run([working_bash(), "-c", f'MERALION_WORKER_GIT_COMMIT="{commit}"; {helper}\nrequire_clean_commit'], capture_output=True, text=True)
    assert (result.returncode == 0) is ok


# ---- sync_code.sh: a dirty tree is refused, the commit is recorded ----------------------------------------------------------------------------------------------------------

def make_repo(tmp_path):
    bash = working_bash()
    repo = tmp_path / "repo"
    module = repo / "stt-service" / "meralion"
    for directory in ("meralion_worker/vendored", "configs", "tools/origami", "patches"):
        (module / directory).mkdir(parents=True)
    for name in ("patches/manifest.json", "patches/vllm-0.16.0-mig-single-device.patch", "tools/apply_patches.py"):          # carried and hash-compared by sync_code.sh (REV 3.0 A3)
        (module / name).write_text("fixture " + name, encoding="utf-8")
    (module / "meralion_worker" / "vendored" / "SOURCES.json").write_text("{}", encoding="utf-8")
    (module / "tools" / "make_jobs.py").write_text("x", encoding="utf-8")
    shutil.copy(ORIGAMI / "sync_code.sh", module / "tools" / "origami" / "sync_code.sh")
    (repo / "stt-experiments" / "meralion-runner").mkdir(parents=True)
    (repo / "stt-experiments" / "meralion-runner" / "pilot_calls_overlap_v1.json").write_text("{}", encoding="utf-8")
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    log = tmp_path / "calls.log"
    answer = ""
    # a remote `sha256sum '<code folder>/<file>'` is answered with the hash of the same file in the local module, as if the sync had worked
    answer_ssh = 'all="$*"\ncase "$all" in *sha256sum*/code/*) rel="${all#*/code/}"; rel="${rel%%"\'"*}"; sha256sum "$FAKE_MODULE/$rel" | cut -d" " -f1;; esac\n'
    for tool in ("ssh", "scp", "tar"):
        script = fakes / tool
        answer = answer_ssh if tool == "ssh" else ""
        script.write_text(stub_guard.stub_script(tool, log).replace("exit 0\n", answer + "exit 0\n"), encoding="utf-8", newline="\n")
        script.chmod(0o755)
    for command in (["git", "init", "-q"], ["git", "config", "user.email", "t@example.com"], ["git", "config", "user.name", "t"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    return bash, repo, module, fakes, log


@pytest.mark.skipif(working_bash() is None, reason="no working bash")
def test_sync_code_refuses_a_dirty_tree_and_records_the_clean_commit(tmp_path):
    bash, repo, module, fakes, log = make_repo(tmp_path)

    def sync(**extra):
        return sync_run(bash, fakes, module, **extra)

    clean = sync()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert clean.returncode == 0 and "CODE_COMMIT" in log.read_text(encoding="utf-8")
    log.write_text("", encoding="utf-8")
    (module / "meralion_worker" / "new_file.py").write_text("uncommitted", encoding="utf-8")
    dirty = sync()
    assert dirty.returncode == 4 and "refusing to sync" in dirty.stdout and log.read_text(encoding="utf-8") == ""          # nothing was sent
    allowed = sync(ALLOW_DIRTY="1")
    assert allowed.returncode == 0 and "not usable for an acceptance run" in allowed.stdout and head


# ---- the diagnostic n-gram audit is never a gate -----------------------------------------------------------------------------------------------------------------------------

def test_the_audit_helper_reproduces_codexs_counterexample_and_is_labelled_diagnostic():
    audit = load_module(TOOLS / "ngram_audit.py", "ngram_audit")
    ids = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 7]
    assert audit.has_repeat(ids, 5) is True and audit.has_repeat(ids, 6) is False                  # a repeated 5-gram does NOT imply a repeated 6-gram
    result = audit.audit({"a": "x", "b": "y"}, lambda t: ids if t == "x" else [9, 8, 7])
    assert result["windows"] == 2 and result["windows_with_a_repeated_ngram"][5] == 1 and result["windows_with_a_repeated_6gram"] == []
    source = (TOOLS / "ngram_audit.py").read_text(encoding="utf-8")
    assert "NEVER an acceptance gate" in source and "never required to contain repeated 5-grams" in source


def test_neither_the_decision_tool_nor_the_comparison_uses_the_audit_as_a_gate():
    for name in ("decision.py", "compare_backends.py", "live_parity.py"):
        body = (TOOLS / name).read_text(encoding="utf-8")
        assert "ngram_ban_effective" not in body and "ngram_audit" not in body


def test_the_controlled_logits_test_covers_the_cases_the_plan_names():
    body = text("plugin_logits_test.py")
    for fragment in ("[10, 11, 12, 13, 14, 15, 99, 10, 11, 12, 13, 14]", "logits[0, 15]", "bans_the_token_that_would_complete_a_repeated_6gram", "placeholder_history_does_not_ban_the_real_token_15",
                     "the_output_list_is_a_live_reference", "a_slot_swap_moves_the_history_with_it", "ngram_size_3_from_the_environment_bans_7", "is_argmax_invariant_is_false", "the_default_ngram_size_is_6_when_unset"):
        assert fragment in body or fragment.replace("logits[0, 15]", "== [15]") in body, fragment


# ---- follow-up E: the E1 and E2 jobs ------------------------------------------------------------------------------------------------------------------------------------------------

def command_lines(body):
    return [line for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_e1_runs_two_fresh_engines_per_arm_round_robin_each_with_its_own_config_and_never_sets_the_scheduling_variable():
    body = text("e1_profiles.sbatch")
    assert "for n in 1 2; do" + chr(10) + "  for arm in A B C D; do" in body                                  # A1 B1 C1 D1 A2 B2 C2 D2
    assert 'step "${arm}${n}_worker_run" run_worker "$VENV_VLLM/bin/python" vllm "vllm-$GPU_TAG-e1-$arm-$n" "$JOBS3" --config "${CONFIG[$arm]}"' in body
    for arm, suffix in (("A", "vllm-rp105"), ("B", "vllm-rp105-sched0"), ("C", "vllm-stock"), ("D", "vllm-stock-sched0")):
        assert f"CONFIG_{arm}=$RUNTIME/$STEM-{suffix}.json" in body
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in chr(10).join(command_lines(body)) and "export VLLM" not in body
    assert body.count("run_worker ") == 1 and body.count("vllm_windows.py") == 1 and "--settings-id vllm-rp105-sched0-v1 --jobs \"$JOBS_PILOT\"" in body and "--package-parity" not in body


def test_e1_uses_the_three_complete_recordings_in_the_fixed_order_and_declares_the_experiment_before_any_run():
    body = text("e1_profiles.sbatch")
    assert 'wanted = ["CALL_134", "CALL_103", "CALL_217"]' in body and "jobs-e1-3rec.json" in body
    first_run = body.index('step "${arm}${n}_worker_run"')
    assert body.index("already finished: refusing to reuse its id") < body.index("e1-manifest.json") < first_run
    assert "require_clean_commit" in body and "e1-manifest-v1" in body and "settings_id" in body and "set +e" in body and 'test -z "$FAILED"' in body


def test_e2_runs_the_structural_evidence_first_then_the_baseline_then_the_candidate_twice_then_the_optional_control():
    body = text("e2_acceptance.sbatch")
    order = [body.index(marker) for marker in ("need 1_controlled_logits_test", "need 2_probe_stock_default", "4_capture_promoted", "need 5_tf_pilot", "need 6_tf_c4", "need 7_candidate_pilot_r1",
                                                "need 8_candidate_c4", "need 9_candidate_pilot_r2", "10_stock_control_pilot")]
    assert order == sorted(order) and body.index("e2-manifest.json") < order[0] and body.index("already finished: refusing to reuse its id") < body.index("e2-manifest.json")
    assert 'echo "required step $name failed (exit $rc): nothing after it runs"; exit "$rc"; fi' in body and "nothing after it runs" in body and 'if [ "$LEFT" -ge 180 ]' in body and "need 10" not in body                                       # the control is optional and never decides
    assert body.count('--config "$CANDIDATE_CONFIG"') == 3 and "$CONTROL_CONFIG" in body and 'transformers "$TF_PILOT" "$JOBS_PILOT"' in body and 'transformers "$TF_C4" "$JOBS_C4"' in body
    for fragment in ("--capture", "--settings-id stock-vllm-package-v1 --jobs", "--settings-id \"$PROMOTED_SETTINGS\" --jobs", "e2-manifest-v1", "candidate_invocation_ids"):
        assert fragment in body


def test_both_new_jobs_refuse_a_reused_run_id_with_a_nonzero_exit():
    for name in ("e1_profiles.sbatch", "e2_acceptance.sbatch", "dev175_validation.sbatch"):
        assert 'refusing to reuse its id"; exit 3; fi' in text(name)


def test_e2_takes_the_promoted_settings_through_submit_sh_and_checks_the_config_names_them():
    body = text("e2_acceptance.sbatch")
    assert ': "${PROMOTED_SETTINGS:?' in body and ': "${PROMOTED_CONFIG:?' in body and "the config names settings" in body and "frozen.execution_of(wanted)" in body
    assert "PROMOTED_SETTINGS=<settings id> PROMOTED_CONFIG=" in body[:1400]
    for name in ("e1_profiles.sbatch", "e2_acceptance.sbatch"):
        assert 'step "' in text(name) or "need " in text(name)
        assert "#SBATCH --constraint=rtx_pro_6000_blackwell_max" in text(name)                    # only a default: submit.sh puts the real type on the command line


def test_the_scheduling_variable_is_set_in_python_only_by_the_decoder_and_read_by_the_tools():
    package = TOOLS.parent / "meralion_worker"
    setters = [path.name for path in package.rglob("*.py") if "os.environ[SCHEDULING_VARIABLE]" in path.read_text(encoding="utf-8")]
    assert setters == ["vllm_decoder.py"]
    for path in list(TOOLS.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "os.environ[\"VLLM_ENABLE_V1_MULTIPROCESSING\"] =" not in source and "os.environ.setdefault(\"VLLM_ENABLE_V1_MULTIPROCESSING\"" not in source, path.name


# ---- the full development validation job (one fixed configuration, two fresh-engine repetitions) ------------------------------------------------------------------------------------

def test_the_full_validation_is_fixed_to_one_configuration_and_runs_two_fresh_engines_in_order_after_declaring_the_experiment():
    body = text("dev175_validation.sbatch")
    assert "SETTINGS_ID=vllm-rp105-sched0-v1" in body and body.count("--config \"$CONFIG\"") == 2 and "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json" in body
    assert "the config names settings" in body and "exactly 175 distinct recordings" in body
    order = [body.index(marker) for marker in ("already finished: refusing to reuse its id", "full175-manifest-v1", "need 1_full175_r1", "need 2_full175_r2")]
    assert order == sorted(order)
    assert "transformers" not in body.replace("no Transformers run", "") and "tf-" not in body and "stockctl" not in body and "PROMOTED_" not in body
    assert body.count("run_worker ") == 2 and '"$R1"' in body and '"$R2"' in body and 'R1="vllm-$GPU_TAG-full175-r1"' in body and 'R2="vllm-$GPU_TAG-full175-r2"' in body
    assert "need() {" in body and 'exit "$rc"; fi' in body                                       # a failed first repetition stops the job: it is never repeated here
    for key in ("BASELINE_SOURCE_SHA256", "BASELINE_DOCUMENTS_SHA256", "config_fingerprint", "settings_digest", "frozen-historical", "gpu_uuid", "candidate_invocation_ids"):
        assert key in body


def test_the_full_validation_never_sets_an_engine_variable_and_the_budget_is_the_approved_one():
    body = text("dev175_validation.sbatch")
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in body and "VLLM_ATTENTION_BACKEND" not in body and "export VLLM" not in body
    assert "00:28:00" in body and "GPU_CAP_HOURS=5.4" in body


@pytest.mark.skipif(working_bash() is None, reason="no working bash")
def test_sync_code_stops_with_exit_5_when_a_patch_file_differs_between_the_laptop_and_the_cluster(tmp_path):
    bash, repo, module, fakes, log = make_repo(tmp_path)
    other = tmp_path / "other-side"
    shutil.copytree(module, other)
    (other / "patches" / "vllm-0.16.0-mig-single-device.patch").write_text("a different patch", encoding="utf-8")        # what the cluster would hold after a corrupted transfer
    done = sync_run(bash, fakes, module, FAKE_MODULE=other.as_posix())
    assert done.returncode == 5 and "MISMATCH: patches/vllm-0.16.0-mig-single-device.patch differs" in done.stdout


# ---- REV 4.1 A1: what is shipped must be byte-identical to the committed blobs -------------------------------------------------------------------------------------------------------

def sync_run(bash, fakes, module, **extra):
    """sync_code.sh against the stubs, with a host that can never resolve; the stubs are verified to resolve first (tests/stub_guard.py) and the run FAILS closed otherwise."""
    extra_env = {"FAKE_MODULE": module.as_posix(), **extra}
    return stub_guard.run_stubbed(bash, fakes, ["bash", "tools/origami/sync_code.sh", stub_guard.TEST_HOST], cwd=module, extra_env=extra_env)


@pytest.mark.skipif(working_bash() is None, reason="no working bash")
def test_sync_code_refuses_a_working_copy_whose_bytes_differ_from_the_committed_blob_before_anything_is_sent(tmp_path):
    """A CRLF working copy (core.autocrlf=true on a Windows checkout) is 'clean' for git status but is not the committed file: the tool hash recorded on the cluster would not match the blob."""
    bash, repo, module, fakes, log = make_repo(tmp_path)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repo, check=True, capture_output=True)
    target = module / "tools" / "env_identity.py"
    target.write_bytes(b"line one" + bytes([10]) + b"line two" + bytes([10]))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "lf"], cwd=repo, check=True, capture_output=True)
    ok = sync_run(bash, fakes, module)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    log.write_text("", encoding="utf-8")
    target.unlink()
    subprocess.run(["git", "checkout", "--", "stt-service/meralion/tools/env_identity.py"], cwd=repo, check=True, capture_output=True)     # git itself writes the CRLF copy, as a fresh Windows checkout does
    assert target.read_bytes() == b"line one" + bytes([13, 10]) + b"line two" + bytes([13, 10])
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=module, capture_output=True, text=True).stdout
    assert status.strip() == ""                                                                  # git itself calls the tree clean: only the byte check can see the difference
    refused = sync_run(bash, fakes, module)
    assert refused.returncode == 6 and "not the committed bytes: tools/env_identity.py" in refused.stdout and "refusing to sync" in refused.stdout
    assert log.read_text(encoding="utf-8") == ""                                                 # nothing was sent


@pytest.mark.skipif(working_bash() is None, reason="no working bash")
def test_the_byte_identity_guard_is_not_applied_to_a_dirty_diagnostic_sync(tmp_path):
    bash, repo, module, fakes, log = make_repo(tmp_path)
    (module / "meralion_worker" / "new_file.py").write_text("uncommitted", encoding="utf-8")
    allowed = sync_run(bash, fakes, module, ALLOW_DIRTY="1")
    assert allowed.returncode == 0 and "not the committed bytes" not in allowed.stdout


# ---- REV 4.2 Fix 4: a sync test can never reach the production alias; the stubs are verified, and the run fails closed ------------------------------------------------------------

@pytest.mark.skipif(working_bash() is None, reason="no working bash")
def test_every_call_of_a_sync_test_reached_a_stub_with_the_invalid_host_and_none_reached_the_production_alias(tmp_path):
    bash, repo, module, fakes, log = make_repo(tmp_path)
    done = sync_run(bash, fakes, module)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = log.read_text(encoding="utf-8")
    stub_guard.assert_only_stubs_were_called(calls)
    assert "stub:ssh -o BatchMode=yes sync-test.invalid" in calls and "origami-fyp" not in calls


@pytest.mark.skipif(working_bash() is None, reason="no working bash")
def test_the_guard_fails_closed_when_the_stubs_do_not_resolve_and_never_starts_the_script(tmp_path):
    bash, repo, module, fakes, log = make_repo(tmp_path)
    empty = tmp_path / "empty-stubs"
    empty.mkdir()
    with pytest.raises(pytest.fail.Exception, match="stub resolution failed"):
        stub_guard.run_stubbed(bash, empty, ["bash", "tools/origami/sync_code.sh", stub_guard.TEST_HOST], cwd=module, extra_env={"FAKE_MODULE": module.as_posix()})
    partial = tmp_path / "partial-stubs"                                                         # one tool missing from the stub directory is just as fatal
    partial.mkdir()
    (partial / "ssh").write_text("#!/bin/sh\nexit 0\n", newline="\n")
    (partial / "ssh").chmod(0o755)
    with pytest.raises(pytest.fail.Exception, match="stub resolution failed"):
        stub_guard.run_stubbed(bash, partial, ["bash", "tools/origami/sync_code.sh", stub_guard.TEST_HOST], cwd=module)
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


def test_the_guard_refuses_the_production_alias_and_a_missing_host():
    with pytest.raises(pytest.fail.Exception, match="production host alias"):
        stub_guard.run_stubbed("bash", Path("."), ["bash", "tools/origami/sync_code.sh", "origami-fyp"])
    with pytest.raises(pytest.fail.Exception, match="explicit, never-resolving host"):
        stub_guard.run_stubbed("bash", Path("."), ["bash", "tools/origami/sync_code.sh"])
    with pytest.raises(pytest.fail.Exception, match="explicit, never-resolving host"):
        stub_guard.run_stubbed("bash", Path("."), ["bash", "tools/origami/sync_code.sh", "some-real-host"])


def test_the_call_log_check_rejects_a_real_call_and_the_production_alias():
    stub_guard.assert_only_stubs_were_called("stub:ssh -o BatchMode=yes sync-test.invalid true\nstub:tar -xzf -")
    with pytest.raises(AssertionError):
        stub_guard.assert_only_stubs_were_called("ssh other-host true")                                  # not a stub line
    with pytest.raises(AssertionError):
        stub_guard.assert_only_stubs_were_called("stub:ssh -o BatchMode=yes origami-fyp true")              # a stub that was addressed to the production alias


def test_no_test_runs_sync_code_directly_or_by_the_default_host():
    """Every test that executes sync_code.sh goes through stub_guard.run_stubbed (no raw subprocess call of the script anywhere in the tests)."""
    offenders = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "sync_code.sh" in line and "subprocess" in line and "run_stubbed" not in line:
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


def test_sync_code_checks_blob_identity_before_it_touches_the_cluster():
    body = (ORIGAMI / "sync_code.sh").read_text(encoding="utf-8")
    guard, remove = body.index("git hash-object --no-filters"), body.index("rm -rf '$REMOTE'")
    assert guard < remove and "git ls-tree -r HEAD" in body and "exit 6" in body


def test_the_generic_line_ending_rules_come_first_so_the_byte_preserving_exceptions_keep_precedence(tmp_path):
    attributes = (ORIGAMI.parent.parent / ".gitattributes").read_text(encoding="utf-8").splitlines()
    rules = [line for line in attributes if line.strip() and not line.startswith("#")]
    assert rules[:2] == ["*.py text eol=lf", "*.json text eol=lf"]                               # last matching line wins: the exceptions below must come after
    assert rules.index("meralion_worker/vendored/*.py -text") > 1 and rules.index("patches/*.patch -text") > 1
    if shutil.which("git") is None:
        pytest.skip("no git")
    repo = tmp_path / "attrs"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    (repo / ".gitattributes").write_bytes((chr(10).join(attributes) + chr(10)).encode("utf-8"))

    def attributes_of(path):
        out = subprocess.run(["git", "check-attr", "text", "eol", "--", path], cwd=repo, capture_output=True, text=True, check=True).stdout.splitlines()
        return {line.split(":")[1].strip(): line.split(":", 2)[2].strip() for line in out}
    assert attributes_of("tools/decision.py") == {"text": "set", "eol": "lf"}
    assert attributes_of("configs/runtime/x.json") == {"text": "set", "eol": "lf"}
    assert attributes_of("meralion_worker/vendored/overlap.py")["text"] == "unset"                 # byte-preserved: the generic rule did not override the exception
    assert attributes_of("patches/vllm-0.16.0-mig-single-device.patch")["text"] == "unset"
    assert attributes_of("patches/manifest.json") == {"text": "set", "eol": "lf"}
    assert attributes_of("meralion_worker/vendored/SOURCES.json") == {"text": "set", "eol": "lf"}
