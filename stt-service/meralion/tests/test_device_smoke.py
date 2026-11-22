"""tools/origami/device_smoke.py (REV 3.0 A1, REV 3.1 Amendments 1 and 3): the order of initialisation, the named checks, and the classification of outcomes.

vLLM, torch and the MERaLiON package are replaced by small stand-ins written into a temporary folder, so the tests can watch WHEN vLLM is first imported and with which value of
VLLM_ENABLE_V1_MULTIPROCESSING. Nothing here proves anything about a real GPU.
"""
import ast
import importlib.util
import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from meralion_worker import audio as worker_audio
from meralion_worker.decoders import frozen
from meralion_worker.decoders.base import DecodingSettingsError

from vllm_fakes import FakeBackend, completion

ORIGAMI = Path(__file__).resolve().parent.parent / "tools" / "origami"
SOURCE = (ORIGAMI / "device_smoke.py").read_text(encoding="utf-8")
SCHED0, FP16, STOCK = "vllm-rp105-sched0-v1", "vllm-rp105-sched0-fp16-v1", "stock-vllm-package-v1"
VARIABLE = "VLLM_ENABLE_V1_MULTIPROCESSING"
MIG_LIST = "GPU 0: NVIDIA RTX PRO 6000 Blackwell (UUID: GPU-11111111-aaaa)\n  MIG 1g.24gb     Device  0: (UUID: MIG-22222222-bbbb)"


def load():
    spec = importlib.util.spec_from_file_location("device_smoke_under_test", ORIGAMI / "device_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load()


# ---- stand-ins -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InprocClient:
    pass


class SyncMPClient:
    pass


class Dtype:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def engine_config(dtype="torch.float16"):
    ns = types.SimpleNamespace
    return ns(model_config=ns(dtype=Dtype(dtype), max_model_len=1300), scheduler_config=ns(max_num_seqs=64), cache_config=ns(gpu_memory_utilization=0.85, num_gpu_blocks=10, block_size=16),
              parallel_config=ns(tensor_parallel_size=1), attention_config=ns(backend=ns(name="FLASHINFER")))


NEVER = 'never imported'


class Recorder:
    """Reads what the fake vllm package recorded when it was ACTUALLY imported (a spec lookup by importlib.util.find_spec is not an import)."""

    @property
    def first_value(self):
        return __import__('builtins')._vllm_first_value

    @property
    def names(self):
        return list(__import__('builtins')._vllm_imports)



class Setup:
    """A fake world: vllm (a real, tiny package on sys.path whose platform class name is configurable), torch, meralion_3_asr and an audio loader."""

    def __init__(self, tmp_path, monkeypatch, *, platform_class="NonNvmlCudaPlatform", dtype="torch.float16", client=InprocClient, completions=4, text="hello"):
        self.tmp, self.monkeypatch = tmp_path, monkeypatch
        root = tmp_path / "fakes"
        (root / "vllm" / "platforms").mkdir(parents=True)
        recorder_code = "import builtins, os" + chr(10) + "if builtins._vllm_first_value == 'never imported':" + chr(10) + "    builtins._vllm_first_value = os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')" + chr(10) + "builtins._vllm_imports.append(__name__)" + chr(10)
        (root / "vllm" / "__init__.py").write_text(recorder_code, encoding="utf-8")
        (root / "vllm" / "platforms" / "__init__.py").write_text(recorder_code + f"class {platform_class}:" + chr(10) + "    pass" + chr(10) + f"current_platform = {platform_class}()" + chr(10), encoding="utf-8")
        import builtins
        monkeypatch.setattr(builtins, "_vllm_first_value", NEVER, raising=False)
        monkeypatch.setattr(builtins, "_vllm_imports", [], raising=False)
        monkeypatch.syspath_prepend(str(root))
        for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
            monkeypatch.delitem(sys.modules, name)
        self.recorder = Recorder()
        self.backend = FakeBackend([completion(text) for _ in range(completions)])
        self.backend.llm.llm_engine = types.SimpleNamespace(engine_core=client(), vllm_config=engine_config(dtype))
        self.from_pretrained_calls = []
        monkeypatch.setitem(sys.modules, "torch", self.torch())
        self.package()
        monkeypatch.setattr(worker_audio, "load_recording", lambda call, path: types.SimpleNamespace(num_samples=16000 * 130, get=lambda a, b: __import__("numpy").zeros(b - a, dtype="float32")))
        monkeypatch.setattr(ds.subprocess, "run", self.smi)
        self.jobs = tmp_path / "jobs.json"
        self.jobs.write_text(json.dumps({"jobs": [{"job_name": "CALL_064", "input": "x.wav"}]}), encoding="utf-8")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-22222222-bbbb")
        monkeypatch.setenv("MERALION_NGRAM_SIZE", "6")

    @staticmethod
    def smi(command, *args, **kwargs):
        text = MIG_LIST if "-L" in command else "NVIDIA RTX PRO 6000 Blackwell, 97887 MiB, 610.57.04, 12.0"
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    @staticmethod
    def torch():
        ns = types.SimpleNamespace
        module = types.ModuleType("torch")
        module.__version__ = "2.9.1+cu128"
        module.version = ns(cuda="12.8")
        module.cuda = ns(get_device_capability=lambda i: (12, 0), get_arch_list=lambda: ["sm_90", "sm_120"], is_available=lambda: True, get_device_name=lambda i: "Fake Blackwell slice",
                         get_device_properties=lambda i: ns(name="Fake Blackwell slice", total_memory=24 * 2 ** 30, multi_processor_count=47))
        return module

    def package(self):
        calls, backend = self.from_pretrained_calls, self.backend

        def from_pretrained(path, backend_name=None, **kwargs):
            calls.append(kwargs)
            import vllm.platforms                                                     # the real package imports vLLM here
            return types.SimpleNamespace(_backend=backend)
        package = types.ModuleType("meralion_3_asr")
        package.Meralion3ASR = types.SimpleNamespace(from_pretrained=lambda path, backend=None, **kw: from_pretrained(path, backend, **kw))
        backends = types.ModuleType("meralion_3_asr.backends")
        vllm_backend = types.ModuleType("meralion_3_asr.backends.vllm_backend")

        class VllmBackend:
            def __init__(self, model, gpu_memory_utilization=0.85, max_model_len=1300, max_num_seqs=64, tensor_parallel_size=1, dtype="bfloat16", attention_backend="FLASHINFER"):
                pass
        vllm_backend.VllmBackend = VllmBackend
        prompts = types.ModuleType("meralion_3_asr.prompts")
        prompts.VLLM_SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 1.0, "top_k": 50, "max_tokens": 512, "repetition_penalty": 1.0}
        for name, module in (("meralion_3_asr", package), ("meralion_3_asr.backends", backends), ("meralion_3_asr.backends.vllm_backend", vllm_backend), ("meralion_3_asr.prompts", prompts)):
            self.monkeypatch.setitem(sys.modules, name, module)

    def args(self, settings_id, **over):
        values = dict(settings_id=settings_id, model_dir="model", jobs=str(self.jobs), call="CALL_064", windows=4, out=str(self.tmp / "out.json"), expect_capability=None, expect_platform=None,
                      require_no_empty_text=False)
        values.update(over)
        return types.SimpleNamespace(**values)


# ---- (a) the order of initialisation ---------------------------------------------------------------------------------------------------------------------------------------------

def test_vllm_is_first_imported_with_the_scheduling_variable_already_set_by_the_decoder_for_a_scheduling_profile(tmp_path, monkeypatch):
    setup = Setup(tmp_path, monkeypatch)
    assert ds.run(setup.args(SCHED0)) == 0
    assert setup.recorder.first_value == "0" and setup.recorder.names[0] == "vllm"
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["post_facts"]["platform_class"] == "NonNvmlCudaPlatform" and report["pre_facts"]["mig_device_visible"] is True


def test_a_default_profile_imports_vllm_with_the_variable_unset(tmp_path, monkeypatch):
    setup = Setup(tmp_path, monkeypatch, platform_class="NvmlCudaPlatform", client=SyncMPClient)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert ds.run(setup.args(STOCK)) == 0
    assert setup.recorder.first_value is None


def test_looking_at_vllm_before_the_decoder_exists_is_what_the_rev_3_0_order_did_and_the_test_detects_it(tmp_path, monkeypatch):
    setup = Setup(tmp_path, monkeypatch)
    real_pre_facts = ds.pre_facts

    def buggy_pre_facts(*args, **kwargs):                                                  # the REV 3.0 order: the platform is inspected first
        import vllm.platforms
        return real_pre_facts(*args, **kwargs)
    monkeypatch.setattr(ds, "pre_facts", buggy_pre_facts)
    ds.run(setup.args(SCHED0))
    assert setup.recorder.first_value is None and setup.recorder.first_value != "0"       # vLLM would have chosen its platform with the variable unset: the NVML class on a MIG slice


def test_post_facts_before_build_decoder_fails_loudly_and_imports_no_vllm(tmp_path, monkeypatch):
    setup = Setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="vLLM was reached before the decoder set it"):
        ds.post_facts(types.SimpleNamespace(describe=lambda: {"engine_effective": None}), SCHED0)
    assert setup.recorder.names == []


def test_a_conflicting_launcher_value_is_refused_and_vllm_is_not_imported(tmp_path, monkeypatch):
    setup = Setup(tmp_path, monkeypatch)
    monkeypatch.setenv(VARIABLE, "1")
    with pytest.raises(DecodingSettingsError, match="themselves"):
        ds.run(setup.args(SCHED0))
    assert setup.recorder.names == []


def test_assert_profile_environment_truth_table():
    ds.assert_profile_environment(SCHED0, {VARIABLE: "0"})
    ds.assert_profile_environment(FP16, {VARIABLE: "0"})
    ds.assert_profile_environment(STOCK, {})
    ds.assert_profile_environment(STOCK, {VARIABLE: "1"})
    for settings_id, env in ((SCHED0, {}), (SCHED0, {VARIABLE: "1"}), (STOCK, {VARIABLE: "0"})):
        with pytest.raises(RuntimeError):
            ds.assert_profile_environment(settings_id, env)


def test_the_settings_id_is_required():
    with pytest.raises(SystemExit) as stopped:
        ds.main(["--model-dir", "m", "--jobs", "j", "--call", "c", "--out", "o"])
    assert stopped.value.code == 2


def test_the_fp16_profile_passes_the_package_constructor_its_one_override_and_its_engine_is_checked(tmp_path, monkeypatch):
    setup = Setup(tmp_path, monkeypatch, dtype="torch.float16")
    assert ds.run(setup.args(FP16, require_no_empty_text=True, expect_capability="12.0")) == 0
    assert setup.from_pretrained_calls == [{"dtype": "float16"}]
    bf16 = Setup(tmp_path / "b", monkeypatch, dtype="torch.bfloat16")
    with pytest.raises(DecodingSettingsError, match="dtype"):
        ds.run(bf16.args(FP16))


# ---- (d) static structure --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def function(name):
    tree = ast.parse(SOURCE)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_no_module_level_vllm_or_torch_import_and_pre_facts_touches_neither():
    tree = ast.parse(SOURCE)
    names = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names} | {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert not names & {"torch", "vllm", "meralion_3_asr"}
    body = ast.get_source_segment(SOURCE, function("pre_facts"))
    assert "torch" not in body and "vllm" not in body and "current_platform" not in body
    assert "import torch" not in ast.get_source_segment(SOURCE, function("build_decoder"))


def test_in_run_the_order_is_validate_then_pre_facts_then_build_decoder_then_post_facts():
    body = ast.get_source_segment(SOURCE, function("run"))
    order = [body.index(marker) for marker in ("frozen.resolve_settings_id", "pre_facts()", "build_decoder(", "post_facts(")]
    assert order == sorted(order)
    post = ast.get_source_segment(SOURCE, function("post_facts"))
    statements = function("post_facts").body
    assert "assert_profile_environment" in ast.get_source_segment(SOURCE, statements[1]) or "assert_profile_environment" in ast.get_source_segment(SOURCE, statements[0])
    assert post.index("assert_profile_environment") < post.index("import torch") < post.index("import vllm.platforms")


@pytest.mark.parametrize("name", ["vllm_windows.py", "vllm_plugin_probe.py", "device_smoke.py"])
def test_the_other_tools_that_build_a_decoder_import_vllm_only_inside_functions(name):
    tree = ast.parse((ORIGAMI / name).read_text(encoding="utf-8"))
    names = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names} | {(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert not names & {"torch", "vllm", "meralion_3_asr"}, name


def test_importing_the_tool_loads_neither_torch_nor_vllm():
    code = f"import sys; sys.path.insert(0, r'{ORIGAMI}'); import device_smoke; bad=[m for m in ('torch','vllm','meralion_3_asr') if m in sys.modules]; assert not bad, bad"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ORIGAMI.parent.parent))
    assert done.returncode == 0, done.stderr


def test_the_old_smoke_tool_says_it_is_historical_and_stock_only():
    assert (ORIGAMI / "vllm_smoke.py").read_text(encoding="utf-8").startswith('"""HISTORICAL: stock settings only; NOT used to validate profiles')


# ---- the named checks ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def good_parts(settings_id=SCHED0):
    describe = {"frozen_digest": frozen.frozen_digest(settings_id), "settings_id": settings_id, "settings": {"observed": {}}}
    mode = "profile-in-process" if frozen.execution_of(settings_id) else "default"
    post = {"capability": [12, 0], "arch_list": ["sm_90", "sm_120"], "platform_class": "NonNvmlCudaPlatform", "engine_effective": frozen.effective_engine_expectation(settings_id)}
    return dict(settings_id=settings_id, describe=describe, execution={"mode": mode}, pre={"mig_device_visible": True}, post=post, rows=[{"stop_reason": "end_token", "text": "x"}])


def checks_of(monkeypatch, parts, **options):
    monkeypatch.setattr(frozen, "check_frozen", lambda settings_id, observed: None)               # the frozen-settings check has its own tests; here it is a named check of the plan
    return ds.plan_checks(**parts, **options)


def test_every_check_holds_for_a_consistent_run(monkeypatch):
    assert all(checks_of(monkeypatch, good_parts()).values())
    assert all(checks_of(monkeypatch, good_parts(FP16), expect_capability=(12, 0), require_no_empty_text=True).values())


@pytest.mark.parametrize("mutation, failing", [
    (lambda p: p["describe"].update(frozen_digest="0" * 64), "profile_digest_is_the_frozen_digest"),
    (lambda p: p["describe"].update(settings_id=STOCK), "profile_digest_is_the_frozen_digest"),
    (lambda p: p.update(execution={"mode": "default"}), "execution_mode_matches_the_profile"),
    (lambda p: p["post"].update(platform_class="NvmlCudaPlatform"), "platform_class_is_the_expected_one"),       # an unpatched NVML class on a MIG value
    (lambda p: p["post"].update(arch_list=["sm_90"]), "arch_list_contains_the_device_architecture"),
    (lambda p: p["post"].update(capability=None), "arch_list_contains_the_device_architecture"),
    (lambda p: p.update(rows=[{"stop_reason": "max_new_tokens", "text": "x"}]), "every_decode_ended_with_an_end_token_and_none_is_missing"),
    (lambda p: p.update(rows=[]), "every_decode_ended_with_an_end_token_and_none_is_missing"),
    (lambda p: p.update(rows=[{"stop_reason": None, "text": "x"}]), "every_decode_ended_with_an_end_token_and_none_is_missing"),
])
def test_each_deviation_fails_exactly_its_named_check(monkeypatch, mutation, failing):
    parts = good_parts()
    mutation(parts)
    checks = checks_of(monkeypatch, parts)
    assert checks[failing] is False and [name for name, ok in checks.items() if not ok] == [failing]


def test_the_expected_capability_the_empty_text_rule_and_the_engine_expectation_are_checks(monkeypatch):
    parts = good_parts(FP16)
    assert checks_of(monkeypatch, parts, expect_capability=(7, 5))["capability_is_the_expected_one"] is False
    parts["rows"] = [{"stop_reason": "end_token", "text": "  "}]
    assert checks_of(monkeypatch, parts, require_no_empty_text=True)["no_empty_text_windows"] is False
    assert "no_empty_text_windows" not in checks_of(monkeypatch, parts)                                         # reported, not gated, unless asked
    parts = good_parts(FP16)
    parts["post"]["engine_effective"] = {**frozen.effective_engine_expectation(FP16), "dtype": "torch.bfloat16"}
    assert checks_of(monkeypatch, parts)["effective_engine_equals_the_profile_expectation"] is False


def test_the_real_frozen_check_is_one_of_the_named_checks():
    parts = good_parts()
    assert ds.plan_checks(**parts)["observed_settings_pass_the_frozen_check"] is False                          # an empty observed block does not pass the real check


# ---- classification (REV 3.1 Amendment 3) --------------------------------------------------------------------------------------------------------------------------------------------

LOGS = {
    "oom": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "torch_oom": "torch.OutOfMemoryError: CUDA out of memory",
    "no_cache": "ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization",
    "max_seq": "ValueError: To serve at least one request with the models's max seq len (1300), 0.4 GiB KV cache is needed",
    "free_mem": "ValueError: Free memory on device (10.1/14.5 GiB) on startup is less than desired GPU memory utilization",
    "ninja": "Traceback (most recent call last):\n  File x\nRuntimeError: Ninja build failed\nninja: build stopped: subcommand failed.",
    "kernel": "Traceback (most recent call last):\nRuntimeError: CUDA error: no kernel image is available for execution on the device",
    "other": "Traceback (most recent call last):\nValueError: something else entirely",
    "silent": "the log says nothing recognisable",
}


@pytest.mark.parametrize("key, expected", [("oom", "memory_limited"), ("torch_oom", "memory_limited"), ("no_cache", "memory_limited"), ("max_seq", "memory_limited"), ("free_mem", "memory_limited"),
                                           ("ninja", "software_failure"), ("kernel", "software_failure"), ("other", "software_failure"), ("silent", "unclassified")])
def test_log_signatures_are_classified(key, expected):
    assert ds.classify_outcome(1, LOGS[key], None) == expected


def test_memory_signatures_win_over_the_traceback_that_carries_them_and_unknown_text_is_never_mapped():
    assert ds.classify_outcome(1, "Traceback (most recent call last):\n" + LOGS["torch_oom"], None) == "memory_limited"
    assert ds.classify_outcome(1, "", None) == "unclassified"


def test_a_completed_run_with_empty_or_unfinished_decodes_is_numerically_suspect_and_a_passing_one_is_a_pass():
    assert ds.classify_outcome(0, "", {"passed": True, "completed": True, "empty_text_windows": 0, "all_end_token": True}) == "pass"
    assert ds.classify_outcome(1, "", {"passed": False, "completed": True, "empty_text_windows": 2, "all_end_token": True}) == "numerical_suspect"
    assert ds.classify_outcome(1, "", {"passed": False, "completed": True, "empty_text_windows": 0, "all_end_token": False}) == "numerical_suspect"
    assert ds.classify_outcome(1, "", {"passed": False, "completed": True, "empty_text_windows": 0, "all_end_token": True, "checks": {"platform_class_is_the_expected_one": False}}) == "software_failure"


def test_every_class_has_a_meaning_and_none_is_automatic():
    assert set(ds.MEANING) == {"pass", "memory_limited", "software_failure", "numerical_suspect", "unclassified"}
    assert "pending T1" in ds.MEANING["pass"] and "INCONCLUSIVE" in ds.MEANING["memory_limited"]


@pytest.mark.parametrize("measured, wording", [(11.7, "INCONCLUSIVE for the T4"), (14.5, "a finding against the profile on 16 GB")])
def test_the_classify_command_reads_the_measured_memory_against_the_t4_reference(tmp_path, capsys, measured, wording):
    log = tmp_path / "job.log"
    log.write_text(LOGS["oom"] + "\nINFO [kv_cache_utils.py:1307] GPU KV cache size: 308,592 tokens", encoding="utf-8")
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"memory_reference": {"measured_total_memory_gib": measured}}), encoding="utf-8")
    assert ds.main(["classify", str(log), "1", str(report)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "memory_limited" and wording in result["meaning"] and result["log_facts"]["kv_cache_tokens"] == 308592


def test_log_facts_reads_the_vllm_0_16_0_line_formats():
    text = textwrap.dedent("""\
        INFO 11-17 01:53:37 [cuda.py:331] Using AttentionBackendEnum.FLASHINFER backend.
        INFO 11-17 01:53:39 [gpu_model_runner.py:4221] Model loading took 6.19 GiB memory and 2.768464 seconds
        INFO 11-17 01:53:47 [gpu_worker.py:373] Available KV cache memory: 30.61 GiB
        INFO 11-17 01:53:47 [kv_cache_utils.py:1307] GPU KV cache size: 308,592 tokens
        INFO 11-17 01:53:47 [kv_cache_utils.py:1312] Maximum concurrency for 1,300 tokens per request: 233.79x
        WARNING Request 7 preempted by scheduler""")
    assert ds.log_facts(text) == {"kv_cache_tokens": 308592, "maximum_concurrency": "233.79", "attention_backend_selected": "FLASHINFER", "model_memory_gib": "6.19", "available_kv_cache_gib": "30.61",
                                  "preemption_lines": 1}
    assert ds.log_facts("nothing")["kv_cache_tokens"] is None


def test_the_gpu_query_parser_ignores_a_header_and_reads_one_row_per_device():
    rows = ds.parse_gpu_query("name, memory.total [MiB], driver_version, compute_cap\nTesla T4, 15360 MiB, 570.172.08, 7.5")
    assert rows == [{"name": "Tesla T4", "memory_total": "15360 MiB", "driver_version": "570.172.08", "compute_cap": "7.5"}]
    assert ds.parse_gpu_query("") == []
