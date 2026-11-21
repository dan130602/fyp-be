"""REV 3.0 part B: the FP16 profile `vllm-rp105-sched0-fp16-v1` for GPUs below compute capability 8.0 (the Singapore T4).

It is a SEPARATE, explicitly named profile: the validated BF16 profile and its digests are untouched, the only addition is one constructor override (dtype float16) and the engine values that must be
observed. Nothing falls back from one profile to the other. FP16 on this checkpoint is NOT validated by any of these tests (they use fakes).
"""
import ast
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

import synthetic
from harness import run
from meralion_worker import cli
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen
from meralion_worker.decoders import vllm_decoder as vd
from meralion_worker.decoders.base import Decoder, DecodingSettingsError
from meralion_worker.state import ResumeError

from vllm_fakes import FakeBackend, completion, package_info

FP16 = "vllm-rp105-sched0-fp16-v1"
BF16 = "vllm-rp105-sched0-v1"
FP16_DIGEST = "b7be16b45da4b6d005175cd3b3f3e036eb06fe637c36b777050533ff504bed4a"
EXISTING = {"stock-transformers-v1": "511903ac4249ad1873ce95b209e907e0b89a94bbd4ab393e37313313ff1fb01d",
            "stock-vllm-package-v1": "e526468f938c33b955799b80a3eeae437ea1e09ee19e3286971c72d67b0941d2",
            "vllm-rp105-v1": "4b5e073d1e4cbd09c5283023264712de6f17920a04ce9f2d81834037a253dbfe",
            "stock-vllm-package-sched0-v1": "753e2d38eeae719ae6412823cfbe9638f959583adf466077e3799ac7898683e2",
            BF16: "9e809fdeba5cde9a063cc2880eddc084f98ba739a1b34ac7af316d385233e82f"}
FP16_CONFIG = Path(cfg.CONFIG_ROOT) / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0-fp16.json"
BF16_CONFIG = Path(cfg.CONFIG_ROOT) / "runtime" / "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json"
TOOLS = Path(__file__).resolve().parent.parent / "tools"


class InprocClient:
    pass


class SyncMPClient:
    pass


class Dtype:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def engine_config(dtype="torch.float16", max_model_len=1300, max_num_seqs=64, utilisation=0.85, tensor_parallel=1, attention="FLASHINFER"):
    """What backend.llm.llm_engine.vllm_config exposes (the attribute paths read by vd.engine_effective)."""
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=Dtype(dtype), max_model_len=max_model_len), scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=types.SimpleNamespace(gpu_memory_utilization=utilisation, num_gpu_blocks=4321, block_size=16), parallel_config=types.SimpleNamespace(tensor_parallel_size=tensor_parallel),
        attention_config=types.SimpleNamespace(backend=None if attention is None else types.SimpleNamespace(name=attention)))


def fake_backend(config="default", client=InprocClient):
    backend = FakeBackend([completion("x")])
    engine = types.SimpleNamespace(engine_core=client())
    if config is not None:
        engine.vllm_config = engine_config() if config == "default" else config
    backend.llm.llm_engine = engine
    return backend


def build(settings_id=FP16, backend=None, **kwargs):
    return vd.VllmDecoder(backend=backend or fake_backend(), package_info=kwargs.pop("package_info", package_info()), settings_id=settings_id, **kwargs)


# ---- the table and the registry --------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_fp16_digest_is_pinned_and_the_five_existing_digests_are_unchanged():
    assert frozen.frozen_digest(FP16) == FP16_DIGEST
    for settings_id, digest in EXISTING.items():
        assert frozen.frozen_digest(settings_id) == digest
    assert len({FP16_DIGEST, *EXISTING.values()}) == 6


def test_the_fp16_table_is_the_bf16_scheduling_table_plus_exactly_the_override_and_the_expected_engine():
    table, base = frozen.frozen_settings(FP16), frozen.frozen_settings(BF16)
    assert table.pop("constructor_overrides") == {"dtype": "float16"}
    assert table.pop("expected_effective_engine") == {"dtype": "torch.float16", "max_model_len": 1300, "max_num_seqs": 64, "gpu_memory_utilization": 0.85, "tensor_parallel_size": 1,
                                                      "attention_backend": "FLASHINFER"}
    assert table == base and table["constructor_defaults"]["dtype"] == "bfloat16"                      # what the package ships is still recorded, so a package change is still detected
    assert frozen.execution_of(FP16) == frozen.execution_of(BF16) == {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "engine_core_client": "InprocClient"}
    assert frozen.BACKEND_OF[FP16] == "vllm" and "float16" in " ".join(frozen.KNOWN_DIFFERENCES_BY_SETTINGS[FP16])


def test_overrides_and_expectations_exist_for_the_fp16_profile_only():
    assert frozen.constructor_overrides_of(FP16) == {"dtype": "float16"} and frozen.effective_engine_expectation(FP16)["dtype"] == "torch.float16"
    for settings_id in EXISTING:
        assert frozen.constructor_overrides_of(settings_id) == {} and frozen.effective_engine_expectation(settings_id) is None
    assert frozen.constructor_overrides_of("vllm") == {}                                              # legacy backend names resolve to the stock profile
    first = frozen.constructor_overrides_of(FP16)
    first["dtype"] = "bfloat16"
    assert frozen.constructor_overrides_of(FP16) == {"dtype": "float16"}                              # a deep copy: the table cannot be changed through the helper


def test_the_config_pair_is_allowed_and_a_contradictory_pair_is_rejected_and_the_fingerprints_differ():
    effective = cfg.load_runtime_config(FP16_CONFIG, env={})
    bf16 = cfg.load_runtime_config(BF16_CONFIG, env={})
    assert effective.data["decoder"] == {"backend": "vllm", "settings_id": FP16} and effective.data["config_id"] == FP16_CONFIG.stem
    assert {k for k in effective.data if effective.data[k] != bf16.data[k]} == {"config_id", "decoder"} and effective.fingerprint != bf16.fingerprint
    with pytest.raises(cfg.ConfigError, match="contradicts"):
        cfg.load_runtime_config(None, env={"MERALION_DECODER_SETTINGS": FP16})                         # the default Transformers backend cannot take a vLLM profile
    assert cfg.load_runtime_config(None, env={"MERALION_BACKEND": "vllm"}).data["decoder"]["settings_id"] == "stock-vllm-package-v1"      # FP16 is selected only by naming it


# ---- the constructor call --------------------------------------------------------------------------------------------------------------------------------------------------------------

def stub_package(monkeypatch, backend_factory):
    calls = []

    def from_pretrained(path, backend=None, **kwargs):
        calls.append({"path": path, "backend": backend, "kwargs": dict(kwargs)})
        return types.SimpleNamespace(_backend=backend_factory())
    module = types.ModuleType("meralion_3_asr")
    module.Meralion3ASR = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "meralion_3_asr", module)
    return calls


def test_the_package_constructor_receives_exactly_dtype_float16_for_fp16_and_nothing_for_every_existing_profile(ngram_env, monkeypatch):
    calls = stub_package(monkeypatch, lambda: fake_backend())
    vd.VllmDecoder("model", package_info=package_info(), settings_id=FP16)
    assert calls[-1]["kwargs"] == {"dtype": "float16"} and calls[-1]["backend"] == "vllm"
    for settings_id in EXISTING:
        if frozen.BACKEND_OF[settings_id] != "vllm":
            continue
        scheduling = frozen.execution_of(settings_id) is not None
        calls.clear()
        monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
        stub_calls = stub_package(monkeypatch, lambda scheduling=scheduling: fake_backend(client=InprocClient if scheduling else SyncMPClient))
        vd.VllmDecoder("model", package_info=package_info(), settings_id=settings_id)
        assert stub_calls[-1]["kwargs"] == {}, settings_id                                           # the BF16 and stock calls are exactly what they were before this profile existed


# ---- the engine observation and the frozen check ----------------------------------------------------------------------------------------------------------------------------

def test_a_matching_engine_passes_and_is_recorded_with_its_capacity_facts(ngram_env):
    decoder = build()
    effective = decoder.describe()["engine_effective"]
    assert effective["dtype"] == "torch.float16" and effective["attention_backend"] == "FLASHINFER" and effective["num_gpu_blocks"] == 4321 and effective["block_size"] == 16
    assert decoder.describe()["constructor_overrides"] == {"dtype": "float16"} and decoder.describe()["package_constructor_defaults"]["dtype"] == "bfloat16"
    assert decoder.observed["effective_engine"]["max_num_seqs"] == 64


@pytest.mark.parametrize("override, fragment", [
    (dict(dtype="torch.bfloat16"), "dtype"), (dict(attention="TRITON_ATTN"), "attention_backend"), (dict(max_num_seqs=32), "max_num_seqs"), (dict(max_model_len=2048), "max_model_len"),
    (dict(utilisation=0.9), "gpu_memory_utilization"), (dict(tensor_parallel=2), "tensor_parallel_size"), (dict(attention=None), "attention_backend")])
def test_an_engine_that_differs_from_the_expectation_is_refused(ngram_env, override, fragment):
    with pytest.raises(DecodingSettingsError, match=fragment):
        build(backend=fake_backend(config=engine_config(**override)))


def test_an_engine_built_here_that_cannot_be_observed_is_an_error_but_an_injected_double_is_still_checked(ngram_env, monkeypatch):
    stub_package(monkeypatch, lambda: fake_backend(config=None))
    with pytest.raises(DecodingSettingsError, match="cannot be observed"):
        vd.VllmDecoder("model", package_info=package_info(), settings_id=FP16)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    with pytest.raises(DecodingSettingsError, match="effective_engine"):                                # an injected double without an engine is not silently accepted either
        build(backend=fake_backend(config=None))


def test_the_bf16_profiles_describe_records_no_override_and_no_engine_when_the_double_has_none(ngram_env):
    decoder = vd.VllmDecoder(backend=fake_backend(config=None), package_info=package_info(), settings_id=BF16)
    described = decoder.describe()
    assert described["constructor_overrides"] == {} and described["engine_effective"] is None and "effective_engine" not in decoder.observed


def test_a_changed_package_default_is_still_detected_under_the_fp16_profile(ngram_env):
    defaults = dict(__import__("vllm_fakes").CONSTRUCTOR_DEFAULTS, dtype="float16")                    # the package itself now defaulting to FP16 would be a package change
    with pytest.raises(DecodingSettingsError, match="constructor_defaults"):
        build(package_info=package_info(constructor_defaults=defaults))


def test_check_frozen_rejects_wrong_or_missing_overrides_for_the_fp16_profile():
    base = {"requested_sampling": frozen.frozen_settings(FP16)["requested_sampling"], "normalised_sampling": frozen.frozen_settings(FP16)["normalised_sampling"],
            "package_requested_sampling": frozen.frozen_settings(FP16)["package_requested_sampling"], "environment": {"MERALION_NGRAM_SIZE": "6"},
            "constructor_defaults": frozen.frozen_settings(FP16)["constructor_defaults"], "execution": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "engine_core_client": "InprocClient"},
            "effective_engine": frozen.effective_engine_expectation(FP16)}
    frozen.check_frozen(FP16, {**base, "constructor_overrides": {"dtype": "float16"}})
    for overrides in ({}, {"dtype": "bfloat16"}, {"dtype": "float16", "max_num_seqs": 8}):
        with pytest.raises(DecodingSettingsError, match="constructor_overrides"):
            frozen.check_frozen(FP16, {**base, "constructor_overrides": overrides})
    with pytest.raises(DecodingSettingsError, match="effective_engine"):
        frozen.check_frozen(FP16, {**base, "constructor_overrides": {"dtype": "float16"}, "effective_engine": {**base["effective_engine"], "dtype": "torch.bfloat16"}})


# ---- identity, resume and no fallback --------------------------------------------------------------------------------------------------------------------------------------------

class WithSettings(Decoder):
    def __init__(self, inner, settings_id):
        self.inner, self.settings_id = inner, settings_id

    def decode(self, piece, **kwargs):
        return self.inner.decode(piece, **kwargs)

    def describe(self):
        return {**self.inner.describe(), "backend": "vllm", "settings_id": self.settings_id, "frozen_digest": frozen.frozen_digest(self.settings_id)}


@pytest.mark.parametrize("first, second", [(BF16, FP16), (FP16, BF16)])
def test_a_state_started_under_bf16_cannot_be_resumed_as_fp16_and_the_reverse(tmp_path, first, second):
    world = synthetic.World({"alpha": dict(break_at=2)})
    run(world, tmp_path, decoder=WithSettings(world.decoder(), first))
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(world, tmp_path, decoder=WithSettings(world.decoder(), second))
    assert run(world, tmp_path, decoder=WithSettings(world.decoder(), first)).exit_code == 0


def test_there_is_no_automatic_fallback_between_the_bf16_and_fp16_profiles():
    """cli.make_decoder builds exactly the configured settings id: no try/except around the construction, no second construction with another id, no dtype or capability probing."""
    tree = ast.parse(inspect.getsource(cli))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "make_decoder")
    assert not [node for node in ast.walk(function) if isinstance(node, (ast.Try, ast.ExceptHandler))]
    constructions = [node for node in ast.walk(function) if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("VllmDecoder", "TransformersDecoder")]
    assert len(constructions) == 2
    source = inspect.getsource(cli.make_decoder)
    assert "fp16" not in source and "float16" not in source and "get_device_capability" not in source and "is_bf16_supported" not in source
    assert "fallback" not in Path(vd.__file__).read_text(encoding="utf-8").lower().replace("no fallback", "")          # the decoder has no retry-with-another-profile path either


# ---- the decision tool refuses documents of another engine -------------------------------------------------------------------------------------------------------------------------

def load_decision():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("decision_for_fp16", TOOLS / "decision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_decision_tool_requires_the_fp16_engine_to_be_the_observed_one():
    decision = load_decision()
    good = {"engine_effective": {**frozen.effective_engine_expectation(FP16), "num_gpu_blocks": 10, "block_size": 16}}
    assert decision.engine_problem(FP16, good) is None
    assert "dtype" in decision.engine_problem(FP16, {"engine_effective": {**good["engine_effective"], "dtype": "torch.bfloat16"}})
    assert "max_num_seqs" in decision.engine_problem(FP16, {"engine_effective": {**good["engine_effective"], "max_num_seqs": 16}})
    assert "no engine_effective" in decision.engine_problem(FP16, {})
    assert "no engine_effective" in decision.engine_problem(FP16, {"engine_effective": None})
    assert decision.engine_problem(BF16, {}) is None and decision.engine_problem("stock-vllm-package-v1", None) is None        # a profile without an expectation has nothing to check
