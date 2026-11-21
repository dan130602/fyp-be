"""Stage 3 follow-up E: execution profiles. Deterministic offline scheduling (VLLM_ENABLE_V1_MULTIPROCESSING=0, the engine core inside the worker process) is an explicitly identified profile of the
frozen settings, never a launcher variable: it is part of the settings id, the frozen digest, the config fingerprint and the resume identity. The three original tables are untouched."""
import ast
import inspect
import os
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

VARIABLE = "VLLM_ENABLE_V1_MULTIPROCESSING"
STOCK, CANDIDATE = "stock-vllm-package-v1", "vllm-rp105-v1"
STOCK_SCHED0, CANDIDATE_SCHED0 = "stock-vllm-package-sched0-v1", "vllm-rp105-sched0-v1"
DEFAULT_IDS, SCHED0_IDS = (STOCK, CANDIDATE), (STOCK_SCHED0, CANDIDATE_SCHED0)
DIGESTS = {"stock-transformers-v1": "511903ac4249ad1873ce95b209e907e0b89a94bbd4ab393e37313313ff1fb01d",
           STOCK: "e526468f938c33b955799b80a3eeae437ea1e09ee19e3286971c72d67b0941d2",
           CANDIDATE: "4b5e073d1e4cbd09c5283023264712de6f17920a04ce9f2d81834037a253dbfe",
           STOCK_SCHED0: "753e2d38eeae719ae6412823cfbe9638f959583adf466077e3799ac7898683e2",
           CANDIDATE_SCHED0: "9e809fdeba5cde9a063cc2880eddc084f98ba739a1b34ac7af316d385233e82f"}
RUNTIME = Path(cfg.CONFIG_ROOT) / "runtime"
CONFIG_STEM = "fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-"
CONFIGS = {STOCK: RUNTIME / (CONFIG_STEM + "vllm-stock.json"), CANDIDATE: cfg.CANDIDATE_RP105_CONFIG, STOCK_SCHED0: RUNTIME / (CONFIG_STEM + "vllm-stock-sched0.json"),
           CANDIDATE_SCHED0: RUNTIME / (CONFIG_STEM + "vllm-rp105-sched0.json")}


class SyncMPClient:
    pass


class InprocClient:
    pass


def backend_with(client=None, **kwargs):
    backend = FakeBackend([completion("x")], **kwargs)
    if client is not None:
        backend.llm.llm_engine = types.SimpleNamespace(engine_core=client())
    return backend


def build(settings_id, client=None, **kwargs):
    return vd.VllmDecoder(backend=backend_with(client), package_info=package_info(), settings_id=settings_id, **kwargs)


# ---- the tables: the three originals untouched, the two profiles distinct -----------------------------------------------------------------------------------------------------

def test_the_original_digests_are_unchanged_and_all_five_are_distinct():
    for settings_id, digest in DIGESTS.items():
        assert frozen.frozen_digest(settings_id) == digest
    assert len(set(DIGESTS.values())) == 5


def test_only_the_two_profiles_have_an_execution_entry_and_it_is_the_only_difference_from_their_base():
    for settings_id in ("stock-transformers-v1", *DEFAULT_IDS):
        assert frozen.execution_of(settings_id) is None
    for settings_id, base in ((STOCK_SCHED0, STOCK), (CANDIDATE_SCHED0, CANDIDATE)):
        assert frozen.execution_of(settings_id) == {VARIABLE: "0", "engine_core_client": "InprocClient"}
        table, base_table = frozen.frozen_settings(settings_id), frozen.frozen_settings(base)
        assert table.pop("execution") and table == base_table
        assert frozen.BACKEND_OF[settings_id] == "vllm"


def test_execution_of_resolves_like_the_other_helpers_and_rejects_unknown_ids():
    assert frozen.execution_of("vllm") is None and frozen.execution_of("transformers") is None
    with pytest.raises(DecodingSettingsError, match="no frozen decoding settings"):
        frozen.execution_of("vllm-rp105-sched1-v1")


def test_the_profiles_record_what_they_change_as_a_known_difference_and_the_originals_are_unchanged():
    for settings_id, base in ((STOCK_SCHED0, STOCK), (CANDIDATE_SCHED0, CANDIDATE)):
        record = frozen.settings_record(settings_id, {})
        base_differences = frozen.settings_record(base, {})["known_differences"]
        assert record["known_differences"][:-1] == base_differences and "InprocClient" in record["known_differences"][-1] and "deterministic" in record["known_differences"][-1]
        assert record["settings_id"] == settings_id and record["frozen_digest"] == DIGESTS[settings_id]
    assert frozen.settings_record("vllm", {})["known_differences"] == frozen.KNOWN_DIFFERENCES["vllm"]


# ---- the frozen check compares the observed execution --------------------------------------------------------------------------------------------------------------------------

def observed_for(settings_id, execution):
    decoder = build(settings_id, client=InprocClient if execution else SyncMPClient)
    return decoder.observed, decoder


def test_the_frozen_check_accepts_the_profiles_own_execution_and_rejects_every_other():
    for settings_id in SCHED0_IDS:
        observed, _ = observed_for(settings_id, True)
        frozen.check_frozen(settings_id, observed)
        for bad in ({VARIABLE: None, "engine_core_client": "InprocClient"}, {VARIABLE: "0", "engine_core_client": "SyncMPClient"}, {VARIABLE: "1", "engine_core_client": "InprocClient"}, {}):
            with pytest.raises(DecodingSettingsError, match="execution"):
                frozen.check_frozen(settings_id, {**observed, "execution": bad})
        without = dict(observed)
        without.pop("execution")
        with pytest.raises(DecodingSettingsError, match="execution"):
            frozen.check_frozen(settings_id, without)


def test_a_default_profile_has_no_execution_expectation_and_its_check_is_unchanged():
    for settings_id in DEFAULT_IDS:
        observed, _ = observed_for(settings_id, False)
        assert "execution" not in observed
        frozen.check_frozen(settings_id, observed)


# ---- plan_execution: the truth table -------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("settings_id, launcher, diagnostic, expected", [
    (STOCK, None, False, ("default", "SyncMPClient")), (STOCK, "1", False, ("default", "SyncMPClient")), (CANDIDATE, None, False, ("default", "SyncMPClient")),
    (STOCK, "0", True, ("diagnostic-in-process", "InprocClient")), (CANDIDATE, "0", True, ("diagnostic-in-process", "InprocClient")),
    (STOCK_SCHED0, None, False, ("profile-in-process", "InprocClient")), (STOCK_SCHED0, "0", False, ("profile-in-process", "InprocClient")),
    (CANDIDATE_SCHED0, None, False, ("profile-in-process", "InprocClient")), (CANDIDATE_SCHED0, "0", False, ("profile-in-process", "InprocClient")),
])
def test_plan_execution_accepts_exactly_the_allowed_combinations(settings_id, launcher, diagnostic, expected):
    assert vd.plan_execution(settings_id, launcher, diagnostic) == expected


@pytest.mark.parametrize("settings_id, launcher, diagnostic, message", [
    (STOCK, "0", False, "never run with"), (CANDIDATE, "0", False, "never run with"), (STOCK, "2", False, "never run with"),            # a default profile never runs in process in production
    (STOCK, None, True, "needs"), (STOCK, "1", True, "needs"),                                                                        # a diagnostic construction must set the variable on its command line
    (STOCK_SCHED0, "1", False, "set .* themselves"), (CANDIDATE_SCHED0, "2", False, "set .* themselves"),                            # a profile refuses a conflicting launcher
    (STOCK_SCHED0, "0", True, "not a diagnostic"), (CANDIDATE_SCHED0, None, True, "not a diagnostic"),                               # a profile is not a diagnostic
])
def test_plan_execution_rejects_every_conflict(settings_id, launcher, diagnostic, message):
    with pytest.raises(DecodingSettingsError, match=message):
        vd.plan_execution(settings_id, launcher, diagnostic)


# ---- the decoder ---------------------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("settings_id", DEFAULT_IDS)
def test_a_default_profile_decoder_refuses_the_variable_set_to_zero_and_accepts_unset_or_one(settings_id, ngram_env, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    with pytest.raises(DecodingSettingsError, match="never run with"):
        build(settings_id)
    monkeypatch.setenv(VARIABLE, "1")
    assert build(settings_id, client=SyncMPClient).execution["mode"] == "default"
    monkeypatch.delenv(VARIABLE)
    description = build(settings_id, client=SyncMPClient).describe()
    assert description["execution"] == {"mode": "default", "variable": None, "engine_core_client": "SyncMPClient", "expected_engine_core_client": "SyncMPClient"}
    assert VARIABLE not in os.environ                                                       # a default profile never sets it


@pytest.mark.parametrize("settings_id", SCHED0_IDS)
def test_a_scheduling_profile_decoder_sets_the_variable_itself_before_the_backend_is_built(settings_id, ngram_env, monkeypatch):
    seen = {}

    class Order(FakeBackend):
        pass

    decoder = vd.VllmDecoder(backend=backend_with(InprocClient), package_info=lambda: seen.setdefault("at_info", os.environ.get(VARIABLE)) and package_info()(), settings_id=settings_id)
    assert seen["at_info"] == "0" and os.environ[VARIABLE] == "0"                           # set before the package information (and, in production, the engine) is touched
    assert decoder.execution == {"mode": "profile-in-process", "variable": "0", "engine_core_client": "InprocClient", "expected_engine_core_client": "InprocClient"}
    assert decoder.describe()["execution"] == decoder.execution and decoder.observed["execution"] == {VARIABLE: "0", "engine_core_client": "InprocClient"}


@pytest.mark.parametrize("settings_id", SCHED0_IDS)
def test_a_scheduling_profile_decoder_refuses_a_conflicting_launcher_and_a_diagnostic_construction(settings_id, ngram_env, monkeypatch):
    monkeypatch.setenv(VARIABLE, "1")
    with pytest.raises(DecodingSettingsError, match="set .* themselves"):
        build(settings_id, client=InprocClient)
    monkeypatch.delenv(VARIABLE)
    with pytest.raises(DecodingSettingsError, match="not a diagnostic"):
        build(settings_id, client=InprocClient, diagnostic_in_process=True)
    monkeypatch.setenv(VARIABLE, "0")
    assert build(settings_id, client=InprocClient).execution["mode"] == "profile-in-process"          # its own value is tolerated


@pytest.mark.parametrize("settings_id", DEFAULT_IDS)
def test_the_diagnostic_construction_of_a_default_profile_is_stamped_and_needs_the_variable(settings_id, ngram_env, monkeypatch):
    with pytest.raises(DecodingSettingsError, match="needs"):
        build(settings_id, client=InprocClient, diagnostic_in_process=True)
    monkeypatch.setenv(VARIABLE, "0")
    decoder = build(settings_id, client=InprocClient, diagnostic_in_process=True)
    assert decoder.execution["mode"] == "diagnostic-in-process" and decoder.describe()["execution"]["engine_core_client"] == "InprocClient"
    assert "execution" not in decoder.observed and decoder.settings_id == settings_id           # the frozen table of a default profile is still compared unchanged


def test_the_observed_engine_client_must_be_the_expected_one(ngram_env, monkeypatch):
    with pytest.raises(DecodingSettingsError, match="expects the engine client SyncMPClient, observed InprocClient"):
        build(STOCK, client=InprocClient)
    with pytest.raises(DecodingSettingsError, match="expects the engine client InprocClient, observed SyncMPClient"):
        build(STOCK_SCHED0, client=SyncMPClient)
    monkeypatch.setenv(VARIABLE, "0")
    with pytest.raises(DecodingSettingsError, match="expects the engine client InprocClient, observed SyncMPClient"):
        build(CANDIDATE, client=SyncMPClient, diagnostic_in_process=True)


def test_an_injected_backend_without_an_engine_path_records_none_and_a_built_one_without_it_is_an_error(ngram_env, monkeypatch):
    decoder = build(STOCK)
    assert decoder.execution["engine_core_client"] is None and decoder.execution["mode"] == "default"
    assert vd.engine_client_name(backend_with(InprocClient)) == "InprocClient" and vd.engine_client_name(backend_with()) is None

    class Built(FakeBackend):
        pass

    fake_package = types.ModuleType("meralion_3_asr")
    fake_package.Meralion3ASR = types.SimpleNamespace(from_pretrained=lambda path, backend=None: types.SimpleNamespace(_backend=Built([])))
    monkeypatch.setitem(__import__("sys").modules, "meralion_3_asr", fake_package)
    with pytest.raises(DecodingSettingsError, match="cannot be observed"):
        vd.VllmDecoder("model", package_info=package_info())


def test_the_profile_enters_the_identity_through_the_settings_id_and_the_digest(ngram_env):
    base, profile = build(CANDIDATE), build(CANDIDATE_SCHED0, client=InprocClient)
    assert base.identity()["settings_id"] != profile.identity()["settings_id"] and base.identity()["frozen_digest"] != profile.identity()["frozen_digest"]
    assert set(Decoder.identity.__code__.co_names) >= {"describe"} and vd.__name__.endswith("vllm_decoder")
    assert profile.identity()["frozen_digest"] == DIGESTS[CANDIDATE_SCHED0]


def test_the_candidate_profile_still_generates_with_an_adapter_owned_object(ngram_env):
    backend = backend_with(InprocClient)
    original = backend.sampling_params
    snapshot = dict(vars(original))
    decoder = vd.VllmDecoder(backend=backend, package_info=package_info(), settings_id=CANDIDATE_SCHED0)
    assert decoder.sampling_params is not original and decoder.sampling_params.repetition_penalty == 1.05 and backend.sampling_params is original and dict(vars(original)) == snapshot


def test_production_construction_never_passes_the_diagnostic_argument():
    tree = ast.parse(inspect.getsource(cli))
    names = {node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword)}
    assert "diagnostic_in_process" not in names and "diagnostic_in_process" not in inspect.getsource(cli)
    users = [path.name for path in (Path(__file__).resolve().parent.parent / "tools").rglob("*.py") if "diagnostic_in_process=" in path.read_text(encoding="utf-8")]
    assert users == ["vllm_plugin_probe.py"]                                                 # the one tool that must inspect the engine in process


# ---- selection and identity through the configuration ----------------------------------------------------------------------------------------------------------------------------

def load(path=None, **kwargs):
    kwargs.setdefault("env", {})
    return cfg.load_runtime_config(path, **kwargs)


def test_the_five_vllm_settings_ids_are_allowed_and_each_config_names_its_own():
    assert set(cfg.ALLOWED_SETTINGS["vllm"]) == {STOCK, CANDIDATE, STOCK_SCHED0, CANDIDATE_SCHED0, "vllm-rp105-sched0-fp16-v1"} and cfg.ALLOWED_SETTINGS["transformers"] == ("stock-transformers-v1",)
    fingerprints = {load().fingerprint}
    default = load()
    for settings_id, path in CONFIGS.items():
        effective = load(path)
        assert effective.data["decoder"] == {"backend": "vllm", "settings_id": settings_id} and effective.data["config_id"] == path.stem
        assert {k for k in effective.data if effective.data[k] != default.data[k]} == {"config_id", "decoder"}            # same strategy, formatter and manifest as the default
        fingerprints.add(effective.fingerprint)
    assert len(fingerprints) == 5


def test_a_backend_override_alone_still_selects_the_stock_default_and_a_profile_is_selected_only_by_name():
    assert load(env={"MERALION_BACKEND": "vllm"}).data["decoder"]["settings_id"] == STOCK
    assert load(cfg.CANDIDATE_RP105_CONFIG, env={"MERALION_BACKEND": "transformers"}).data["decoder"]["settings_id"] == "stock-transformers-v1"
    for settings_id in SCHED0_IDS:
        assert load(env={"MERALION_BACKEND": "vllm"}, cli={"decoder_settings": settings_id}).data["decoder"]["settings_id"] == settings_id
        with pytest.raises(cfg.ConfigError, match="contradicts"):
            load(env={"MERALION_DECODER_SETTINGS": settings_id})                                                 # the default Transformers backend cannot take a vLLM profile


class WithSettings(Decoder):
    def __init__(self, inner, settings_id):
        self.inner, self.settings_id = inner, settings_id

    def decode(self, piece, **kwargs):
        return self.inner.decode(piece, **kwargs)

    def describe(self):
        return {**self.inner.describe(), "backend": "vllm", "settings_id": self.settings_id, "frozen_digest": frozen.frozen_digest(self.settings_id)}


@pytest.mark.parametrize("first, second", [(STOCK, STOCK_SCHED0), (STOCK_SCHED0, STOCK), (CANDIDATE, CANDIDATE_SCHED0), (CANDIDATE_SCHED0, CANDIDATE), (STOCK_SCHED0, CANDIDATE_SCHED0)])
def test_a_state_started_under_one_profile_cannot_be_resumed_under_another(tmp_path, first, second):
    world = synthetic.World({"alpha": dict(break_at=2)})
    run(world, tmp_path, decoder=WithSettings(world.decoder(), first))
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(world, tmp_path, decoder=WithSettings(world.decoder(), second))
    assert run(world, tmp_path, decoder=WithSettings(world.decoder(), first)).exit_code == 0


# ---- the probe's pure planning ---------------------------------------------------------------------------------------------------------------------------------------------------

def load_tool(name, folder="origami"):
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "tools" / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("settings_id, launcher, expected_kind, refused", [
    (STOCK, "0", "diagnostic-in-process", False), (CANDIDATE, "0", "diagnostic-in-process", False), (STOCK, None, None, True), (CANDIDATE, "1", None, True),
    (STOCK_SCHED0, None, "profile-in-process", False), (CANDIDATE_SCHED0, None, "profile-in-process", False), (STOCK_SCHED0, "0", None, True), (CANDIDATE_SCHED0, "1", None, True)])
def test_the_probe_plans_a_diagnostic_for_default_profiles_and_a_profile_probe_for_scheduling_profiles(settings_id, launcher, expected_kind, refused):
    probe = load_tool("vllm_plugin_probe")
    kind, problem = probe.plan_probe(settings_id, launcher)
    assert kind == expected_kind and bool(problem) is refused
