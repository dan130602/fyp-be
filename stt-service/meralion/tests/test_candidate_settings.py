"""Stage 3 follow-up D: the ONE vLLM candidate (repetition_penalty 1.05) as separate settings. The stock settings, their digests and the package object stay exactly as they were."""
import hashlib
import json
import types

import pytest

import synthetic
from harness import run
from meralion_worker import cli
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen
from meralion_worker.decoders import transformers_decoder as td
from meralion_worker.decoders import vllm_decoder as vd
from meralion_worker.decoders.base import Decoder, DecodingSettingsError
from meralion_worker.state import ResumeError

from vllm_fakes import CONSTRUCTOR_DEFAULTS, REQUESTED, FakeBackend, FakeSamplingParams, completion, package_info

# the digests of the stock tables as they were BEFORE the candidate was added: a silent edit of a stock table fails here
STOCK_TRANSFORMERS_DIGEST = "511903ac4249ad1873ce95b209e907e0b89a94bbd4ab393e37313313ff1fb01d"
STOCK_VLLM_DIGEST = "e526468f938c33b955799b80a3eeae437ea1e09ee19e3286971c72d67b0941d2"
CANDIDATE_DIGEST = "4b5e073d1e4cbd09c5283023264712de6f17920a04ce9f2d81834037a253dbfe"
CANDIDATE = "vllm-rp105-v1"
EFFECTIVE = {**REQUESTED, "repetition_penalty": 1.05}


@pytest.fixture
def ngram_env(monkeypatch):
    monkeypatch.delenv("MERALION_NGRAM_SIZE", raising=False)


# ---- the stock tables are untouched, and both call forms resolve the same way -------------------------------------------------------------------------------------------

def test_the_stock_digests_are_exactly_what_they_were_and_the_candidate_digest_is_pinned_too():
    assert frozen.frozen_digest("transformers") == frozen.frozen_digest("stock-transformers-v1") == STOCK_TRANSFORMERS_DIGEST
    assert frozen.frozen_digest("vllm") == frozen.frozen_digest("stock-vllm-package-v1") == STOCK_VLLM_DIGEST
    assert frozen.frozen_digest(CANDIDATE) == CANDIDATE_DIGEST and len({STOCK_TRANSFORMERS_DIGEST, STOCK_VLLM_DIGEST, CANDIDATE_DIGEST}) == 3


def test_a_legacy_backend_name_resolves_to_its_stock_settings_and_anything_else_is_an_error():
    assert frozen.resolve_settings_id("transformers") == "stock-transformers-v1" and frozen.resolve_settings_id("vllm") == "stock-vllm-package-v1"
    assert frozen.resolve_settings_id(CANDIDATE) == CANDIDATE and frozen.resolve_settings_id("stock-vllm-package-v1") == "stock-vllm-package-v1"
    for bad in ("onnx", "", "vllm-rp105", None):
        with pytest.raises(DecodingSettingsError, match="no frozen decoding settings"):
            frozen.resolve_settings_id(bad)


def test_every_helper_gives_identical_results_for_the_backend_name_and_the_settings_id():
    for name, settings_id in (("transformers", "stock-transformers-v1"), ("vllm", "stock-vllm-package-v1")):
        assert frozen.frozen_settings(name) == frozen.frozen_settings(settings_id)
        assert frozen.frozen_digest(name) == frozen.frozen_digest(settings_id)
        assert frozen.settings_record(name, {"x": 1}) == frozen.settings_record(settings_id, {"x": 1})
    assert frozen.FROZEN_DECODING["vllm"] == frozen.DECODING_SETTINGS["stock-vllm-package-v1"] and frozen.FROZEN_DECODING["transformers"] == frozen.DECODING_SETTINGS["stock-transformers-v1"]
    assert set(frozen.FROZEN_DECODING) == {"transformers", "vllm"}                            # the alias is stock-only: the candidate is never reachable through a backend name


def test_the_candidate_differs_from_the_stock_vllm_table_only_in_the_override_and_what_follows_from_it():
    stock, candidate = frozen.frozen_settings("vllm"), frozen.frozen_settings(CANDIDATE)
    assert candidate["package_requested_sampling"] == stock["requested_sampling"] and candidate["overrides"] == {"repetition_penalty": 1.05}
    assert candidate["requested_sampling"] == EFFECTIVE and candidate["normalised_sampling"] == {**stock["normalised_sampling"], "repetition_penalty": 1.05}
    for same in ("environment", "constructor_defaults", "no_stop_configured"):
        assert candidate[same] == stock[same]


def test_the_candidate_has_its_own_known_differences_which_name_the_data_path_confound():
    record = frozen.settings_record(CANDIDATE, {})
    assert record["settings_id"] == CANDIDATE and record["frozen_digest"] == CANDIDATE_DIGEST
    text = " ".join(record["known_differences"])
    assert "1.05" in text and "token ids itself" in text and "not by generate()" in text
    assert frozen.settings_record("vllm", {})["known_differences"] == frozen.KNOWN_DIFFERENCES["vllm"]            # stock unchanged


# ---- the frozen check under the candidate ---------------------------------------------------------------------------------------------------------------------------------

def candidate_observed(effective=None, package=None, normalised=None, environment=None):
    effective = dict(effective or EFFECTIVE)
    return {"package_requested_sampling": dict(package or REQUESTED), "requested_sampling": effective,
            "normalised_sampling": vd.normalised_sampling(FakeSamplingParams(**effective)) if normalised is None else normalised,
            "environment": environment or {"MERALION_NGRAM_SIZE": "6"}, "constructor_defaults": dict(CONSTRUCTOR_DEFAULTS)}


def test_the_candidate_check_passes_the_expected_observation():
    frozen.check_frozen(CANDIDATE, candidate_observed())


@pytest.mark.parametrize("kwargs, message", [
    (dict(effective={**EFFECTIVE, "repetition_penalty": 1.0}), "requested_sampling.repetition_penalty"),          # the override was not applied
    (dict(effective={**EFFECTIVE, "repetition_penalty": 1.1}), "requested_sampling.repetition_penalty"),
    (dict(package={**REQUESTED, "top_k": 40}), "package_requested_sampling.top_k"),                                # the installed package changed what it requests
    (dict(package={**REQUESTED, "repetition_penalty": 1.05}), "package_requested_sampling.repetition_penalty"),
    (dict(environment={"MERALION_NGRAM_SIZE": "4"}), "environment.MERALION_NGRAM_SIZE"),
])
def test_the_candidate_check_rejects_every_deviation(kwargs, message):
    with pytest.raises(DecodingSettingsError, match=message):
        frozen.check_frozen(CANDIDATE, candidate_observed(**kwargs))


def test_the_candidate_check_rejects_a_stop_and_a_missing_package_request():
    with pytest.raises(DecodingSettingsError, match="a stop is configured"):
        frozen.check_frozen(CANDIDATE, candidate_observed(effective={**EFFECTIVE, "stop": ["###"]}))
    observed = candidate_observed()
    del observed["package_requested_sampling"]
    with pytest.raises(DecodingSettingsError, match="package_requested_sampling"):
        frozen.check_frozen(CANDIDATE, observed)


def test_the_stock_check_never_accepts_the_candidates_value():
    with pytest.raises(DecodingSettingsError, match="repetition_penalty"):
        frozen.check_frozen("vllm", {"requested_sampling": EFFECTIVE, "normalised_sampling": vd.normalised_sampling(FakeSamplingParams(**EFFECTIVE)),
                                     "environment": {"MERALION_NGRAM_SIZE": "6"}, "constructor_defaults": dict(CONSTRUCTOR_DEFAULTS)})


# ---- the adapter: stock path unchanged, candidate path adapter-owned -----------------------------------------------------------------------------------------------------

def pieces(n=1):
    import numpy as np
    return [np.full(16000, i, dtype=np.float32) for i in range(n)]


def requests_for(items):
    from meralion_worker.decoders.base import DecodeRequest
    return [DecodeRequest("rec", i * 16000, (i + 1) * 16000, p) for i, p in enumerate(items)]


def test_the_stock_decoder_generates_with_the_packages_own_object(ngram_env):
    backend = FakeBackend([completion("x")])
    decoder = vd.VllmDecoder(backend=backend, package_info=package_info())
    assert decoder.sampling_params is backend.sampling_params and decoder.settings_id == "stock-vllm-package-v1"
    decoder.decode_many(requests_for(pieces()))
    assert backend.llm.calls[0][1] is backend.sampling_params


def test_the_candidate_generates_with_an_adapter_owned_object_and_never_touches_the_packages(ngram_env):
    backend = FakeBackend([completion("x")])
    package_object, package_repr = backend.sampling_params, repr(backend.sampling_params.__dict__)
    decoder = vd.VllmDecoder(backend=backend, package_info=package_info(), settings_id=CANDIDATE)
    assert decoder.sampling_params is not package_object and decoder.sampling_params.repetition_penalty == 1.05
    assert backend.sampling_params is package_object and package_object.repetition_penalty == 1.0 and repr(package_object.__dict__) == package_repr
    decoder.decode_many(requests_for(pieces()))
    assert backend.llm.calls[0][1] is decoder.sampling_params


def test_the_production_path_never_assigns_the_backends_sampling_params(ngram_env):
    class Guarded(FakeBackend):
        locked = False

        def __setattr__(self, name, value):
            if name == "sampling_params" and self.locked:
                raise AssertionError("the adapter assigned backend.sampling_params")
            super().__setattr__(name, value)

    for settings_id in ("stock-vllm-package-v1", CANDIDATE):
        backend = Guarded([completion("x")])
        backend.locked = True
        decoder = vd.VllmDecoder(backend=backend, package_info=package_info(), settings_id=settings_id)
        decoder.decode_many(requests_for(pieces()))
        decoder.describe()


def test_the_candidate_description_records_the_package_and_effective_settings_separately(ngram_env):
    backend = FakeBackend([completion("x")])
    description = vd.VllmDecoder(backend=backend, package_info=package_info(), settings_id=CANDIDATE).describe()
    assert description["settings_id"] == CANDIDATE and description["frozen_digest"] == CANDIDATE_DIGEST and description["backend"] == "vllm"
    sampling = description["sampling"]
    assert sampling["package_requested"]["repetition_penalty"] == 1.0 and sampling["effective_requested"]["repetition_penalty"] == 1.05
    assert sampling["effective_normalised"]["repetition_penalty"] == 1.05 and sampling["effective_normalised"]["top_k"] == 0
    assert isinstance(sampling["effective_object_repr"], str) and isinstance(sampling["package_object_repr"], str) and sampling["effective_object_repr"] != sampling["package_object_repr"]
    assert description["settings"]["observed"]["package_requested_sampling"]["repetition_penalty"] == 1.0


def test_a_settings_id_of_another_backend_cannot_build_a_vllm_decoder(ngram_env):
    with pytest.raises(DecodingSettingsError, match="belong to the"):
        vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info(), settings_id="stock-transformers-v1")
    with pytest.raises(DecodingSettingsError, match="no frozen decoding settings"):
        vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info(), settings_id="vllm-rp110-v1")


def test_a_package_that_changed_its_request_stops_the_candidate_too(ngram_env):
    with pytest.raises(DecodingSettingsError, match="package_requested_sampling.top_k"):
        vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info(requested={**REQUESTED, "top_k": 40}), settings_id=CANDIDATE)


def test_the_token_id_hash_is_recorded_and_absent_ids_stay_unavailable(ngram_env):
    ids = [5, 6, 7]
    decoder = vd.VllmDecoder(backend=FakeBackend([completion("x", token_ids=ids), completion("y", omit=("token_ids",))]), package_info=package_info())
    first, second = decoder.decode_many(requests_for(pieces(2)))
    assert first.native["token_ids_sha256"] == hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest() and second.native["token_ids_sha256"] is None and second.tokens is None


# ---- the two stock decoders' frozen-settings paths with explicit ids (no GPU) ---------------------------------------------------------------------------------------------

def stub_transformers_decoder():
    decoder = object.__new__(td.TransformersDecoder)
    decoder._torch = types.SimpleNamespace(__version__="2.8.0", cuda=types.SimpleNamespace(get_device_name=lambda index: "stub GPU"))
    decoder.load_seconds = 1.0
    decoder.observed = td.observed_settings(package_generation_config={"max_new_tokens": 512, "do_sample": False, "no_repeat_ngram_size": 6},
                                            model_generation_config={"repetition_penalty": 1.05, "no_repeat_ngram_size": 6, "eos_token_id": 1, "bos_token_id": 2, "pad_token_id": 0},
                                            attn_implementation="eager", batch_size=1, dtype="torch.bfloat16", bf16_supported=True, has_lora_parameters=False)
    return decoder


def test_the_transformers_decoder_checks_and_describes_under_its_explicit_stock_id():
    decoder = stub_transformers_decoder()
    decoder._verify_frozen(decoder.observed)
    description = decoder.describe()
    assert description["settings_id"] == "stock-transformers-v1" and description["frozen_digest"] == STOCK_TRANSFORMERS_DIGEST and description["settings"]["settings_id"] == "stock-transformers-v1"
    assert description["settings"]["frozen"] == frozen.FROZEN_DECODING["transformers"]
    bad = dict(decoder.observed, attn_implementation="sdpa")
    with pytest.raises(DecodingSettingsError, match="attn_implementation"):
        decoder._verify_frozen(bad)
    assert decoder.identity()["frozen_digest"] == STOCK_TRANSFORMERS_DIGEST


def test_the_stock_vllm_decoder_still_describes_itself_with_the_stock_digest(ngram_env):
    description = vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info()).describe()
    assert description["settings_id"] == "stock-vllm-package-v1" and description["frozen_digest"] == STOCK_VLLM_DIGEST
    assert description["sampling"]["package_requested"] == description["sampling"]["effective_requested"] == REQUESTED


def test_both_modules_use_their_explicit_settings_ids_and_no_legacy_name():
    import inspect
    for module in (td, vd):
        source = inspect.getsource(module)
        assert 'check_frozen(BACKEND' not in source and 'frozen_digest(BACKEND' not in source and 'settings_record(BACKEND' not in source
    assert td.SETTINGS_ID == "stock-transformers-v1" and vd.SETTINGS_ID == "stock-vllm-package-v1"


# ---- selection: file < environment < CLI; the candidate only by name; identities differ -------------------------------------------------------------------------------------

def load(path=None, **kwargs):
    kwargs.setdefault("env", {})
    return cfg.load_runtime_config(path, **kwargs)


def test_the_default_config_selects_the_stock_transformers_settings():
    assert load().data["decoder"] == {"backend": "transformers", "settings_id": "stock-transformers-v1"}


def test_a_backend_override_alone_selects_the_stock_vllm_settings_never_the_candidate():
    for chosen in (load(env={"MERALION_BACKEND": "vllm"}), load(cli={"backend": "vllm"})):
        assert chosen.data["decoder"] == {"backend": "vllm", "settings_id": "stock-vllm-package-v1"}


def test_the_candidate_config_file_selects_the_candidate_and_names_its_own_config_id():
    effective = load(cfg.CANDIDATE_RP105_CONFIG)
    assert effective.data["decoder"] == {"backend": "vllm", "settings_id": CANDIDATE} and effective.data["config_id"].endswith("vllm-rp105")
    default = load()
    assert effective.fingerprint != default.fingerprint and effective.data["strategy"] == default.data["strategy"] and effective.formatter == default.formatter
    assert {k for k in effective.data if effective.data[k] != default.data[k]} == {"config_id", "decoder"}


def test_the_environment_and_the_cli_select_the_candidate_by_name_with_the_usual_precedence():
    by_env = load(env={"MERALION_BACKEND": "vllm", "MERALION_DECODER_SETTINGS": CANDIDATE})
    by_cli = load(env={"MERALION_BACKEND": "vllm"}, cli={"decoder_settings": CANDIDATE})
    assert by_env.data["decoder"] == by_cli.data["decoder"] == {"backend": "vllm", "settings_id": CANDIDATE} and by_env.fingerprint == by_cli.fingerprint
    cli_wins = load(env={"MERALION_BACKEND": "vllm", "MERALION_DECODER_SETTINGS": CANDIDATE}, cli={"decoder_settings": "stock-vllm-package-v1"})
    assert cli_wins.data["decoder"]["settings_id"] == "stock-vllm-package-v1"
    file_names_it = load(cfg.CANDIDATE_RP105_CONFIG, env={"MERALION_DECODER_SETTINGS": "stock-vllm-package-v1"})
    assert file_names_it.data["decoder"]["settings_id"] == "stock-vllm-package-v1"                  # the environment overrides the file


def test_naming_the_files_own_backend_again_keeps_the_settings_the_file_names_but_switching_backend_resets_to_stock():
    assert load(cfg.CANDIDATE_RP105_CONFIG, env={"MERALION_BACKEND": "vllm"}).data["decoder"]["settings_id"] == CANDIDATE
    assert load(cfg.CANDIDATE_RP105_CONFIG, env={"MERALION_BACKEND": "transformers"}).data["decoder"] == {"backend": "transformers", "settings_id": "stock-transformers-v1"}


@pytest.mark.parametrize("env, cli", [
    ({"MERALION_DECODER_SETTINGS": CANDIDATE}, {}),                                          # the candidate with the default (Transformers) backend
    ({"MERALION_BACKEND": "transformers", "MERALION_DECODER_SETTINGS": CANDIDATE}, {}),
    ({"MERALION_BACKEND": "vllm", "MERALION_DECODER_SETTINGS": "stock-transformers-v1"}, {}),
    ({"MERALION_BACKEND": "vllm"}, {"decoder_settings": "made-up"}),
    ({"MERALION_BACKEND": "vllm", "MERALION_DECODER_SETTINGS": "vllm-rp110-v1"}, {}),
])
def test_a_pair_that_is_not_allowed_is_rejected_not_repaired(env, cli):
    with pytest.raises(cfg.ConfigError, match="contradicts"):
        load(env=env, cli=cli)


def test_the_cli_flag_and_make_decoder_pass_the_settings_id_through(monkeypatch):
    args = cli.build_parser().parse_args(["run", "--invocation-id", "i", "--job-name", "n", "--input", "x.wav", "--output-dir", "o", "--state-dir", "s", "--backend", "vllm", "--decoder-settings", CANDIDATE])
    assert cli._cli_overrides(args)["decoder_settings"] == CANDIDATE
    effective = load(cli=cli._cli_overrides(args))
    seen = {}

    class Recorder:
        def __init__(self, model_dir, settings_id=None):
            seen.update(model_dir=model_dir, settings_id=settings_id)

    monkeypatch.setattr(vd, "VllmDecoder", Recorder)
    cli.make_decoder(effective)
    assert seen["settings_id"] == CANDIDATE
    cli.make_decoder(load(env={"MERALION_BACKEND": "vllm"}))
    assert seen["settings_id"] == "stock-vllm-package-v1"


class WithSettings(Decoder):
    def __init__(self, inner, settings_id):
        self.inner, self.settings_id = inner, settings_id

    def decode(self, piece, **kwargs):
        return self.inner.decode(piece, **kwargs)

    def describe(self):
        return {**self.inner.describe(), "backend": "vllm", "settings_id": self.settings_id, "frozen_digest": frozen.frozen_digest(self.settings_id)}


@pytest.mark.parametrize("first, second", [("stock-vllm-package-v1", CANDIDATE), (CANDIDATE, "stock-vllm-package-v1")])
def test_a_state_started_under_one_vllm_settings_id_cannot_be_resumed_under_the_other(tmp_path, first, second):
    world = synthetic.World({"alpha": dict(break_at=2)})
    run(world, tmp_path, decoder=WithSettings(world.decoder(), first))
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(world, tmp_path, decoder=WithSettings(world.decoder(), second))
    assert run(world, tmp_path, decoder=WithSettings(world.decoder(), first)).exit_code == 0
