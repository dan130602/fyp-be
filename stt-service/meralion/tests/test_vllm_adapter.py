"""The vLLM adapter (plan rev 2.2, D1 and D7): the same requests, prompt, sampling parameters and text cleanup as the official backend, plus the telemetry the package discards."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from meralion_worker.decoders import vllm_decoder as vd
from meralion_worker.decoders.base import DecodeRequest

from vllm_fakes import FakeBackend, FakeSamplingParams, REQUESTED, completion, package_info

WHEEL = Path(__file__).resolve().parent.parent / "artifacts" / "meralion_3_asr-0.1.0-py3-none-any.whl"
TEXT_CASES = [
    ("hello", "hello"),
    ("  hello  ", "hello"),
    ("<Speaker1>: hello", "hello"),
    ("<Speaker1>hello", "hello"),
    ("<Speaker1> hello", "hello"),
    ("Speaker 1: hello", "hello"),
    ("<Speaker1>:   hello world", "hello world"),
    ("  Speaker 1:  spaced  ", "spaced"),
    ("", ""),
    ("   ", ""),
    ("Speaker 1:", ""),
    ("好的，谢谢您。", "好的，谢谢您。"),
    ("mid <Speaker1>: tag stays", "mid <Speaker1>: tag stays"),
    ("<Speaker1>: <Speaker1>: twice", ": twice"),                          # the package tests its three patterns in sequence on the already-trimmed text, so the second pattern also fires
]


def pieces(n=2, size=16000):
    return [np.full(size, i, dtype=np.float32) for i in range(n)]


def adapter(completions, **kwargs):
    backend = FakeBackend(completions, **kwargs)
    return vd.VllmDecoder(backend=backend, package_info=package_info()), backend


def requests_for(items, start=0):
    return [DecodeRequest("rec", start + i * 16000, start + (i + 1) * 16000, p) for i, p in enumerate(items)]


# ---- request construction and sampling parameters ------------------------------------------------------------------------------------------------------------------

def test_the_adapter_builds_the_same_requests_as_the_official_backend(ngram_env):
    decoder, backend = adapter([completion("a"), completion("b")], prompt="THE PROMPT")
    items = pieces(2)
    decoder.decode_many(requests_for(items))
    sent, params = backend.llm.calls[0]
    assert len(sent) == 2 and params is backend.sampling_params                 # the backend's own sampling object, unchanged
    for request, item in zip(sent, items):
        assert set(request) == {"prompt", "multi_modal_data"} and request["prompt"] == "THE PROMPT"
        assert set(request["multi_modal_data"]) == {"audio"}
        audio, rate = request["multi_modal_data"]["audio"]
        assert audio is item and rate == 16000


def test_a_batch_is_one_generate_call_and_results_keep_request_order(ngram_env):
    decoder, backend = adapter([completion("first"), completion("second"), completion("third")])
    results = decoder.decode_many(requests_for(pieces(3)))
    assert len(backend.llm.calls) == 1
    assert [r.text for r in results] == ["first", "second", "third"]


def test_decode_is_a_one_request_batch(ngram_env):
    decoder, backend = adapter([completion("only")])
    result = decoder.decode(pieces(1)[0], recording_id="rec", start_sample=0, end_sample=16000)
    assert result.text == "only" and len(backend.llm.calls) == 1 and len(backend.llm.calls[0][0]) == 1


def test_an_empty_batch_does_not_call_the_engine(ngram_env):
    decoder, backend = adapter([])
    assert decoder.decode_many([]) == [] and backend.llm.calls == []


def test_an_invalid_piece_is_rejected_before_the_engine_is_called(ngram_env):
    decoder, backend = adapter([completion("x")])
    with pytest.raises(ValueError):
        decoder.decode_many([DecodeRequest("rec", 0, 16000, np.zeros(16000, dtype=np.float64))])
    assert backend.llm.calls == []


def test_a_wrong_number_of_outputs_is_an_error(ngram_env):
    decoder, _ = adapter([completion("only one")])
    with pytest.raises(RuntimeError, match="outputs for 2 requests"):
        decoder.decode_many(requests_for(pieces(2)))


# ---- text cleanup ------------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", TEXT_CASES)
def test_the_text_cleanup_matches_the_official_backend(raw, expected, ngram_env):
    decoder, _ = adapter([completion(raw)])
    assert decoder.decode_many(requests_for(pieces(1)))[0].text == expected == vd.clean_text(raw)


# ---- parity with the real package code (verified wheel; skipped where the git-ignored wheel is absent) ---------------------------------------------------------------

@pytest.fixture
def real_backend_class(monkeypatch):
    if not WHEEL.is_file():
        pytest.skip("artifacts/meralion_3_asr-0.1.0-py3-none-any.whl is not present")
    before = set(sys.modules)
    monkeypatch.syspath_prepend(str(WHEEL))                                       # a pure-Python wheel can be imported from the zip
    if "librosa" not in sys.modules:
        try:
            import librosa                                                        # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "librosa", types.ModuleType("librosa"))      # the package imports librosa at load time; transcribe_chunks never uses it
    from meralion_3_asr.backends.vllm_backend import VllmBackend
    yield VllmBackend
    for name in set(sys.modules) - before:
        if name.startswith("meralion_3_asr"):
            del sys.modules[name]


@pytest.mark.parametrize("raw, _expected", TEXT_CASES)
def test_the_adapter_equals_the_real_transcribe_chunks_on_requests_and_text(raw, _expected, real_backend_class, ngram_env):
    items = pieces(1)
    sentinel = object()
    package_llm = FakeBackend([completion(raw)]).llm
    package_instance = object.__new__(real_backend_class)
    package_instance.llm, package_instance.sampling_params, package_instance._prompt = package_llm, sentinel, "THE PROMPT"
    package_texts = real_backend_class.transcribe_chunks(package_instance, items)

    decoder, backend = adapter([completion(raw)], prompt="THE PROMPT")
    result = decoder.decode_many(requests_for(items))[0]

    package_requests, package_params = package_llm.calls[0]
    adapter_requests, adapter_params = backend.llm.calls[0]
    assert package_params is sentinel and adapter_params is backend.sampling_params
    assert len(package_requests) == len(adapter_requests) == 1
    assert package_requests[0]["prompt"] == adapter_requests[0]["prompt"] == "THE PROMPT"
    package_audio, package_rate = package_requests[0]["multi_modal_data"]["audio"]
    adapter_audio, adapter_rate = adapter_requests[0]["multi_modal_data"]["audio"]
    assert package_audio is adapter_audio is items[0] and package_rate == adapter_rate == 16000
    assert set(package_requests[0]) == set(adapter_requests[0]) and set(package_requests[0]["multi_modal_data"]) == set(adapter_requests[0]["multi_modal_data"])
    assert [result.text] == package_texts


def test_the_real_package_constants_equal_the_adapters_assumptions(real_backend_class):
    """The adapter hard-codes the speaker tags and the 16 kHz rate; the package's source must agree."""
    import inspect
    source = inspect.getsource(real_backend_class.transcribe_chunks)
    for tag in vd.SPEAKER_TAGS:
        assert f'"{tag}"' in source
    assert "16000" in source and '"multi_modal_data"' in source and '"prompt"' in source
    from meralion_3_asr.prompts import VLLM_SAMPLING_PARAMS
    assert VLLM_SAMPLING_PARAMS == REQUESTED


# ---- telemetry and the stop-reason mapping (D7) -------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("finish, native_stop, stops, expected", [
    ("length", None, False, "max_new_tokens"),
    ("length", None, True, "max_new_tokens"),
    ("stop", None, False, "end_token"),
    ("stop", 107, False, "other"),                          # a configured stop token id
    ("stop", "###", False, "other"),                        # a configured stop string
    ("stop", None, True, "other"),                          # a stop is configured: an unexplained stop is not shown to be EOS
    ("stop", 0, False, "other"),                            # token id 0 is a stop reason, not "no stop reason"
    ("stop", "", False, "other"),
    ("abort", None, False, "other"),
    ("error", None, False, "other"),
    ("repetition", None, False, "other"),
    ("", None, False, "other"),
    (None, None, False, None),                              # a missing finish reason is unavailable, not a guess
])
def test_the_stop_reason_mapping(finish, native_stop, stops, expected):
    assert vd.map_stop_reason(finish, native_stop, stops) == expected


@pytest.mark.parametrize("finish, native_stop, expected", [
    ("length", None, "max_new_tokens"), ("stop", None, "end_token"), ("stop", 107, "other"), ("stop", "###", "other"), (None, None, None)])
def test_the_decoder_applies_the_mapping_and_keeps_the_native_values(finish, native_stop, expected, ngram_env):
    decoder, _ = adapter([completion("x", token_ids=range(7), finish_reason=finish, stop_reason=native_stop)])
    result = decoder.decode_many(requests_for(pieces(1)))[0]
    assert result.stop_reason == expected and result.tokens == 7
    assert result.native["finish_reason"] == finish and result.native["stop_reason"] == native_stop
    assert result.native["batch_size"] == 1


def test_a_stop_configured_on_the_backend_makes_an_unexplained_stop_other(ngram_env):
    decoder, _ = adapter([completion("x", finish_reason="stop", stop_reason=None)])
    decoder.stops_configured = True                                              # the frozen check forbids this at construction; the mapping must still be safe
    assert decoder.decode_many(requests_for(pieces(1)))[0].stop_reason == "other"


def test_missing_telemetry_is_none_never_zero(ngram_env):
    decoder, _ = adapter([completion("x", omit=("token_ids", "finish_reason", "stop_reason"))])
    result = decoder.decode_many(requests_for(pieces(1)))[0]
    assert result.tokens is None and result.stop_reason is None
    assert result.native["finish_reason"] is None and result.native["stop_reason"] is None


def test_an_empty_token_list_is_zero_tokens_not_unavailable(ngram_env):
    decoder, _ = adapter([completion("", token_ids=[])])
    assert decoder.decode_many(requests_for(pieces(1)))[0].tokens == 0


def test_the_batch_share_and_the_batch_facts_are_recorded(ngram_env):
    decoder, _ = adapter([completion("a"), completion("b"), completion("c")])
    results = decoder.decode_many(requests_for(pieces(3)))
    for result in results:
        assert result.native["batch_size"] == 3
        assert result.decode_seconds == pytest.approx(result.native["batch_seconds"] / 3)


def test_describe_says_which_telemetry_is_available(ngram_env):
    decoder, _ = adapter([])
    assert decoder.describe()["telemetry"] == {"tokens": True, "stop_reason": True, "native": True}
    assert set(decoder.identity()) == {"backend", "settings_id", "frozen_digest", "runtime"} and decoder.identity()["settings_id"] == "stock-vllm-package-v1"
    assert set(decoder.identity()["runtime"]) == {"packages", "patches"}
    assert [(entry["target"], entry["state"]) for entry in decoder.identity()["runtime"]["patches"]] == [("vllm/platforms/cuda.py", "not_installed")]          # the tracked patch is listed; vllm is not installed here


def test_the_normalised_object_is_what_the_decoder_checks(ngram_env):
    decoder, backend = adapter([])
    assert decoder.observed["normalised_sampling"]["top_k"] == 0 and decoder.observed["requested_sampling"]["top_k"] == 50
    assert isinstance(backend.sampling_params, FakeSamplingParams)


def test_the_offline_environment_is_set_before_the_package_or_model_is_touched(monkeypatch):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    real_info = package_info()

    def reading_package_info():
        seen.update({n: __import__("os").environ.get(n) for n in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")})      # read_package_info runs before the model is built
        return real_info()

    vd.VllmDecoder(backend=FakeBackend([]), package_info=reading_package_info)
    assert seen == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


# ---- Stage 3 follow-up: an aborted or failed completion is a runtime failure, never a transcript (Codex finding 2) ----------------------------------------------------------------

@pytest.mark.parametrize("finish", ["abort", "error"])
def test_an_aborted_or_failed_completion_raises_decode_error_and_no_partial_result_is_returned(finish, ngram_env):
    from meralion_worker.decoders.base import DecodeError
    decoder, backend = adapter([completion("fine", finish_reason="stop"), completion("partial text", finish_reason=finish, stop_reason=None)])
    with pytest.raises(DecodeError, match=rf"request 1 of 2.*finish_reason='{finish}'"):
        decoder.decode_many(requests_for(pieces(2)))
    assert len(backend.llm.calls) == 1                                                       # one batch was submitted; nothing was returned from it


@pytest.mark.parametrize("finish", ["stop", "length", None, "something-new"])
def test_other_finish_reasons_still_return_results_and_the_gate_judges_them(finish, ngram_env):
    decoder, _ = adapter([completion("x", finish_reason=finish)])
    assert len(decoder.decode_many(requests_for(pieces(1)))) == 1


def test_decode_error_is_the_existing_resumable_runtime_error():
    from meralion_worker.decoders.base import DecodeError
    assert issubclass(DecodeError, RuntimeError) and "resumable" in DecodeError.__doc__ and "UNSUCCESSFUL_FINISH_REASONS" in dir(vd) and vd.UNSUCCESSFUL_FINISH_REASONS == ("abort", "error")


# ---- candidate parity uses the CANDIDATE's own effective parameters, not the package's stock object (Stage 3 follow-up D) -------------------------------------------------

@pytest.mark.parametrize("raw, _expected", TEXT_CASES)
def test_the_candidate_adapter_equals_the_real_transcribe_chunks_run_with_the_candidates_own_parameters(raw, _expected, real_backend_class, ngram_env):
    items = pieces(1)
    backend = FakeBackend([completion(raw)], prompt="THE PROMPT")
    decoder = vd.VllmDecoder(backend=backend, package_info=package_info(), settings_id="vllm-rp105-v1")
    package_llm = FakeBackend([completion(raw)]).llm
    package_instance = object.__new__(real_backend_class)
    # the HARNESS gives the package's own code path the candidate's effective object (the production adapter never does): same parameters on both sides
    package_instance.llm, package_instance.sampling_params, package_instance._prompt = package_llm, decoder.sampling_params, "THE PROMPT"
    package_texts = real_backend_class.transcribe_chunks(package_instance, items)
    result = decoder.decode_many(requests_for(items))[0]
    package_requests, package_params = package_llm.calls[0]
    adapter_requests, adapter_params = backend.llm.calls[0]
    assert package_params is adapter_params is decoder.sampling_params and adapter_params.repetition_penalty == 1.05
    assert package_requests == adapter_requests and [result.text] == package_texts
    assert backend.sampling_params.repetition_penalty == 1.0                                       # the package's own object kept its stock value throughout
