"""The vLLM decoder: a narrow adapter around the official in-process backend (plan rev 2.2, D1, D6, D7).

The package's VllmBackend.transcribe_chunks returns text only. This adapter calls the SAME backend's own `llm.generate` with the SAME requests, prompt and sampling parameters, applies the
SAME text cleanup, and additionally keeps what vLLM returns (token ids, finish reason, stop reason). It reuses the backend's objects (backend.llm, backend.sampling_params, backend._prompt)
and never modifies, patches or subclasses the installed package.

  request   {"prompt": backend._prompt, "multi_modal_data": {"audio": (piece, 16000)}}        (vllm_backend.py:69-75)
  generate  backend.llm.generate(requests, backend.sampling_params)                           (:76)
  cleanup   text.strip(), then a leading "<Speaker1>:", "<Speaker1>" or "Speaker 1:" removed  (:79-83)

Pieces are not passed through load_audio here: the worker loads and normalises the whole recording once (as Meralion3ASR.transcribe does), so a piece is already normalised.
"""
import hashlib
import inspect
import json
import os
import time
from importlib import metadata

from meralion_worker.decoders.base import DecodeError, DecodeRequest, DecodeResult, DecodingSettingsError, Decoder, validate_piece
from meralion_worker.decoders.frozen import (BACKEND_OF, DECODING_SETTINGS, FROZEN_DECODING, check_frozen, constructor_overrides_of, execution_of, frozen_digest, resolve_settings_id,
                                             settings_record)
from meralion_worker.runtime_identity import runtime_identity

BACKEND = "vllm"
SETTINGS_ID = "stock-vllm-package-v1"
SAMPLE_RATE = 16000
SPEAKER_TAGS = ("<Speaker1>:", "<Speaker1>", "Speaker 1:")
NGRAM_VARIABLE = "MERALION_NGRAM_SIZE"
SCHEDULING_VARIABLE = "VLLM_ENABLE_V1_MULTIPROCESSING"          # "0" runs the engine core inside this process (InprocClient); unset or "1" is the default multiprocess mode (SyncMPClient)
DEFAULT_ENGINE_CLIENT = "SyncMPClient"
UNSUCCESSFUL_FINISH_REASONS = ("abort", "error")
NORMALISED_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "repetition_penalty", "stop", "stop_token_ids")


def clean_text(text):
    """vllm_backend.py:79-83."""
    text = text.strip()
    for tag in SPEAKER_TAGS:
        if text.startswith(tag):
            text = text[len(tag):].lstrip()
    return text


def build_requests(prompt, pieces):
    """vllm_backend.py:69-75."""
    return [{"prompt": prompt, "multi_modal_data": {"audio": (piece, SAMPLE_RATE)}} for piece in pieces]


def map_stop_reason(finish_reason, stop_reason, stops_configured):
    """D7. Maps vLLM's finish reason to the worker's three values; the native values are always kept next to it.

      "length"                                                   -> max_new_tokens   (a limit hit; vLLM also uses "length" for the context cap)
      "stop", stop_reason None, no stop string/token id configured -> end_token       (the EOS check, vllm/v1/core/sched/utils.py:49-52)
      "stop" with any stop_reason, or while a stop is configured -> other             (a stop token id or a stop string, or an unexplained stop)
      "abort", "error", any other string                         -> other
      a missing finish reason                                    -> None (unavailable)
    """
    if finish_reason is None:
        return None
    if finish_reason == "length":
        return "max_new_tokens"
    if finish_reason == "stop" and stop_reason is None and not stops_configured:
        return "end_token"
    return "other"


def read_package_info():
    """What the installed package requests and defaults to (imports the package, but not vLLM)."""
    from meralion_3_asr.backends.vllm_backend import VllmBackend
    from meralion_3_asr.prompts import VLLM_SAMPLING_PARAMS

    parameters = inspect.signature(VllmBackend.__init__).parameters
    wanted = FROZEN_DECODING["vllm"]["constructor_defaults"]
    return {"requested_sampling": dict(VLLM_SAMPLING_PARAMS), "constructor_defaults": {name: parameters[name].default for name in wanted if name in parameters}}


def normalised_sampling(sampling_params):
    return {key: getattr(sampling_params, key, None) for key in NORMALISED_KEYS}


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def engine_client_name(backend):
    """The class name of the engine client of a built backend (vllm/v1/engine/llm_engine.py:108 `self.engine_core = EngineCoreClient.make_client(...)`; core_client.py:75-97 chooses SyncMPClient
    or InprocClient). None when the backend has no such path (an injected test double)."""
    client = getattr(getattr(getattr(backend, "llm", None), "llm_engine", None), "engine_core", None)
    return None if client is None else type(client).__name__


def engine_effective(backend):
    """The values the BUILT engine actually runs with, read from its vllm_config (vllm/v1/engine/llm_engine.py:64-67 `self.vllm_config`; config/vllm.py model_config, scheduler_config, cache_config,
    parallel_config, attention_config). A value that cannot be read is None, never 0; the whole result is None when the backend has no such path (an injected test double).

    `attention_backend` is the REQUESTED backend (config/attention.py: backend, None means automatic); the one actually selected is also in the engine log, which the smoke tool records.
    `num_gpu_blocks` and `block_size` are capacity facts: recorded, never gated."""
    config = getattr(getattr(getattr(backend, "llm", None), "llm_engine", None), "vllm_config", None)
    if config is None:
        return None

    def pick(*path):
        value = config
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                return None
        return value

    dtype = pick("model_config", "dtype")
    requested_backend = pick("attention_config", "backend")
    return {"dtype": None if dtype is None else str(dtype), "max_model_len": pick("model_config", "max_model_len"), "max_num_seqs": pick("scheduler_config", "max_num_seqs"),
            "gpu_memory_utilization": pick("cache_config", "gpu_memory_utilization"), "tensor_parallel_size": pick("parallel_config", "tensor_parallel_size"),
            "attention_backend": None if requested_backend is None else getattr(requested_backend, "name", str(requested_backend)),
            "num_gpu_blocks": pick("cache_config", "num_gpu_blocks"), "block_size": pick("cache_config", "block_size")}


def plan_execution(settings_id, launcher, diagnostic_in_process=False):
    """What the settings id requires of the execution mode, before anything heavy is imported: returns (mode, expected engine client). Raises DecodingSettingsError for a conflict.

    A default profile (no `execution` entry in its frozen table) runs in the default multiprocess mode: the launcher must not set the variable to anything but "1". Only a DIAGNOSTIC construction
    (the plugin probe, which must inspect the engine in process) may use such a profile with the variable "0". A scheduling profile sets the variable itself and refuses a conflicting launcher."""
    execution = execution_of(settings_id)
    if execution is None:
        if diagnostic_in_process:
            if launcher != "0":
                raise DecodingSettingsError(f"a diagnostic in-process construction of {settings_id!r} needs {SCHEDULING_VARIABLE}=0 on its command line (found {launcher!r})")
            return "diagnostic-in-process", "InprocClient"
        if launcher not in (None, "1"):
            raise DecodingSettingsError(f"settings {settings_id!r} are a default-profile configuration and never run with {SCHEDULING_VARIABLE}={launcher!r}")
        return "default", DEFAULT_ENGINE_CLIENT
    if diagnostic_in_process:
        raise DecodingSettingsError(f"settings {settings_id!r} are an execution profile, not a diagnostic: it sets {SCHEDULING_VARIABLE} itself")
    if launcher not in (None, execution[SCHEDULING_VARIABLE]):
        raise DecodingSettingsError(f"settings {settings_id!r} set {SCHEDULING_VARIABLE}={execution[SCHEDULING_VARIABLE]!r} themselves; the launcher set {launcher!r}")
    return "profile-in-process", execution["engine_core_client"]


class VllmDecoder(Decoder):
    def __init__(self, model_dir=None, *, settings_id=SETTINGS_ID, backend=None, package_info=None, diagnostic_in_process=False):
        """settings_id selects the frozen table: the stock package settings (default), a named candidate or a scheduling profile. backend / package_info are injectable for tests; in production both come
        from the installed package. The package's own `backend.sampling_params` object and files are never modified: a candidate generates with an adapter-owned SamplingParams.

        diagnostic_in_process is for the plugin probe ONLY (it must inspect the engine in this process): it lets a default profile run with VLLM_ENABLE_V1_MULTIPROCESSING=0 and stamps the execution
        as a diagnostic. cli.make_decoder never passes it, so production construction of a default profile with the variable set stays an error."""
        self.settings_id = resolve_settings_id(settings_id)
        if BACKEND_OF[self.settings_id] != BACKEND:
            raise DecodingSettingsError(f"settings {self.settings_id!r} belong to the {BACKEND_OF[self.settings_id]!r} backend, not {BACKEND!r}")
        table = DECODING_SETTINGS[self.settings_id]
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")           # before any package or model initialisation, exactly as TransformersDecoder does
        configured = os.environ.setdefault(NGRAM_VARIABLE, table["environment"][NGRAM_VARIABLE])
        if configured != table["environment"][NGRAM_VARIABLE]:
            raise DecodingSettingsError(f"{NGRAM_VARIABLE} is {configured!r}; the frozen value is {table['environment'][NGRAM_VARIABLE]!r}")
        execution_mode, expected_client = plan_execution(self.settings_id, os.environ.get(SCHEDULING_VARIABLE), diagnostic_in_process)       # before the package or vLLM is touched
        if "execution" in table:                                                                  # an execution profile owns the variable (the frozen table, not the launcher, decides)
            os.environ[SCHEDULING_VARIABLE] = table["execution"][SCHEDULING_VARIABLE]
        info = (package_info or read_package_info)()
        self.constructor_overrides = constructor_overrides_of(self.settings_id)
        self.package_constructor_defaults = dict(info["constructor_defaults"])
        started = time.monotonic()
        injected = backend is not None
        if backend is None:
            from meralion_3_asr import Meralion3ASR
            backend = Meralion3ASR.from_pretrained(str(model_dir), backend="vllm", **self.constructor_overrides)._backend
        self.load_seconds = time.monotonic() - started
        self.backend = backend
        observed_client = engine_client_name(backend)
        if observed_client is None and not injected:
            raise DecodingSettingsError(f"the engine client of the built vLLM backend cannot be observed; expected {expected_client}")
        if observed_client is not None and observed_client != expected_client:
            raise DecodingSettingsError(f"execution mode {execution_mode!r} of settings {self.settings_id!r} expects the engine client {expected_client}, observed {observed_client}")
        self.execution = {"mode": execution_mode, "variable": os.environ.get(SCHEDULING_VARIABLE), "engine_core_client": observed_client, "expected_engine_core_client": expected_client}
        self.package_requested_sampling = dict(info["requested_sampling"])
        if "overrides" in table:                                                                  # a candidate: the package's request plus the named overrides, in an adapter-owned object
            self.effective_requested_sampling = {**self.package_requested_sampling, **table["overrides"]}
            self.sampling_params = type(backend.sampling_params)(**self.effective_requested_sampling)
        else:                                                                                     # stock: exactly the package's own object, as before
            self.effective_requested_sampling = dict(self.package_requested_sampling)
            self.sampling_params = backend.sampling_params
        self.observed = {"requested_sampling": self.effective_requested_sampling, "normalised_sampling": normalised_sampling(self.sampling_params),
                         "environment": {NGRAM_VARIABLE: os.environ.get(NGRAM_VARIABLE)}, "constructor_defaults": info["constructor_defaults"]}
        if "package_requested_sampling" in table:
            self.observed["package_requested_sampling"] = self.package_requested_sampling
        if "execution" in table:
            self.observed["execution"] = {SCHEDULING_VARIABLE: os.environ.get(SCHEDULING_VARIABLE), "engine_core_client": observed_client}
        self.engine_effective = engine_effective(backend)
        if "constructor_overrides" in table:                                                      # what was passed to the package constructor (never guessed from the table alone)
            self.observed["constructor_overrides"] = dict(self.constructor_overrides)
        if "expected_effective_engine" in table:                                                  # the engine must be observable and must run with the expected values
            if self.engine_effective is None and not injected:
                raise DecodingSettingsError(f"the effective engine settings of the built vLLM backend cannot be observed; settings {self.settings_id!r} require {table['expected_effective_engine']}")
            self.observed["effective_engine"] = self.engine_effective or {}
        check_frozen(self.settings_id, self.observed)
        self.stops_configured = bool(self.observed["normalised_sampling"].get("stop")) or bool(self.observed["normalised_sampling"].get("stop_token_ids"))

    def decode_many(self, requests):
        for request in requests:
            validate_piece(request.piece, request.start_sample, request.end_sample)
        if not requests:
            return []
        started = time.monotonic()
        outputs = self.backend.llm.generate(build_requests(self.backend._prompt, [r.piece for r in requests]), self.sampling_params)
        batch_seconds = time.monotonic() - started
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} requests")
        for index, output in enumerate(outputs):                                                # an aborted or failed completion is a runtime failure, never a transcript
            reason = getattr(output.outputs[0], "finish_reason", None)
            if reason in UNSUCCESSFUL_FINISH_REASONS:
                raise DecodeError(f"vLLM did not complete request {index} of {len(outputs)}: finish_reason={reason!r}, stop_reason={getattr(output.outputs[0], 'stop_reason', None)!r}")
        results = []
        for output in outputs:
            completion = output.outputs[0]
            token_ids = getattr(completion, "token_ids", None)
            finish_reason = getattr(completion, "finish_reason", None)
            native_stop = getattr(completion, "stop_reason", None)
            results.append(DecodeResult(
                text=clean_text(completion.text),
                tokens=None if token_ids is None else len(token_ids),
                stop_reason=map_stop_reason(finish_reason, native_stop, self.stops_configured),
                decode_seconds=batch_seconds / len(requests),                        # an amortised share of the batch, not a measured per-window latency
                native={"finish_reason": finish_reason, "stop_reason": native_stop, "batch_size": len(requests), "batch_seconds": batch_seconds,
                        "token_ids_sha256": None if token_ids is None else hashlib.sha256(json.dumps([int(i) for i in token_ids]).encode("utf-8")).hexdigest()}))
        return results

    def decode(self, piece, *, recording_id, start_sample, end_sample):
        return self.decode_many([DecodeRequest(recording_id, start_sample, end_sample, piece)])[0]

    def describe(self):
        try:
            import torch
            device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:                                                                 # the adapter must stay describable where torch is not importable (tests)
            device = None
        return {
            "backend": BACKEND, "settings_id": self.settings_id, "frozen_digest": frozen_digest(self.settings_id), "runtime": runtime_identity(),
            "versions": {"vllm": _version("vllm"), "meralion_3_asr": _version("meralion-3-asr"), "torch": _version("torch")},
            "device": device, "model_load_seconds": round(self.load_seconds, 3),
            "settings": settings_record(self.settings_id, self.observed),
            "sampling": {"package_requested": self.package_requested_sampling, "effective_requested": self.effective_requested_sampling,
                         "effective_normalised": self.observed["normalised_sampling"], "effective_object_repr": repr(self.sampling_params), "package_object_repr": repr(self.backend.sampling_params)},
            "stops_configured": self.stops_configured, "execution": dict(self.execution),
            "constructor_overrides": dict(self.constructor_overrides), "package_constructor_defaults": dict(self.package_constructor_defaults), "engine_effective": self.engine_effective,
            "telemetry": {"tokens": True, "stop_reason": True, "native": True},
        }
