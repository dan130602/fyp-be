"""The frozen decoding settings (plan rev 2.2, D2 and D6) and the check that enforces them.

A decoder verifies the settings it is actually running with against this table when it is built (a deviation aborts with exit code 3) and records both the table and what it observed in
describe(). The Transformers and vLLM tables differ on purpose: the known differences are recorded as data (KNOWN_DIFFERENCES), not hidden.

vLLM has TWO views of its sampling parameters (D6): the settings the package REQUESTS (meralion_3_asr.prompts.VLLM_SAMPLING_PARAMS, with top_k 50) and the settings of the CONSTRUCTED
SamplingParams object, where vLLM 0.16.0 normalises greedy sampling (vllm/sampling_params.py:367-372: temperature < 1e-5 sets top_p = 1.0, top_k = 0, min_p = 0.0). Both are checked.
"""
import copy
import hashlib
import json

from meralion_worker.decoders.base import DecodingSettingsError

STOCK_TRANSFORMERS_ID = "stock-transformers-v1"
STOCK_VLLM_ID = "stock-vllm-package-v1"
CANDIDATE_VLLM_RP105_ID = "vllm-rp105-v1"

_STOCK_TABLES = {
    "transformers": {
        # the package's GENERATION_CONFIG, exactly
        "package_generation_config": {"max_new_tokens": 512, "do_sample": False, "no_repeat_ngram_size": 6},
        # contained in the loaded model's effective generation config (the model's generation_config.json, as recorded at run time)
        "model_generation_config": {"repetition_penalty": 1.05, "no_repeat_ngram_size": 6, "eos_token_id": 1, "bos_token_id": 2, "pad_token_id": 0},
        "attn_implementation": "eager",
        "batch_size": 1,
        "dtype_rule": "bfloat16 when torch.cuda.is_bf16_supported(), otherwise float32",
    },
    "vllm": {
        "requested_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 50, "max_tokens": 512, "repetition_penalty": 1.0},
        "normalised_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 512, "repetition_penalty": 1.0},
        "no_stop_configured": True,                           # no stop strings and no stop token ids
        "environment": {"MERALION_NGRAM_SIZE": "6"},
        "constructor_defaults": {"gpu_memory_utilization": 0.85, "max_model_len": 1300, "max_num_seqs": 64, "tensor_parallel_size": 1, "dtype": "bfloat16", "attention_backend": "FLASHINFER"},
    },
}

# The ONE controlled candidate of Stage 3 follow-up D: the stock vLLM settings with repetition_penalty 1.05 (the value of the Transformers baseline). The package's own request stays what it ships
# (`package_requested_sampling`, checked against the installed package so a package change is detected); `requested_sampling` is the EFFECTIVE request, built by the adapter into its own
# SamplingParams object. The package object and its files are never modified.
_CANDIDATE_VLLM_RP105 = {
    "package_requested_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 50, "max_tokens": 512, "repetition_penalty": 1.0},
    "overrides": {"repetition_penalty": 1.05},
    "requested_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 50, "max_tokens": 512, "repetition_penalty": 1.05},
    "normalised_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 512, "repetition_penalty": 1.05},
    "no_stop_configured": True,
    "environment": {"MERALION_NGRAM_SIZE": "6"},
    "constructor_defaults": {"gpu_memory_utilization": 0.85, "max_model_len": 1300, "max_num_seqs": 64, "tensor_parallel_size": 1, "dtype": "bfloat16", "attention_backend": "FLASHINFER"},
}

# Stage 3 follow-up E: the same two vLLM settings with DETERMINISTIC OFFLINE SCHEDULING as an explicitly identified execution profile. The engine core then runs inside the worker process
# (VLLM_ENABLE_V1_MULTIPROCESSING=0, `InprocClient`), which vLLM documents as making offline scheduling deterministic. The decoder sets the variable itself (a launcher cannot bypass the
# settings id, the frozen digest or the resume identity) and checks the observed engine client class. The three tables above are NOT edited: their digests are pinned by tests.
STOCK_VLLM_SCHED0_ID = "stock-vllm-package-sched0-v1"
CANDIDATE_VLLM_RP105_SCHED0_ID = "vllm-rp105-sched0-v1"
_EXECUTION_SCHED0 = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "engine_core_client": "InprocClient"}
_STOCK_VLLM_SCHED0 = {**copy.deepcopy(_STOCK_TABLES["vllm"]), "execution": dict(_EXECUTION_SCHED0)}
_CANDIDATE_VLLM_RP105_SCHED0 = {**copy.deepcopy(_CANDIDATE_VLLM_RP105), "execution": dict(_EXECUTION_SCHED0)}

# REV 3.0 part B: the FP16 profile for GPUs below compute capability 8.0 (the Singapore T4). The validated BF16 profile is REFUSED there by vLLM 0.16.0 (vllm/platforms/cuda.py:441-460), so this is a
# SEPARATE, explicitly named profile; nothing falls back to it automatically. It is the BF16 scheduling profile plus ONE constructor override (dtype float16, forwarded by
# Meralion3ASR.from_pretrained to the package's VllmBackend) and the engine values that must be observed. `constructor_defaults` stays EXACTLY what the package ships (bfloat16 ...), so a change of the
# package's own defaults is still detected; the override and the expected effective values are separate keys. FP16 on this bf16-trained Gemma2 checkpoint is NOT validated until T0/T1.
CANDIDATE_VLLM_RP105_SCHED0_FP16_ID = "vllm-rp105-sched0-fp16-v1"
_CANDIDATE_VLLM_RP105_SCHED0_FP16 = {**copy.deepcopy(_CANDIDATE_VLLM_RP105_SCHED0), "constructor_overrides": {"dtype": "float16"},
                                     "expected_effective_engine": {"dtype": "torch.float16", "max_model_len": 1300, "max_num_seqs": 64, "gpu_memory_utilization": 0.85,
                                                                   "tensor_parallel_size": 1, "attention_backend": "FLASHINFER"}}

# the tables by settings id, and which backend each belongs to
DECODING_SETTINGS = {STOCK_TRANSFORMERS_ID: _STOCK_TABLES["transformers"], STOCK_VLLM_ID: _STOCK_TABLES["vllm"], CANDIDATE_VLLM_RP105_ID: _CANDIDATE_VLLM_RP105,
                     STOCK_VLLM_SCHED0_ID: _STOCK_VLLM_SCHED0, CANDIDATE_VLLM_RP105_SCHED0_ID: _CANDIDATE_VLLM_RP105_SCHED0,
                     CANDIDATE_VLLM_RP105_SCHED0_FP16_ID: _CANDIDATE_VLLM_RP105_SCHED0_FP16}
BACKEND_OF = {STOCK_TRANSFORMERS_ID: "transformers", STOCK_VLLM_ID: "vllm", CANDIDATE_VLLM_RP105_ID: "vllm", STOCK_VLLM_SCHED0_ID: "vllm", CANDIDATE_VLLM_RP105_SCHED0_ID: "vllm",
              CANDIDATE_VLLM_RP105_SCHED0_FP16_ID: "vllm"}
LEGACY_BACKEND_NAMES = {"transformers": STOCK_TRANSFORMERS_ID, "vllm": STOCK_VLLM_ID}          # the old call form: a backend name means that backend's STOCK settings
FROZEN_DECODING = {"transformers": DECODING_SETTINGS[STOCK_TRANSFORMERS_ID], "vllm": DECODING_SETTINGS[STOCK_VLLM_ID]}          # kept for existing callers: backend-keyed, stock only


def resolve_settings_id(name):
    """A settings id passes through; the legacy backend names 'transformers' and 'vllm' resolve to their STOCK ids; anything else is an error (never a silent default)."""
    if name in DECODING_SETTINGS:
        return name
    if name in LEGACY_BACKEND_NAMES:
        return LEGACY_BACKEND_NAMES[name]
    raise DecodingSettingsError(f"no frozen decoding settings for {name!r}; known settings ids: {sorted(DECODING_SETTINGS)}; legacy backend names: {sorted(LEGACY_BACKEND_NAMES)}")

KNOWN_DIFFERENCES = {
    "vllm": [
        "repetition_penalty is 1.0 (package sampling parameters) versus 1.05 in the Transformers baseline (the model's generation_config.json)",
        "no_repeat_ngram is applied by a vLLM logits-processor plugin (MERALION_NGRAM_SIZE, default 6) versus no_repeat_ngram_size in generate()",
        "attention backend FLASHINFER versus eager",
        "the prompt is rendered without <bos> (vLLM adds it) versus the model's chat template",
        "audio features come from the plugin's own processor (not confirmed) versus the model folder's processor",
        "token counts and stop reasons are defined by vLLM (finish_reason, stop_reason) and are not directly comparable with the Transformers counts",
        "greedy sampling is normalised by vLLM 0.16.0: the requested top_k 50 becomes 0",
    ],
    "transformers": [
        "repetition_penalty is 1.05 (the model's generation_config.json) versus 1.0 in the vLLM backend",
        "no_repeat_ngram_size is applied inside generate() versus a vLLM logits-processor plugin",
        "attention is eager versus FLASHINFER in the vLLM backend",
        "the prompt comes from the model's chat template including <bos>",
        "token counts count every generated id including the trailing end-of-turn, newline and EOS",
    ],
}
KNOWN_DIFFERENCES_BY_SETTINGS = {
    CANDIDATE_VLLM_RP105_ID: [
        "repetition_penalty is 1.05, the VALUE of the Transformers baseline, but applied by vLLM's own implementation (prompt and output tokens, after the custom logits processors) and not by generate()",
        "a non-1.0 repetition penalty makes vLLM track the real output token ids itself, which also changes the data path of the no-repeat-ngram plugin (stock relies on the package's InputBatch patch)",
        "no_repeat_ngram is applied by a vLLM logits-processor plugin (MERALION_NGRAM_SIZE, 6) versus no_repeat_ngram_size in generate()",
        "attention backend FLASHINFER versus eager",
        "the prompt is rendered without <bos> (vLLM adds it) versus the model's chat template",
        "audio features come from the plugin's own processor (not confirmed) versus the model folder's processor",
        "token counts and stop reasons are defined by vLLM (finish_reason, stop_reason) and are not directly comparable with the Transformers counts",
        "greedy sampling is normalised by vLLM 0.16.0: the requested top_k 50 becomes 0",
    ],
}
SCHED0_DIFFERENCE = ("the engine core runs inside the worker process (VLLM_ENABLE_V1_MULTIPROCESSING=0, InprocClient); vLLM documents this as making offline scheduling deterministic; it does not "
                     "make output independent of batch composition or equal to Transformers")
KNOWN_DIFFERENCES_BY_SETTINGS[STOCK_VLLM_SCHED0_ID] = KNOWN_DIFFERENCES["vllm"] + [SCHED0_DIFFERENCE]
KNOWN_DIFFERENCES_BY_SETTINGS[CANDIDATE_VLLM_RP105_SCHED0_ID] = KNOWN_DIFFERENCES_BY_SETTINGS[CANDIDATE_VLLM_RP105_ID] + [SCHED0_DIFFERENCE]
FP16_DIFFERENCE = ("dtype float16 instead of the checkpoint's bfloat16 (for GPUs below compute capability 8.0, which cannot run bfloat16); FP16 may change token choices or overflow; "
                   "validated only by the quality measured on the tested GPU")
KNOWN_DIFFERENCES_BY_SETTINGS[CANDIDATE_VLLM_RP105_SCHED0_FP16_ID] = KNOWN_DIFFERENCES_BY_SETTINGS[CANDIDATE_VLLM_RP105_SCHED0_ID] + [FP16_DIFFERENCE]


def execution_of(name):
    """The execution profile a settings id requires (None for the three original ids: they run in the default multiprocess mode)."""
    return copy.deepcopy(DECODING_SETTINGS[resolve_settings_id(name)].get("execution"))


def constructor_overrides_of(name):
    """The keyword arguments the worker passes to Meralion3ASR.from_pretrained for this settings id ({} for every profile except the FP16 one: their call is exactly the package default)."""
    return copy.deepcopy(DECODING_SETTINGS[resolve_settings_id(name)].get("constructor_overrides", {}))


def effective_engine_expectation(name):
    """The engine values that must be OBSERVED in the built engine (None when the profile states none)."""
    return copy.deepcopy(DECODING_SETTINGS[resolve_settings_id(name)].get("expected_effective_engine"))


def engine_differences(name, observed_engine):
    """The deviations of an OBSERVED effective engine from the settings id's expectation (an empty list when the profile states none)."""
    expected = effective_engine_expectation(name)
    if expected is None:
        return []
    return _differences("effective_engine", expected, observed_engine if isinstance(observed_engine, dict) else {})


def frozen_settings(name):
    return copy.deepcopy(DECODING_SETTINGS[resolve_settings_id(name)])


def frozen_digest(name):
    canonical = json.dumps(DECODING_SETTINGS[resolve_settings_id(name)], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _same(a, b):
    """Strict equality: a bool never equals a number (0 is not False), containers compare element by element."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _differences(name, expected, observed):
    out = []
    for key, want in expected.items():
        got = observed.get(key, "<absent>") if isinstance(observed, dict) else "<absent>"
        if not _same(got, want):
            out.append(f"{name}.{key}: expected {want!r}, observed {got!r}")
    return out


def _check_transformers(observed, frozen):
    problems = []
    if not _same(observed.get("package_generation_config"), frozen["package_generation_config"]):
        problems.append(f"package_generation_config: expected {frozen['package_generation_config']!r}, observed {observed.get('package_generation_config')!r}")
    problems += _differences("model_generation_config", frozen["model_generation_config"], observed.get("model_generation_config") or {})
    if not _same(observed.get("attn_implementation"), frozen["attn_implementation"]):
        problems.append(f"attn_implementation: expected {frozen['attn_implementation']!r}, observed {observed.get('attn_implementation')!r}")
    if not _same(observed.get("batch_size"), frozen["batch_size"]):
        problems.append(f"batch_size: expected {frozen['batch_size']!r}, observed {observed.get('batch_size')!r}")
    expected_dtype = "bfloat16" if observed.get("bf16_supported") else "float32"
    if str(observed.get("dtype", "")).replace("torch.", "") != expected_dtype:
        problems.append(f"dtype: expected {expected_dtype}, observed {observed.get('dtype')!r}")
    if observed.get("has_lora_parameters") is not False:
        problems.append("the model has lora_ parameters (or this could not be established): complete checkpoints only")
    return problems


def _check_vllm(observed, frozen):
    problems = []
    if "package_requested_sampling" in frozen:                                                        # a candidate: what the INSTALLED PACKAGE requests must still be what it shipped
        problems += _differences("package_requested_sampling", frozen["package_requested_sampling"], observed.get("package_requested_sampling") or {})
        if set(observed.get("package_requested_sampling") or {}) != set(frozen["package_requested_sampling"]):
            problems.append(f"package_requested_sampling keys: expected {sorted(frozen['package_requested_sampling'])}, observed {sorted(observed.get('package_requested_sampling') or {})}")
    problems += _differences("requested_sampling", frozen["requested_sampling"], observed.get("requested_sampling") or {})
    if set(observed.get("requested_sampling") or {}) != set(frozen["requested_sampling"]):
        problems.append(f"requested_sampling keys: expected {sorted(frozen['requested_sampling'])}, observed {sorted(observed.get('requested_sampling') or {})}")
    normalised = observed.get("normalised_sampling") or {}
    problems += _differences("normalised_sampling", frozen["normalised_sampling"], normalised)
    for stop_key in ("stop", "stop_token_ids"):
        if normalised.get(stop_key):
            problems.append(f"normalised_sampling.{stop_key}: a stop is configured ({normalised.get(stop_key)!r}); the frozen configuration has none")
    problems += _differences("environment", frozen["environment"], observed.get("environment") or {})
    problems += _differences("constructor_defaults", frozen["constructor_defaults"], observed.get("constructor_defaults") or {})
    if "execution" in frozen:                                                                         # an execution profile (follow-up E): the variable and the observed engine client
        problems += _differences("execution", frozen["execution"], observed.get("execution") or {})
    if "constructor_overrides" in frozen:                                                             # REV 3.0 B: exactly the overrides this profile names were passed, no more and no fewer
        if not _same(observed.get("constructor_overrides"), frozen["constructor_overrides"]):
            problems.append(f"constructor_overrides: expected {frozen['constructor_overrides']!r}, observed {observed.get('constructor_overrides')!r}")
    if "expected_effective_engine" in frozen:                                                         # and the built engine really runs with them (a BF16 or lowered-concurrency engine is a deviation)
        problems += _differences("effective_engine", frozen["expected_effective_engine"], observed.get("effective_engine") or {})
    return problems


def check_frozen(name, observed):
    """Raise DecodingSettingsError listing every deviation of `observed` from the frozen table of that settings id (or of a legacy backend name's stock settings)."""
    settings_id = resolve_settings_id(name)
    table = DECODING_SETTINGS[settings_id]
    problems = _check_transformers(observed, table) if BACKEND_OF[settings_id] == "transformers" else _check_vllm(observed, table)
    if problems:
        raise DecodingSettingsError(f"decoding settings differ from the frozen table {settings_id!r}: " + "; ".join(problems))


def settings_record(name, observed):
    """What describe() records: the frozen table, what was observed, the digest and the known differences."""
    settings_id = resolve_settings_id(name)
    differences = KNOWN_DIFFERENCES_BY_SETTINGS.get(settings_id) or KNOWN_DIFFERENCES[BACKEND_OF[settings_id]]
    return {"settings_id": settings_id, "frozen": frozen_settings(settings_id), "observed": copy.deepcopy(observed), "frozen_digest": frozen_digest(settings_id), "known_differences": list(differences)}
