import copy

import pytest

from meralion_worker.decoders import frozen
from meralion_worker.decoders import transformers_decoder as td
from meralion_worker.decoders import vllm_decoder as vd
from meralion_worker.decoders.base import DecodingSettingsError

from vllm_fakes import CONSTRUCTOR_DEFAULTS, REQUESTED, FakeBackend, FakeSamplingParams, package_info

TF_MODEL_CONFIG = {"_from_model_config": True, "bos_token_id": 2, "cache_implementation": "hybrid", "eos_token_id": 1, "no_repeat_ngram_size": 6, "pad_token_id": 0,
                   "repetition_penalty": 1.05, "transformers_version": "4.50.1"}            # the model's generation_config.json, as recorded at run time


def tf_observed(**changes):
    observed = td.observed_settings(package_generation_config={"max_new_tokens": 512, "do_sample": False, "no_repeat_ngram_size": 6}, model_generation_config=TF_MODEL_CONFIG,
                                    attn_implementation="eager", batch_size=1, dtype="torch.bfloat16", bf16_supported=True, has_lora_parameters=False)
    observed.update(changes)
    return observed


def vllm_observed(requested=None, normalised=None, environment=None, constructor=None):
    sampling = FakeSamplingParams(**(requested or REQUESTED))
    return {"requested_sampling": dict(requested or REQUESTED), "normalised_sampling": vd.normalised_sampling(sampling) if normalised is None else normalised,
            "environment": environment or {"MERALION_NGRAM_SIZE": "6"}, "constructor_defaults": constructor or dict(CONSTRUCTOR_DEFAULTS)}


def test_the_expected_observed_values_pass_for_each_backend():
    frozen.check_frozen("transformers", tf_observed())
    frozen.check_frozen("vllm", vllm_observed())
    frozen.check_frozen("transformers", tf_observed(dtype="torch.float32", bf16_supported=False))          # fp32 is the rule when bf16 is not supported


def test_the_requested_and_the_normalised_vllm_views_differ_and_both_are_accepted():
    """D6 regression: the package requests top_k 50, vLLM 0.16.0 normalises greedy sampling to top_k 0, and the constructed object must be accepted."""
    observed = vllm_observed()
    assert observed["requested_sampling"]["top_k"] == 50 and observed["normalised_sampling"]["top_k"] == 0
    assert observed["normalised_sampling"] == {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 512, "repetition_penalty": 1.0, "stop": [], "stop_token_ids": []}
    frozen.check_frozen("vllm", observed)


def test_the_stand_in_applies_the_greedy_normalisation_only_at_zero_temperature():
    assert FakeSamplingParams(temperature=0.0, top_k=50).top_k == 0
    assert FakeSamplingParams(temperature=0.7, top_k=50).top_k == 50


@pytest.mark.parametrize("changed", [{"temperature": 0.5}, {"top_k": 40}, {"max_tokens": 256}, {"repetition_penalty": 1.05}, {"top_p": 0.9}])
def test_a_changed_requested_package_setting_is_rejected(changed):
    with pytest.raises(DecodingSettingsError, match="requested_sampling"):
        frozen.check_frozen("vllm", vllm_observed(requested={**REQUESTED, **changed}))


def test_a_requested_dict_with_an_extra_or_missing_key_is_rejected():
    with pytest.raises(DecodingSettingsError, match="requested_sampling keys"):
        frozen.check_frozen("vllm", vllm_observed(requested={**REQUESTED, "min_p": 0.1}))
    smaller = dict(REQUESTED)
    del smaller["top_k"]
    with pytest.raises(DecodingSettingsError, match="requested_sampling"):
        frozen.check_frozen("vllm", vllm_observed(requested=smaller))


def test_an_unnormalised_top_k_of_50_on_the_constructed_object_is_rejected():
    bad = vllm_observed()
    bad["normalised_sampling"]["top_k"] = 50
    with pytest.raises(DecodingSettingsError, match=r"normalised_sampling\.top_k"):
        frozen.check_frozen("vllm", bad)


@pytest.mark.parametrize("key, value", [("stop", ["</s>"]), ("stop_token_ids", [107])])
def test_a_configured_stop_is_rejected(key, value):
    bad = vllm_observed()
    bad["normalised_sampling"][key] = value
    with pytest.raises(DecodingSettingsError, match="a stop is configured"):
        frozen.check_frozen("vllm", bad)


@pytest.mark.parametrize("path, value, text", [
    (("normalised_sampling", "repetition_penalty"), 1.05, "normalised_sampling.repetition_penalty"),
    (("normalised_sampling", "max_tokens"), 600, "normalised_sampling.max_tokens"),
    (("normalised_sampling", "temperature"), 0.3, "normalised_sampling.temperature"),
    (("environment", "MERALION_NGRAM_SIZE"), "5", "environment.MERALION_NGRAM_SIZE"),
    (("environment", "MERALION_NGRAM_SIZE"), None, "environment.MERALION_NGRAM_SIZE"),
    (("constructor_defaults", "max_model_len"), 2048, "constructor_defaults.max_model_len"),
    (("constructor_defaults", "attention_backend"), "FLASH_ATTN", "constructor_defaults.attention_backend"),
    (("constructor_defaults", "gpu_memory_utilization"), 0.9, "constructor_defaults.gpu_memory_utilization"),
])
def test_each_vllm_deviation_is_rejected_and_named(path, value, text):
    bad = copy.deepcopy(vllm_observed())
    bad[path[0]][path[1]] = value
    with pytest.raises(DecodingSettingsError, match=text.replace(".", r"\.")):
        frozen.check_frozen("vllm", bad)


@pytest.mark.parametrize("changes, text", [
    ({"package_generation_config": {"max_new_tokens": 256, "do_sample": False, "no_repeat_ngram_size": 6}}, "package_generation_config"),
    ({"package_generation_config": {"max_new_tokens": 512, "do_sample": True, "no_repeat_ngram_size": 6}}, "package_generation_config"),
    ({"package_generation_config": {"max_new_tokens": 512, "do_sample": 0, "no_repeat_ngram_size": 6}}, "package_generation_config"),     # 0 is not False
    ({"package_generation_config": {"max_new_tokens": 512, "do_sample": False, "no_repeat_ngram_size": 3}}, "package_generation_config"),
    ({"model_generation_config": {**TF_MODEL_CONFIG, "repetition_penalty": 1.0}}, "model_generation_config.repetition_penalty"),
    ({"model_generation_config": {**TF_MODEL_CONFIG, "no_repeat_ngram_size": 0}}, "model_generation_config.no_repeat_ngram_size"),
    ({"model_generation_config": {**TF_MODEL_CONFIG, "eos_token_id": 107}}, "model_generation_config.eos_token_id"),
    ({"model_generation_config": {k: v for k, v in TF_MODEL_CONFIG.items() if k != "repetition_penalty"}}, "model_generation_config.repetition_penalty"),
    ({"attn_implementation": "sdpa"}, "attn_implementation"),
    ({"batch_size": 4}, "batch_size"),
    ({"dtype": "torch.float16"}, "dtype"),
    ({"dtype": "torch.float32"}, "dtype"),                                     # fp32 while bf16 IS supported
    ({"has_lora_parameters": True}, "lora_"),
    ({"has_lora_parameters": None}, "lora_"),
])
def test_each_transformers_deviation_is_rejected_and_named(changes, text):
    with pytest.raises(DecodingSettingsError, match=text.replace(".", r"\.")):
        frozen.check_frozen("transformers", tf_observed(**changes))


def test_every_deviation_is_listed_not_only_the_first():
    bad = tf_observed(attn_implementation="sdpa", batch_size=4)
    with pytest.raises(DecodingSettingsError) as info:
        frozen.check_frozen("transformers", bad)
    assert "attn_implementation" in str(info.value) and "batch_size" in str(info.value)


def test_an_unknown_backend_is_rejected():
    with pytest.raises(DecodingSettingsError, match="no frozen decoding settings"):
        frozen.check_frozen("onnx", {})


# ---- the vLLM decoder enforces the table when it is built ------------------------------------------------------------------------------------------------------------

def test_the_ngram_variable_is_set_to_six_when_unset(ngram_env):
    import os
    assert vd.NGRAM_VARIABLE not in os.environ
    decoder = vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info())
    assert os.environ[vd.NGRAM_VARIABLE] == "6"
    assert decoder.observed["environment"] == {"MERALION_NGRAM_SIZE": "6"}


def test_the_ngram_variable_is_rejected_when_set_to_anything_else(ngram_env):
    ngram_env.setenv(vd.NGRAM_VARIABLE, "5")
    with pytest.raises(DecodingSettingsError, match="MERALION_NGRAM_SIZE"):
        vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info())


def test_the_vllm_decoder_accepts_the_normalised_object_and_rejects_a_changed_package(ngram_env):
    vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info())
    with pytest.raises(DecodingSettingsError, match="requested_sampling"):
        vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info(requested={**REQUESTED, "top_k": 40}))
    with pytest.raises(DecodingSettingsError, match="constructor_defaults"):
        vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info(constructor_defaults={**CONSTRUCTOR_DEFAULTS, "max_model_len": 4096}))
    with pytest.raises(DecodingSettingsError, match="a stop is configured"):
        vd.VllmDecoder(backend=FakeBackend([], sampling_params=FakeSamplingParams(**REQUESTED, stop=["x"])), package_info=package_info())


def test_describe_records_both_views_and_the_known_differences(ngram_env):
    description = vd.VllmDecoder(backend=FakeBackend([]), package_info=package_info()).describe()
    settings = description["settings"]
    assert settings["observed"]["requested_sampling"]["top_k"] == 50 and settings["observed"]["normalised_sampling"]["top_k"] == 0
    assert settings["observed"]["environment"] == {"MERALION_NGRAM_SIZE": "6"}
    assert settings["frozen_digest"] == description["frozen_digest"] == frozen.frozen_digest("vllm")
    joined = " ".join(settings["known_differences"])
    assert "repetition_penalty is 1.0" in joined and "1.05" in joined                   # the Transformers/vLLM difference is explicit
    assert description["backend"] == "vllm" and description["settings_id"] == "stock-vllm-package-v1"
    import json
    json.dumps(description)                                                              # recordable


def test_the_digest_is_stable_and_differs_per_backend():
    assert frozen.frozen_digest("vllm") == frozen.frozen_digest("vllm") and len(frozen.frozen_digest("vllm")) == 64
    assert frozen.frozen_digest("vllm") != frozen.frozen_digest("transformers")


def test_the_known_differences_state_the_repetition_penalty_both_ways():
    assert any("1.0" in d and "1.05" in d for d in frozen.KNOWN_DIFFERENCES["vllm"])
    assert any("1.05" in d and "1.0" in d for d in frozen.KNOWN_DIFFERENCES["transformers"])
