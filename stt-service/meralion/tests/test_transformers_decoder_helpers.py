"""R5: the Transformers decoder needs torch and a GPU, so only its pure helpers and import hygiene are tested here; its equivalence to run_windows.py is shown by the live parity run (Stage 3)."""
import subprocess
import sys
from pathlib import Path

import pytest

from meralion_worker.decoders import transformers_decoder as td

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("ids, eos, limit, expected", [
    ([5, 6, 107, 108, 1], [1], 512, "end_token"),
    ([5, 6, 7], [1], 512, "other"),
    ([], [1], 512, "other"),
    ([5] * 512, [1], 512, "max_new_tokens"),
    ([5] * 511 + [1], [1], 512, "end_token"),                  # the last id is an EOS id: a normal end even at full length (run_windows.py:118 checks this first)
    ([5] * 513, [1], 512, "max_new_tokens"),
    ([1, 5], [1], 512, "other"),                               # EOS in the middle is not an end
    ([5, 2], [1, 2], 512, "end_token"),
])
def test_stop_reason_of(ids, eos, limit, expected):
    assert td.stop_reason_of(ids, eos, limit) == expected


def test_eos_ids_accept_an_int_or_a_list():
    assert td.eos_ids_of(1) == [1] and td.eos_ids_of([1, 107]) == [1, 107] and td.eos_ids_of((1,)) == [1]


def test_observed_settings_shape_and_copies():
    package = {"max_new_tokens": 512}
    observed = td.observed_settings(package_generation_config=package, model_generation_config={"a": 1}, attn_implementation="eager", batch_size=1, dtype="torch.bfloat16",
                                    bf16_supported=1, has_lora_parameters=0)
    assert observed["bf16_supported"] is True and observed["has_lora_parameters"] is False and observed["dtype"] == "torch.bfloat16"
    package["max_new_tokens"] = 1
    assert observed["package_generation_config"]["max_new_tokens"] == 512


def test_the_backend_constants():
    assert (td.BACKEND, td.SETTINGS_ID) == ("transformers", "stock-transformers-v1")


def test_importing_the_decoder_modules_loads_no_heavy_library():
    program = ("import sys; sys.path.insert(0, '.');\n"
               "import meralion_worker.decoders.transformers_decoder, meralion_worker.decoders.vllm_decoder, meralion_worker.decoders.replay_decoder, meralion_worker.decoders.frozen\n"
               "bad = [m for m in ('torch', 'transformers', 'vllm', 'meralion_3_asr', 'librosa') if m in sys.modules]\n"
               "assert not bad, bad\n")
    result = subprocess.run([sys.executable, "-c", program], cwd=PACKAGE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_constructing_the_transformers_decoder_without_the_stack_fails_cleanly():
    """On a machine without torch or the MERaLiON package the constructor must fail at its lazy import, not at module import."""
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            td.TransformersDecoder("no-such-model-dir")
    else:
        pytest.skip("torch is installed here")
