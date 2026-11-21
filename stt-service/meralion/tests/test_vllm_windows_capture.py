"""tools/origami/vllm_windows.py --capture (follow-up E, section 8.5): the harness-only shim around ONE llm.generate call. No vLLM, no GPU."""
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vllm_fakes import REQUESTED, FakeLLM, FakeSamplingParams, completion

ORIGAMI = Path(__file__).resolve().parent.parent / "tools" / "origami"
spec = importlib.util.spec_from_file_location("vllm_windows", ORIGAMI / "vllm_windows.py")
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)


def requests_for(pieces, prompt="PROMPT"):
    return [{"prompt": prompt, "multi_modal_data": {"audio": (piece, 16000)}} for piece in pieces]


def test_the_shim_records_the_request_structure_and_the_completions_and_returns_the_original_outputs():
    llm = FakeLLM([completion("a", token_ids=(5, 6)), completion("<Speaker1>: b", finish_reason="length")])
    pieces = [np.arange(10, dtype=np.float32), np.ones(7, dtype=np.float32)]
    capture = windows.Capture(llm)
    capture.install()
    outputs = llm.generate(requests_for(pieces), FakeSamplingParams(**REQUESTED))
    capture.remove()
    assert [o.outputs[0].text for o in outputs] == ["a", "<Speaker1>: b"]                       # the caller still gets what the engine returned
    (record,) = capture.records
    assert [r["audio_shape"] for r in record["requests"]] == [[10], [7]] and all(r["audio_rate"] == 16000 and r["audio_dtype"] == "float32" and r["prompt"] == "PROMPT" for r in record["requests"])
    assert [r["audio_sha256"] for r in record["requests"]] == [hashlib.sha256(p.tobytes()).hexdigest() for p in pieces]
    assert record["completions"][0] == {"text": "a", "token_ids": [5, 6], "finish_reason": "stop", "stop_reason": None} and record["completions"][1]["finish_reason"] == "length"
    assert record["sampling"]["attrs"]["top_k"] == 0 and record["sampling"]["attrs"]["repetition_penalty"] == 1.0 and "audio" not in str(record).lower().replace("audio_", "")


def test_a_call_after_remove_is_not_recorded_and_the_class_method_is_back():
    llm = FakeLLM([completion("a")])
    capture = windows.Capture(llm)
    capture.install()
    assert "generate" in vars(llm)
    capture.remove()
    assert "generate" not in vars(llm)
    llm.generate(requests_for([np.zeros(3, dtype=np.float32)]), FakeSamplingParams(**REQUESTED))
    assert capture.records == [] and len(llm.calls) == 1


def test_the_piece_hash_is_that_of_the_contiguous_float32_bytes_whatever_the_layout():
    base = np.arange(20, dtype=np.float32)
    strided = base[::2]
    assert windows.piece_sha256(strided) == hashlib.sha256(np.ascontiguousarray(strided).tobytes()).hexdigest() != windows.piece_sha256(base)
    assert windows.piece_sha256(base.astype(np.float64)) == windows.piece_sha256(base)


def test_a_capture_with_a_first_pass_that_is_not_a_batch_is_refused_before_anything_heavy_runs(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "VLLM_ENABLE_V1_MULTIPROCESSING"}
    env["PYTHONPATH"] = str(ORIGAMI.parent.parent)
    result = subprocess.run([sys.executable, str(ORIGAMI / "vllm_windows.py"), "--settings-id", "stock-vllm-package-v1", "--jobs", "j", "--windows", "w", "--passes", "single,batch", "--capture",
                             str(tmp_path / "c.json"), "--out", str(tmp_path / "o.json")], capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "first pass must be 'batch'" in result.stderr
