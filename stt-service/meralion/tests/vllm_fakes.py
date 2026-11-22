"""Stand-ins for the parts of vLLM and the MERaLiON vLLM backend the adapter touches. No vllm, torch or meralion_3_asr is imported."""
from types import SimpleNamespace

REQUESTED = {"temperature": 0.0, "top_p": 1.0, "top_k": 50, "max_tokens": 512, "repetition_penalty": 1.0}      # meralion_3_asr.prompts.VLLM_SAMPLING_PARAMS
CONSTRUCTOR_DEFAULTS = {"gpu_memory_utilization": 0.85, "max_model_len": 1300, "max_num_seqs": 64, "tensor_parallel_size": 1, "dtype": "bfloat16", "attention_backend": "FLASHINFER"}
_SAMPLING_EPS = 1e-5


class FakeSamplingParams:
    """Applies the normalisation of vLLM 0.16.0 SamplingParams.__post_init__ (vllm/sampling_params.py:367-372): zero temperature means greedy, which resets top_p, top_k and min_p."""

    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=16, repetition_penalty=1.0, stop=None, stop_token_ids=None):
        self.temperature, self.top_p, self.top_k, self.min_p = temperature, top_p, top_k, min_p
        self.max_tokens, self.repetition_penalty = max_tokens, repetition_penalty
        self.stop = [] if stop is None else stop
        self.stop_token_ids = [] if stop_token_ids is None else stop_token_ids
        if self.temperature < _SAMPLING_EPS:
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0


def completion(text, *, token_ids=(1, 2, 3), finish_reason="stop", stop_reason=None, omit=()):
    attributes = {"text": text, "token_ids": list(token_ids), "finish_reason": finish_reason, "stop_reason": stop_reason}
    for name in omit:
        attributes.pop(name)
    return SimpleNamespace(**attributes)


class FakeLLM:
    """Records every generate call and answers with the scripted completions, in request order."""

    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    def generate(self, requests, sampling_params):
        self.calls.append((requests, sampling_params))
        return [SimpleNamespace(outputs=[c]) for c in self.completions[:len(requests)]]


class FakeBackend:
    def __init__(self, completions, sampling_params=None, prompt="PROMPT"):
        self.llm = FakeLLM(completions)
        self.sampling_params = sampling_params if sampling_params is not None else FakeSamplingParams(**REQUESTED)
        self._prompt = prompt


def package_info(requested=None, constructor_defaults=None):
    return lambda: {"requested_sampling": dict(requested if requested is not None else REQUESTED),
                    "constructor_defaults": dict(constructor_defaults if constructor_defaults is not None else CONSTRUCTOR_DEFAULTS)}
