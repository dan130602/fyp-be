"""The Transformers decoder: the validated decode path, copied from stt-experiments/meralion-runner/run_windows.py:82-121 (git blob 25f392b0...).

    load   Meralion3ASR.from_pretrained(dir, backend="transformers", device="cuda", batch_size=1, dtype=bf16 if supported else fp32); requires_grad_(False); eval(); no lora_ parameter
    decode load_audio(piece) -> processor(text=[backend._prompt], audios=[wav], padding=True) -> floating inputs cast to backend.dtype
           -> hf.generate(**inputs, **GENERATION_CONFIG, return_dict_in_generate=True, output_logits=True) under inference_mode
           -> backend._postprocess(tokenizer.decode(new_ids, skip_special_tokens=True))
Batch size 1, sequential; never Meralion3ASR.transcribe() (it chunks internally). torch, transformers and meralion_3_asr are imported inside the constructor, so importing this module is cheap.

Whether this decodes exactly like run_windows.py is shown by the live parity run (Stage 3); the CPU tests cover the pure helpers only.
"""
import inspect
import os
import time
from importlib import metadata
from pathlib import Path

from meralion_worker.decoders.base import DecodeResult, DecodingSettingsError, Decoder, validate_piece
from meralion_worker.decoders.frozen import check_frozen, frozen_digest, settings_record
from meralion_worker.runtime_identity import runtime_identity

BACKEND = "transformers"
SETTINGS_ID = "stock-transformers-v1"


def stop_reason_of(ids, eos_ids, limit):
    """run_windows.py:118: the last generated id is an EOS id -> end_token; otherwise a full-length output -> max_new_tokens; otherwise other."""
    if ids and ids[-1] in eos_ids:
        return "end_token"
    if len(ids) >= limit:
        return "max_new_tokens"
    return "other"


def eos_ids_of(eos_token_id):
    return [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)


def observed_settings(*, package_generation_config, model_generation_config, attn_implementation, batch_size, dtype, bf16_supported, has_lora_parameters):
    return {"package_generation_config": dict(package_generation_config), "model_generation_config": dict(model_generation_config), "attn_implementation": attn_implementation,
            "batch_size": batch_size, "dtype": str(dtype), "bf16_supported": bool(bf16_supported), "has_lora_parameters": bool(has_lora_parameters)}


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class TransformersDecoder(Decoder):
    def __init__(self, model_dir):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        import torch
        from meralion_3_asr import Meralion3ASR
        from meralion_3_asr._audio_io import load_audio
        from meralion_3_asr.backends.transformers_backend import TransformersBackend
        from meralion_3_asr.prompts import GENERATION_CONFIG

        self._torch, self._load_audio, self._generation_config = torch, load_audio, dict(GENERATION_CONFIG)
        directory = Path(model_dir).resolve()
        if (directory / "adapter_config.json").exists():
            raise DecodingSettingsError("adapter_config.json found: complete checkpoints only")
        if not torch.cuda.is_available():
            raise DecodingSettingsError("a GPU is required for the Transformers decoder")
        bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float32
        started = time.monotonic()
        self.model = Meralion3ASR.from_pretrained(str(directory), backend="transformers", device="cuda", batch_size=1, dtype=dtype)
        self.load_seconds = time.monotonic() - started
        self.backend = self.model._backend
        self.hf, self.processor, self.tokenizer = self.backend.model, self.backend.processor, self.backend.processor.tokenizer
        self.hf.requires_grad_(False)
        self.hf.eval()
        self.observed = observed_settings(
            package_generation_config=GENERATION_CONFIG, model_generation_config=self.hf.generation_config.to_dict(),
            attn_implementation=inspect.signature(TransformersBackend.__init__).parameters["attn_implementation"].default,       # not overridden: the package default is in force
            batch_size=1, dtype=dtype, bf16_supported=bf16, has_lora_parameters=any("lora_" in name for name, _ in self.hf.named_parameters()))
        self._verify_frozen(self.observed)                            # before the first decode
        self.eos_ids = eos_ids_of(self.hf.generation_config.eos_token_id)
        self.limit = self._generation_config["max_new_tokens"]

    def _verify_frozen(self, observed):
        """The frozen-settings check under the explicit settings id (not the legacy backend name)."""
        check_frozen(SETTINGS_ID, observed)

    def decode(self, piece, *, recording_id, start_sample, end_sample):
        validate_piece(piece, start_sample, end_sample)
        torch, hf, backend = self._torch, self.hf, self.backend
        started = time.monotonic()
        with torch.inference_mode():
            wav = self._load_audio(piece)
            inputs = self.processor(text=[backend._prompt], audios=[wav], return_tensors="pt", padding=True).to(backend.device)
            for key, value in inputs.items():
                if value.dtype.is_floating_point:
                    inputs[key] = value.to(backend.dtype)
            prompt_length = int(inputs["input_ids"].shape[1])
            out = hf.generate(**inputs, **self._generation_config, return_dict_in_generate=True, output_logits=True)
            ids = [int(t) for t in out.sequences[0][prompt_length:]]
            text = backend._postprocess(self.tokenizer.decode(ids, skip_special_tokens=True))
            del out
        return DecodeResult(text=text, tokens=len(ids), stop_reason=stop_reason_of(ids, self.eos_ids, self.limit), decode_seconds=time.monotonic() - started, native={})

    def decode_many(self, requests):
        results = super().decode_many(requests)
        self._torch.cuda.empty_cache()
        return results

    def describe(self):
        torch = self._torch
        return {
            "backend": BACKEND, "settings_id": SETTINGS_ID, "frozen_digest": frozen_digest(SETTINGS_ID), "runtime": runtime_identity(),
            "versions": {"torch": torch.__version__, "transformers": _version("transformers"), "meralion_3_asr": _version("meralion-3-asr")},
            "device": torch.cuda.get_device_name(0), "dtype": self.observed["dtype"], "model_load_seconds": round(self.load_seconds, 3),
            "settings": settings_record(SETTINGS_ID, self.observed),
            "telemetry": {"tokens": True, "stop_reason": True, "native": False},
        }
