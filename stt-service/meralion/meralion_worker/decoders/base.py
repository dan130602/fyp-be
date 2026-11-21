"""The decoder contract.

A decoder turns one mono 16 kHz float32 window (0 < length <= 30 s) into text and reports what it can about how decoding ended. Windows are independent, so decode_many may batch.
A field a backend cannot provide is None ("unavailable"), never 0.
"""
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
MAX_WINDOW_SAMPLES = 30 * SAMPLE_RATE
STOP_REASONS = ("end_token", "max_new_tokens", "other")
IDENTITY_KEYS = ("backend", "settings_id", "frozen_digest", "runtime")


class DecodeError(RuntimeError):
    """Decoding failed at run time (exit code 1; the invocation is resumable)."""


class DecodingSettingsError(ValueError):
    """The decoder cannot be set up exactly as frozen (exit code 3): a setting deviates, a GPU or the base checkpoint is missing."""


@dataclass(frozen=True)
class DecodeResult:
    text: str
    tokens: int | None
    stop_reason: str | None            # "end_token" | "max_new_tokens" | "other" | None (unavailable)
    decode_seconds: float
    native: dict = field(default_factory=dict)      # the backend's own termination information ({} where there is none)

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError("DecodeResult.text must be a string")
        if self.tokens is not None and (isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 0):
            raise ValueError("DecodeResult.tokens must be a non-negative integer or None")
        if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
            raise ValueError(f"DecodeResult.stop_reason must be one of {STOP_REASONS} or None")
        if isinstance(self.decode_seconds, bool) or not isinstance(self.decode_seconds, numbers.Real) or self.decode_seconds < 0:
            raise ValueError("DecodeResult.decode_seconds must be a non-negative number")
        if not isinstance(self.native, dict):
            raise ValueError("DecodeResult.native must be a dict")

    def to_dict(self):
        return {"text": self.text, "tokens": self.tokens, "stop_reason": self.stop_reason, "decode_seconds": self.decode_seconds, "native": dict(self.native)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"text", "tokens", "stop_reason", "decode_seconds", "native"}:
            raise ValueError("a DecodeResult dict needs exactly: text, tokens, stop_reason, decode_seconds, native")
        return cls(text=data["text"], tokens=data["tokens"], stop_reason=data["stop_reason"], decode_seconds=data["decode_seconds"], native=dict(data["native"]))


@dataclass(frozen=True)
class DecodeRequest:
    recording_id: str
    start_sample: int
    end_sample: int
    piece: np.ndarray


def validate_piece(piece, start_sample, end_sample):
    if not isinstance(piece, np.ndarray) or piece.ndim != 1 or piece.dtype != np.float32:
        raise ValueError("a window must be a 1-D float32 array")
    if not 0 < len(piece) <= MAX_WINDOW_SAMPLES:
        raise ValueError(f"a window must have 1..{MAX_WINDOW_SAMPLES} samples, not {len(piece)}")
    if len(piece) != end_sample - start_sample:
        raise ValueError(f"window length {len(piece)} differs from end - start = {end_sample - start_sample}")


class Decoder(ABC):
    @abstractmethod
    def decode(self, piece, *, recording_id, start_sample, end_sample):
        """-> DecodeResult"""

    def decode_many(self, requests):
        """-> [DecodeResult] in request order. The default decodes one by one; a backend may batch."""
        return [self.decode(r.piece, recording_id=r.recording_id, start_sample=r.start_sample, end_sample=r.end_sample) for r in requests]

    @abstractmethod
    def describe(self):
        """-> a JSON-serialisable dict that includes backend, settings_id, frozen_digest and runtime (installed package versions and tracked patches: meralion_worker.runtime_identity)."""

    def identity(self):
        description = self.describe()
        return {key: description[key] for key in IDENTITY_KEYS}
