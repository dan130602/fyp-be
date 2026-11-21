"""Training-time audio augmentation (CPU, numpy + scipy, no model, no data files): the two kinds of degradation of the environment-V2 evaluation variants, but built from DIFFERENT, synthetic material and
with DIFFERENT, random settings, so the evaluation noise recordings, SNRs and filter values are never trained on.

    noise       synthetic ambience added at a random SNR in [0, 20] dB: office (pink noise + mains hum), cafe (speech-shaped babble with syllable-rate modulation + clinks), transport (rumble + swells)
    telephone   random 4th-order band-pass (300-700 Hz ... 2000-3400 Hz), 8 kHz, mu-law or A-law 8-bit companding, back to 16 kHz, about -6 dB
    muffle      random 2nd-order low-pass at 800-2000 Hz, -3 to -9 dB

No speed change, no reverberation. The transcript of a clip never changes. Every augmented copy is a pure function of (seed, clip id, epoch): a run is reproducible and a resumed run draws the same copies.
How it is used by the trainer (`--augment extras`): in every epoch a clip's FIRST draw is the clean clip (so every clip is seen clean every epoch, exactly as before); the extra draws of the matched
budget (K = 1067: second occurrences) are augmented. Nothing else about sampling, schedule or budget changes. SNR definition (as in V2): power of the ACTIVE speech (20 ms frames within 40 dB of the
loudest, after a 100 Hz high-pass) over the mean noise power; one gain per clip.
"""
import hashlib

import numpy as np
from scipy import signal

SR = 16000
SNR_RANGE_DB = (0.0, 20.0)
KIND_WEIGHTS = (("noise", 0.5), ("telephone", 0.25), ("muffle", 0.25))
NOISE_TYPES = ("office", "cafe", "transport")
TELEPHONE = {"low_hz": (300.0, 700.0), "high_hz": (2000.0, 3400.0), "gain_db": (-8.0, -4.0), "codec": ("mulaw", "alaw")}
MUFFLE = {"cutoff_hz": (800.0, 2000.0), "gain_db": (-9.0, -3.0)}
TAG = "|aug"


def describe():
    return {"module": "sft_augment", "sample_rate": SR, "snr_range_db": SNR_RANGE_DB, "kind_weights": dict(KIND_WEIGHTS), "noise_types": NOISE_TYPES, "telephone": TELEPHONE, "muffle": MUFFLE,
            "speed_perturbation": False, "reverberation": False, "applied_to": "the extra draws of the matched budget (second occurrences); the first draw of every clip in every epoch stays clean"}


# ------------------------------------------------------------------ draws
def base_id(draw):
    return draw.split(TAG, 1)[0]


def parse_draw(draw):
    """draw id -> (clip id, epoch of the augmented copy or None for a clean draw)."""
    if TAG in draw:
        clip_id, epoch = draw.split(TAG, 1)
        return clip_id, int(epoch)
    return draw, None


def is_augmented(draw):
    return TAG in draw


def mark_extras(draws, epoch):
    """The first occurrence of every clip in the epoch stays clean; every later occurrence (an extra draw) becomes an augmented copy `<clip>|aug<epoch>`. Order and length are unchanged."""
    seen, out = set(), []
    for d in draws:
        out.append(d if d not in seen else f"{d}{TAG}{epoch}")
        seen.add(d)
    return out


def rng_for(seed, clip_id, epoch):
    digest = hashlib.sha256(f"{seed}:{clip_id}:{epoch}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


# ------------------------------------------------------------------ signal helpers
def active_power(x):
    """Mean power of the active speech frames (20 ms frames within 40 dB of the loudest frame, after a 100 Hz high-pass)."""
    if len(x) > 400:
        x = signal.sosfiltfilt(signal.butter(4, 100.0, "highpass", fs=SR, output="sos"), x)
    n = max(len(x) // 320, 1)
    frames = x[: n * 320].reshape(n, -1) if len(x) >= 320 else x.reshape(1, -1)
    power = np.mean(frames ** 2, axis=1)
    keep = power >= power.max() * 1e-4
    return float(power[keep].mean()) if keep.any() and power.max() > 0 else float(np.mean(x ** 2) + 1e-20)


def _pink(n, rng):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, 1.0 / SR)
    freq[0] = freq[1] if len(freq) > 1 else 1.0
    return np.fft.irfft(spectrum / np.sqrt(freq), n)


def _band(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, min(hi, SR / 2 - 100)], "bandpass", fs=SR, output="sos"), x)


def _low(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, "lowpass", fs=SR, output="sos"), x)


def _envelope(n, rng, rate_hz):
    env = _low(np.abs(rng.standard_normal(n)), rate_hz)
    return env / (env.mean() + 1e-12)


def make_noise(kind, n, rng):
    """Synthetic ambience of n samples at 16 kHz, zero mean, unit RMS."""
    t = np.arange(n) / SR
    if kind == "office":
        body = _pink(n, rng)
        body /= body.std() + 1e-12
        hum = sum(a * np.sin(2 * np.pi * 50.0 * k * t + rng.uniform(0, 2 * np.pi)) for k, a in enumerate((1.0, 0.6, 0.3, 0.15), start=1))
        out = (body + 0.35 * hum / (hum.std() + 1e-12)) * (1.0 + 0.2 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi)))
    elif kind == "cafe":
        base = _band(_pink(n, rng), 200.0, 4000.0)
        base /= base.std() + 1e-12
        out = base * np.mean([_envelope(n, rng, 4.0) for _ in range(6)], axis=0)
        for _ in range(rng.poisson(0.4 * n / SR)):
            start, length = int(rng.integers(0, max(n - 1, 1))), int(0.08 * SR)
            k = np.arange(min(length, n - start))
            out[start:start + len(k)] += rng.uniform(0.5, 1.5) * np.exp(-k / (0.02 * SR)) * np.sin(2 * np.pi * rng.uniform(2000.0, 5000.0) * k / SR)
    elif kind == "transport":
        rumble = _low(rng.standard_normal(n), 300.0, order=4)
        rumble /= rumble.std() + 1e-12
        swell = _band(rng.standard_normal(n), 150.0, 2500.0)
        swell /= swell.std() + 1e-12
        bumps = np.full(n, 0.2)
        for _ in range(max(1, int(round(n / SR / 6)))):
            bumps += rng.uniform(0.6, 1.6) * np.exp(-0.5 * ((t - rng.uniform(0, max(n / SR, 1e-3))) / rng.uniform(0.8, 2.5)) ** 2)
        out = rumble + 0.8 * swell * bumps
    else:
        raise ValueError(f"unknown noise type {kind}")
    out = out - out.mean()
    return out / (out.std() + 1e-12)


def _compand(x, law):
    x = np.clip(x, -1.0, 1.0)
    if law == "mulaw":
        mu = 255.0
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        y = np.round((y + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
        return np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu
    a = 87.6
    ax = np.abs(x)
    y = np.sign(x) * np.where(ax < 1.0 / a, a * ax / (1.0 + np.log(a)), (1.0 + np.log(np.maximum(a * ax, 1e-12))) / (1.0 + np.log(a)))
    y = np.round((y + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
    ay = np.abs(y)
    return np.sign(y) * np.where(ay < 1.0 / (1.0 + np.log(a)), ay * (1.0 + np.log(a)) / a, np.exp(ay * (1.0 + np.log(a)) - 1.0) / a)


def telephone(x, rng):
    lo, hi = rng.uniform(*TELEPHONE["low_hz"]), rng.uniform(*TELEPHONE["high_hz"])
    law = TELEPHONE["codec"][int(rng.integers(0, 2))]
    gain_db = rng.uniform(*TELEPHONE["gain_db"])
    y = signal.sosfiltfilt(signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), x)
    y = signal.resample_poly(_compand(signal.resample_poly(y, 1, 2), law), 2, 1)
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)] * 10 ** (gain_db / 20)
    return y, {"kind": "telephone", "band_hz": [round(lo), round(hi)], "codec": law, "gain_db": round(gain_db, 2)}


def muffle(x, rng):
    cutoff, gain_db = rng.uniform(*MUFFLE["cutoff_hz"]), rng.uniform(*MUFFLE["gain_db"])
    y = signal.sosfiltfilt(signal.butter(2, cutoff, "lowpass", fs=SR, output="sos"), x) * 10 ** (gain_db / 20)
    return y, {"kind": "muffle", "cutoff_hz": round(cutoff), "gain_db": round(gain_db, 2)}


def add_noise(x, rng, noise_type=None, snr_range=None):
    noise_type = noise_type or NOISE_TYPES[int(rng.integers(0, len(NOISE_TYPES)))]
    snr_db = float(rng.uniform(*(snr_range or SNR_RANGE_DB)))
    noise = make_noise(noise_type, len(x), rng)
    noise *= np.sqrt(active_power(x) / 10 ** (snr_db / 10.0) / (np.mean(noise ** 2) + 1e-20))
    return x + noise, {"kind": "noise", "type": noise_type, "snr_db": round(snr_db, 2)}


def apply(audio, clip_id, epoch, seed, force_kind=None, snr_range=None):
    """(augmented float32 audio of the same length, what was done). Pure function of its arguments; clips shorter than 0.1 s are returned unchanged."""
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < 1600:
        return np.asarray(audio, dtype=np.float32), {"kind": "none"}
    rng = rng_for(seed, clip_id, epoch)
    kinds, weights = zip(*KIND_WEIGHTS)
    kind = force_kind or kinds[int(rng.choice(len(kinds), p=weights))]
    if kind == "noise":
        y, info = add_noise(x, rng, snr_range=snr_range)
    elif kind == "telephone":
        y, info = telephone(x, rng)
    elif kind == "muffle":
        y, info = muffle(x, rng)
    else:
        raise ValueError(f"unknown augmentation kind {kind}")
    peak = float(np.max(np.abs(y)))
    if peak > 0.98:
        y = y * (0.98 / peak)
    return y.astype(np.float32), info


class Augmenter:
    """What the trainer holds: seed and an optional forced kind (calibration). `last_info` is the description of the most recent copy."""

    def __init__(self, seed, force_kind=None, snr_range=None):
        self.seed, self.force_kind, self.snr_range, self.last_info = seed, force_kind, snr_range, None

    def apply(self, audio, clip_id, epoch):
        out, self.last_info = apply(audio, clip_id, epoch, self.seed, self.force_kind, self.snr_range)
        return out
