"""Signal processing for environment_v2 (numpy + scipy + soundfile only).

Everything works on int16 PCM at the SOURCE sample rate. The speech is never resampled (D1 resamples only inside a round trip that
returns to the exact source length). CLIENT regions are processed independently; every sample outside them is copied unchanged.
"""

import math

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import AugmentationError

FRAME_MS = 20
ACTIVE_DB = 40          # frames within 40 dB of the loudest count as active speech (simplified ITU-T P.56)
HP_HZ = 100             # high-pass on noise assets, and on the speech used to measure the SNR
SPEECH_BAND = (300, 3400)
FULL_SCALE = 32768.0


# ---------------------------------------------------------------------------------------------------------------- loading
def load_audio(path):
    """(int16 mono samples, sample rate). MP3 and WAV go through the same libsndfile decode (checked against ffmpeg in the tests)."""
    info = sf.info(str(path))
    if info.channels != 1:
        raise AugmentationError(f"{path}: expected mono audio, found {info.channels} channels")
    pcm, rate = sf.read(str(path), dtype="int16", always_2d=False)
    return pcm, int(rate)


def to_float(pcm):
    return pcm.astype(np.float32) / FULL_SCALE


def to_pcm(values):
    return np.clip(np.rint(np.asarray(values, dtype=np.float64) * FULL_SCALE), -32768, 32767).astype(np.int16)


def write_wav(path, pcm, rate):
    sf.write(str(path), pcm, rate, subtype="PCM_16", format="WAV")


# ---------------------------------------------------------------------------------------------------------------- regions
def client_regions(turns, roles, n_frames, rate, pad_ms=200):
    """One region per CLIENT turn: the turn widened by `pad_ms` each side, clipped to the neighbouring turns and the audio.

    turns: [{turn_id, start_seconds, end_seconds}] in order; roles: {turn_id: "CLIENT"|"EMPLOYEE"}. `start`/`end` are the turn itself
    (the 'interior' used for level measurements); `lo`/`hi` the padded span that is actually written. The 400 ms gaps are digital
    silence, so the seams of the padded span fall in silence.
    """
    pad = int(round(pad_ms * rate / 1000))
    bounds = [(t["turn_id"], int(round(t["start_seconds"] * rate)), min(n_frames, int(round(t["end_seconds"] * rate)))) for t in turns]
    regions = []
    for index, (turn_id, start, end) in enumerate(bounds):
        if roles[turn_id] != "CLIENT":
            continue
        previous_end = bounds[index - 1][2] if index else 0
        next_start = bounds[index + 1][1] if index + 1 < len(bounds) else n_frames
        lo, hi = max(previous_end, start - pad), min(next_start, end + pad, n_frames)
        if not 0 <= lo <= start < end <= hi <= n_frames:
            raise AugmentationError(f"turn {turn_id}: unusable client region {lo}..{start}..{end}..{hi} of {n_frames} frames")
        regions.append({"turn_id": turn_id, "start": start, "end": end, "lo": lo, "hi": hi})
    return regions


def _sos(kind, rate, edges, order):
    return butter(order, edges, kind, fs=rate, output="sos")


def highpass_sos(rate):
    return _sos("highpass", rate, HP_HZ, 4)


def frame_powers(x, rate):
    size = int(rate * FRAME_MS / 1000)
    count = len(x) // size
    if count == 0:
        return np.empty(0)
    return (np.asarray(x[: count * size], dtype=np.float64).reshape(count, size) ** 2).mean(axis=1)


def active_level(power_arrays):
    """Mean power of the active frames (within ACTIVE_DB of the loudest) over all the given frame-power arrays."""
    power = np.concatenate(power_arrays) if power_arrays else np.empty(0)
    if power.size == 0 or power.max() <= 0:
        raise AugmentationError("no speech energy to measure")
    return float(power[power >= power.max() * 10 ** (-ACTIVE_DB / 10)].mean())


def speech_level(x, regions, rate, band=None):
    """Active speech power over the CLIENT turn interiors, after the 100 Hz high-pass (and an optional band-pass)."""
    sos = highpass_sos(rate)
    arrays = []
    for r in regions:
        y = sosfiltfilt(sos, np.asarray(x[r["start"]:r["end"]], dtype=np.float64))
        if band:
            y = sosfiltfilt(_sos("bandpass", rate, band, 4), y)
        arrays.append(frame_powers(y, rate))
    return active_level(arrays)


# ---------------------------------------------------------------------------------------------------------------- noise
_NOISE_CACHE = {}


def soft_limit(x, crest_db):
    """Round off rare spikes: samples above crest_db over the asset RMS go through a smooth knee (linear below 70% of the limit)."""
    limit = float(np.sqrt((x.astype(np.float64) ** 2).mean())) * 10 ** (crest_db / 20)
    knee = 0.7 * limit
    magnitude = np.abs(x)
    over = magnitude > knee
    out = x.copy()
    out[over] = np.sign(x[over]) * (knee + (limit - knee) * np.tanh((magnitude[over] - knee) / (limit - knee)))
    return out


def prepare_noise(path, rate, crest_limit_db=None):
    """Mono, resampled to the call rate, 100 Hz high-passed, optionally spike-limited float32. Cached per (path, rate, limit)."""
    key = (str(path), rate, crest_limit_db)
    if key not in _NOISE_CACHE:
        data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        divisor = math.gcd(rate, source_rate)
        mono = resample_poly(mono, rate // divisor, source_rate // divisor)
        prepared = sosfiltfilt(highpass_sos(rate), mono).astype(np.float32)
        _NOISE_CACHE[key] = soft_limit(prepared, crest_limit_db).astype(np.float32) if crest_limit_db else prepared
    return _NOISE_CACHE[key]


def _looped(noise, length, rate):
    """Guard for a region longer than the asset: equal-power 1 s crossfade loop. Never reached with the current assets."""
    fade = min(rate, len(noise) // 2)
    ramp = np.linspace(0.0, math.pi / 2, fade, dtype=np.float32)
    out = noise
    while len(out) < length:
        out = np.concatenate([out[:-fade], out[-fade:] * np.cos(ramp) + noise[:fade] * np.sin(ramp), noise[fade:]])
    return out


_CALM_CACHE = {}
CALM_HOP_S = 0.1


def _frame_levels_db(noise, rate):
    """100 ms RMS levels of a prepared noise asset in dB (cached per array)."""
    key = id(noise)
    if key not in _CALM_CACHE:
        hop = int(rate * CALM_HOP_S)
        count = len(noise) // hop
        frames = np.asarray(noise[: count * hop], dtype=np.float64).reshape(count, hop)
        _CALM_CACHE[key] = (noise, 10 * np.log10((frames ** 2).mean(axis=1) + 1e-20))
    return _CALM_CACHE[key][1]


CALM_MARGIN_DB = 6.0      # a calm window never has a 100 ms burst more than this far above the asset typical level
CALM_FLOOR = 0.05         # ... unless that leaves under 5% of the windows; then the calmest 5% are used


def calm_starts(noise, length, rate, calm_fraction):
    """Start frames (100 ms grid) of the windows that keep loud events out.

    Printer runs, beeps, door slams and announcements show up as loud 100 ms frames, which the average noise power does not see, so
    two clips with the same masking score could sound very different. A window is calm when its loudest 100 ms frame is within the
    calmest `calm_fraction` of all windows AND within CALM_MARGIN_DB of the asset median level; if too few windows qualify, the
    calmest CALM_FLOOR fraction is used instead.
    """
    levels = _frame_levels_db(noise, rate)
    size = int(math.ceil(length / int(rate * CALM_HOP_S))) + 1
    if size >= len(levels):
        return np.array([0])
    loudest = np.lib.stride_tricks.sliding_window_view(levels, size).max(axis=1)
    limit = min(float(np.quantile(loudest, calm_fraction)), float(np.median(levels)) + CALM_MARGIN_DB)
    starts = np.flatnonzero(loudest <= limit)
    if len(starts) < CALM_FLOOR * len(loudest):
        starts = np.flatnonzero(loudest <= np.quantile(loudest, CALM_FLOOR))
    return starts


def noise_excerpt(noise, length, rate, rng, calm_fraction=None):
    """(excerpt, offset): one contiguous seeded excerpt, so there is never a loop inside a turn."""
    if len(noise) < length:
        return _looped(noise, length, rate)[:length], 0
    if calm_fraction:
        starts = calm_starts(noise, length, rate, calm_fraction)
        hop = int(rate * CALM_HOP_S)
        offset = int(starts[int(rng.integers(0, len(starts)))]) * hop + int(rng.integers(0, hop))
        offset = min(offset, len(noise) - length)
    else:
        offset = int(rng.integers(0, len(noise) - length + 1))
    return noise[offset:offset + length], offset


def fade_window(length, fade_in, fade_out):
    window = np.ones(length, dtype=np.float32)
    if fade_in:
        window[:fade_in] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_in) / fade_in)
    if fade_out:
        window[length - fade_out:] = (0.5 - 0.5 * np.cos(np.pi * np.arange(fade_out) / fade_out))[::-1]
    return window


def add_background_noise(x, regions, rate, noise, snr_db, rng_for_turn, fade_ms=150, target_ai=None, snr_bounds=None, calm_fraction=None):
    """Mix noise into the client regions at `snr_db` against the ACTIVE CLIENT SPEECH level (not the RM, not the silences).

    Returns (list of processed float arrays, one per region, info). One gain per call, so the environment is constant within the call.
    """
    speech = speech_level(x, regions, rate)
    excerpts, offsets, power_sum, power_count = [], [], 0.0, 0
    for r in regions:
        excerpt, offset = noise_excerpt(noise, r["hi"] - r["lo"], rate, rng_for_turn(r["turn_id"]), calm_fraction)
        inner = np.asarray(excerpt[r["start"] - r["lo"]:r["end"] - r["lo"]], dtype=np.float64)
        power_sum += float((inner ** 2).sum())
        power_count += len(inner)
        excerpts.append(excerpt)
        offsets.append([r["turn_id"], offset / rate])
    noise_power = power_sum / power_count
    clamped = None
    if target_ai is not None:
        snr_db, clamped = solve_snr_for_ai(x, regions, excerpts, rate, speech, noise_power, target_ai, snr_bounds)
    gain = math.sqrt(speech / (noise_power * 10 ** (snr_db / 10)))
    fade = int(round(fade_ms * rate / 1000))
    processed = []
    for r, excerpt in zip(regions, excerpts):
        window = fade_window(r["hi"] - r["lo"], min(fade, r["start"] - r["lo"]), min(fade, r["hi"] - r["end"]))
        processed.append(np.asarray(x[r["lo"]:r["hi"]], dtype=np.float64) + gain * excerpt * window)
    return processed, {"asset_gain_db": 20 * math.log10(gain), "noise_level_dbfs": 10 * math.log10(noise_power * gain ** 2),
                       "speech_level_dbfs": 10 * math.log10(speech), "offsets_s": offsets, "resolved_snr_db": snr_db,
                       "ai_target": target_ai, "snr_clamped": clamped}


# ---------------------------------------------------------------------------------------------------------------- channels
def mulaw_roundtrip(x, mu=255):
    """Sign + 7-bit magnitude companded quantiser (mid-tread, so digital silence stays silent)."""
    x = np.clip(x, -1.0, 1.0)
    companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantised = np.rint(companded * 127) / 127
    return np.sign(quantised) * np.expm1(np.abs(quantised) * np.log1p(mu)) / mu


def telephone_bandpass_sos(rate, low_hz, high_hz, order=4):
    return _sos("bandpass", rate, [low_hz, high_hz], order)


def simulate_telephone(x, rate, band_hz, order=4, codec=True):
    """Band-pass (zero phase) and, if `codec`, a G.711-style 8 kHz mu-law round trip. Same length and sample rate out as in.

    Changing the frequency CONTENT (the band-pass) keeps the sample grid; the 8 kHz step only models where the codec acts and is
    resampled straight back with a linear-phase filter, so nothing shifts in time.
    """
    y = sosfiltfilt(telephone_bandpass_sos(rate, *band_hz, order), np.asarray(x, dtype=np.float64))
    if not codec:
        return y
    divisor = math.gcd(rate, 8000)
    narrow = resample_poly(y, 8000 // divisor, rate // divisor)
    back = resample_poly(mulaw_roundtrip(narrow), rate // divisor, 8000 // divisor)
    return np.pad(back, (0, max(0, len(y) - len(back))))[: len(y)]


def muffle_sos(rate, cutoff_hz, order=2):
    return _sos("lowpass", rate, cutoff_hz, order)


def simulate_muffling(x, rate, cutoff_hz, gain_db=-6.0, order=2):
    return sosfiltfilt(muffle_sos(rate, cutoff_hz, order), np.asarray(x, dtype=np.float64)) * 10 ** (gain_db / 20)


def zero_phase_response(sos, rate, freqs):
    """Magnitude of what sosfiltfilt applies: |H|^2 (forward + backward pass), about -6 dB at a Butterworth design edge."""
    from scipy.signal import sosfreqz
    _, h = sosfreqz(sos, worN=np.asarray(freqs, dtype=float), fs=rate)
    return np.abs(h) ** 2


# ---------------------------------------------------------------------------------------------------------------- assembly
def _outside_mask(n_frames, regions):
    mask = np.ones(n_frames, dtype=bool)
    for r in regions:
        mask[r["lo"]:r["hi"]] = False
    return mask


def augment_pcm(pcm, rate, regions, kind, params, *, noise=None, rng_for_turn=None, fade_ms=150, headroom=0.98):
    """Apply one condition to the CLIENT regions of `pcm`; returns (output int16 array, info).

    kind: "noise" (params: snr_db), "telephone" (band_hz, order, codec) or "muffle" (cutoff_hz, gain_db, order).
    Headroom: if any processed client region would peak above `headroom`, ONE gain for the call scales the client speech and its
    noise together in every client region (SNR unchanged, level constant across turns). Nothing outside the client regions is touched,
    and the limit is enforced only inside them: pre-existing full-scale samples elsewhere are preserved and reported.
    """
    x = to_float(pcm)
    info = {}
    if kind == "noise":
        processed, noise_info = add_background_noise(x, regions, rate, noise, params["snr_db"], rng_for_turn, fade_ms,
                                                      params.get("target_ai"), params.get("snr_bounds"), params.get("calm_fraction"))
        info.update(noise_info)
    elif kind == "telephone":
        processed = [simulate_telephone(x[r["lo"]:r["hi"]], rate, params["band_hz"], params.get("order", 4), params.get("codec", True))
                     for r in regions]
    elif kind == "muffle":
        processed = [simulate_muffling(x[r["lo"]:r["hi"]], rate, params["cutoff_hz"], params.get("gain_db", -6.0), params.get("order", 2))
                     for r in regions]
    else:
        raise AugmentationError(f"unknown condition kind {kind!r}")
    peaks = [float(np.abs(p).max()) for p in processed]
    pre_gain_peak = max(peaks)
    triggering = sum(1 for peak in peaks if peak > headroom)
    client_gain = headroom / pre_gain_peak if pre_gain_peak > headroom else 1.0
    output = pcm.copy()
    for r, p in zip(regions, processed):
        output[r["lo"]:r["hi"]] = to_pcm(p * client_gain)
    outside = _outside_mask(len(pcm), regions)
    outside_abs = np.abs(pcm[outside].astype(np.int32))
    info.update({
        "client_gain_db": 20 * math.log10(client_gain),
        "client_gain_regions_triggering": int(triggering),
        "client_pre_gain_peak": pre_gain_peak,
        "client_peak_after": max(int(np.abs(output[r["lo"]:r["hi"]].astype(np.int32)).max()) for r in regions) / FULL_SCALE,
        "employee_fullscale_samples": int((outside_abs >= 32767).sum()),
        "employee_peak": float(outside_abs.max() / FULL_SCALE) if outside_abs.size else 0.0,
        "outside_identical": bool(np.array_equal(output[outside], pcm[outside])),
    })
    return output, info


# ---------------------------------------------------------------------------------------------------------------- measurement
def measure_snr(pcm, output, regions, rate, client_gain_db):
    """SNR read back from the OUTPUT: noise = output / client_gain - source over the client regions (independent of the build maths)."""
    scale = 10 ** (client_gain_db / 20)
    x = to_float(pcm).astype(np.float64)
    added = to_float(output).astype(np.float64) / scale - x
    sos_hp = highpass_sos(rate)
    sos_band = _sos("bandpass", rate, SPEECH_BAND, 4)
    full, band, per_turn = [], [], []
    noise_sum, band_noise_sum, noise_count = 0.0, 0.0, 0
    for r in regions:
        n = added[r["start"]:r["end"]]  # the noise asset is already high-passed; only the speech is high-passed for the measurement
        noise_sum += float((n ** 2).sum())
        band_noise_sum += float((sosfiltfilt(sos_band, n) ** 2).sum())
        noise_count += len(n)
        s = sosfiltfilt(sos_hp, x[r["start"]:r["end"]])
        full.append(frame_powers(s, rate))
        band.append(frame_powers(sosfiltfilt(sos_band, s), rate))
        per_turn.append(10 * math.log10(active_level([frame_powers(s, rate)]) / ((n ** 2).mean() + 1e-30)))
    return {"fullband_db": 10 * math.log10(active_level(full) / (noise_sum / noise_count)),
            "speechband_db": 10 * math.log10(active_level(band) / (band_noise_sum / noise_count)),
            "per_turn_p5_p50_p95_db": [float(v) for v in np.percentile(per_turn, [5, 50, 95])]}


def level_change_db(pcm, output, regions, rate):
    """Active CLIENT speech level, output versus source, in dB (channel conditions: shows D2 attenuation / D1 band loss)."""
    return 10 * math.log10(speech_level(to_float(output), regions, rate) / speech_level(to_float(pcm), regions, rate))


# Approximate octave-band importance weights of speech (articulation index). A RELATIVE proxy for how much the noise masks speech:
# about 1.0 means clean, below about 0.5 clearly degraded. It is not a recognition score and is only used to compare levels and assets.
AI_WEIGHTS = {250: 0.10, 500: 0.18, 1000: 0.25, 2000: 0.27, 4000: 0.20}


def articulation_index(pcm, output, regions, rate, client_gain_db):
    x = to_float(pcm).astype(np.float64)
    added = to_float(output).astype(np.float64) / 10 ** (client_gain_db / 20) - x
    total = 0.0
    for centre, weight in AI_WEIGHTS.items():
        high = min(centre * math.sqrt(2), rate / 2 * 0.95)
        sos = _sos("bandpass", rate, [centre / math.sqrt(2), high], 4)
        speech = active_level([frame_powers(sosfiltfilt(sos, x[r["start"]:r["end"]]), rate) for r in regions])
        noise = np.concatenate([sosfiltfilt(sos, added[r["start"]:r["end"]]) for r in regions])
        snr = 10 * math.log10(speech / (float((noise ** 2).mean()) + 1e-30))
        total += weight * min(max((snr + 15) / 30, 0.0), 1.0)
    return total


def _ai_sos(rate):
    return [(weight, _sos("bandpass", rate, [centre / math.sqrt(2), min(centre * math.sqrt(2), rate / 2 * 0.95)], 4))
            for centre, weight in AI_WEIGHTS.items()]


def _ai_of(snrs):
    return sum(weight * min(max((snr + 15) / 30, 0.0), 1.0) for weight, snr in snrs)


def solve_snr_for_ai(x, regions, excerpts, rate, speech_full, noise_full, target_ai, bounds):
    """The full-band SNR (within `bounds`) at which the noise masks this call's client speech to `target_ai`.

    AI rises with the SNR, so a bisection finds it. Everything is measured on this call's own client turns and the excerpts that will
    actually be mixed, so a voice whose consonants sit in a noisier band automatically gets a milder SNR. Returns (snr_db, clamped).
    """
    bands = []
    for weight, sos in _ai_sos(rate):
        speech = active_level([frame_powers(sosfiltfilt(sos, np.asarray(x[r["start"]:r["end"]], dtype=np.float64)), rate) for r in regions])
        noise = np.concatenate([sosfiltfilt(sos, np.asarray(e[r["start"] - r["lo"]:r["end"] - r["lo"]], dtype=np.float64))
                                for r, e in zip(regions, excerpts)])
        bands.append((weight, speech, float((noise ** 2).mean())))

    def ai_at(snr_db):
        g2 = speech_full / (noise_full * 10 ** (snr_db / 10))      # squared noise gain that gives this full-band SNR
        return _ai_of([(w, 10 * math.log10(s / (n * g2))) for w, s, n in bands])

    low, high = bounds
    if ai_at(low) >= target_ai:
        return low, True
    if ai_at(high) <= target_ai:
        return high, True
    for _ in range(50):
        middle = (low + high) / 2
        low, high = (middle, high) if ai_at(middle) < target_ai else (low, middle)
    return (low + high) / 2, False
