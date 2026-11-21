"""Audio input. A recording is read once through soundfile and normalised exactly as the package does (meralion_3_asr._audio_io.load_audio: mono, 16 kHz, peak-normalised only if the
peak exceeds 1.0). Windows are slices of that whole-recording waveform.

A recording whose input cannot be used is a FINAL input failure (InputError, with the exact text, recorded as a permanent failed marker). ONLY content problems qualify:
  * an unsupported extension;
  * a libsndfile error that is about the file's CONTENT (is_content_error below);
  * samples that fail our own checks: a sample rate that is not a positive integer, not 1 or 2 dimensions, no frames, non-finite values.
Everything else is a RUNTIME error that propagates (exit code 1, the invocation resumes after the fault is gone): a missing or unreadable file, MemoryError, ImportError, OSError, a libsndfile
system or resource error, and ANY error raised by the package's loader (it runs outside every try, after the checks above have kept known bad content away from it).
"""
import hashlib
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
UNSUPPORTED = "Unsupported or unknown media format"
UNDECODABLE = "Audio could not be decoded"
SAMPLE_RATE = 16000

# libsndfile error codes: an EXPLICIT, REVIEWED allow-list of the codes that describe the file's CONTENT. Anything not listed, including every code this library version does not define,
# is NOT a content failure and propagates as a runtime error (exit code 1, resumable). Source: libsndfile 1.2.2's own table (soundfile 0.14.0, sf_error_number): codes 0 to 183 are defined,
# 184 is "No Error" and everything above is "Not a valid error number".
#
# Listed (112 codes):
#   1 format not recognised, 3 file is malformed, 4 unsupported encoding, 5 and 6 major or minor format is 0, 7 not a regular file (what a garbage MP3 raises; it counts as content only while the path is still a regular file: load_recording re-checks that, because the
#   message also covers a missing file or a pipe), 12 incomplete header at end of parsing, 18 unimplemented format (garbage or truncated FLAC or Ogg), 24 incomplete SF_INFO (a WAV whose header
#   gives a zero sample rate), 32 to 34 bad channel count in the header;
#   61 to 90, 92 to 94, 98 to 119, 121 to 143, 146 and 147, 149 to 155, 158 to 160, 162, 165 to 167, 177 to 182: the format-specific header and stream errors (WAV, AIFF, AU, PAF, 8SVX, NIST,
#   VOC, IRCAM, W64, MAT, PVF, XI, SDS, SD2, FLAC header, sync, sample rate and channel changes, WVE, RF64, CAF, AVR, MPC2K).
# Deliberately NOT listed (propagate): 2 system error, 8 no data could be read, 9 could not open, 15 misconfigured build, 16 no temp dir, 17 malloc failed, 10 to 11, 13 to 14, 19 to 23,
#   25 to 31, 35 to 60 (API misuse or write-side errors: our own bugs); and, from 61 up, the ones that are not about the content or that can be caused by the machine:
#   91 (read-write mode), 95 to 97 (raw files), 120, 144, 145, 163 (pipes), 148 (no file name), 156 (FLAC decoder creation, which libsndfile raises when decoder allocation fails),
#   157 (FLAC decoder initialisation), 161 (unknown FLAC decoder error), 164 (encoder bug, write side), 168 to 172 (chunk API misuse), 173 (temp file for ALAC encoding),
#   174 (file name too long), 175 (negative length), 176 and 183 (Opus and MPEG sample-rate limits on the write side).
# Consequence, stated plainly: a FLAC truncated inside the stream gives 161 and a truncated WAV gives 71, so a truncated FLAC is a RUNTIME error (visible, exit 1) and never a permanent skip.
CONTENT_CODES = frozenset(
    {1, 3, 4, 5, 6, 7, 12, 18, 24, 32, 33, 34}
    | set(range(61, 91)) | set(range(92, 95)) | set(range(98, 120)) | set(range(121, 144)) | {146, 147} | set(range(149, 156)) | {158, 159, 160, 162} | {165, 166, 167} | set(range(177, 183)))


class InputError(Exception):
    """A final input failure (the recording is marked failed; the others continue)."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_load_audio(x):
    """meralion_3_asr._audio_io.load_audio, imported lazily. Tests replace this function."""
    from meralion_3_asr._audio_io import load_audio
    return load_audio(x)


class AudioRecording:
    def __init__(self, job_name, waveform, sha256, source_sample_rate):
        self.job_name = job_name
        self.waveform = waveform
        self.sha256 = sha256
        self.source_sample_rate = source_sample_rate

    @property
    def num_samples(self):
        return int(len(self.waveform))

    @property
    def seconds(self):
        return self.num_samples / SAMPLE_RATE

    def get(self, start, end):
        return np.ascontiguousarray(self.waveform[start:end])


class ReplayRecording:
    """A recording that has a length and no audio: its windows are silent arrays. For replay and tests only."""

    def __init__(self, job_name, num_samples, source_sample_rate=SAMPLE_RATE):
        self.job_name = job_name
        self._num_samples = int(num_samples)
        self.source_sample_rate = source_sample_rate
        self.sha256 = hashlib.sha256(f"replay:{job_name}:{num_samples}".encode("utf-8")).hexdigest()

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def seconds(self):
        return self._num_samples / SAMPLE_RATE

    def get(self, start, end):
        return np.zeros(end - start, dtype=np.float32)


def is_content_error(error):
    """True only for a libsndfile error whose code is in the reviewed CONTENT_CODES list. Any other exception, any other code and any code this library does not define is not."""
    import soundfile
    return isinstance(error, soundfile.LibsndfileError) and error.code in CONTENT_CODES


def load_recording(job_name, path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InputError(UNSUPPORTED)
    digest = sha256_file(path)                                   # a missing or unreadable file raises OSError here: a runtime error, not an input failure
    import soundfile
    try:
        samples, rate = soundfile.read(str(path), dtype="float32")
    except soundfile.LibsndfileError as error:
        if not is_content_error(error) or (error.code == 7 and not path.is_file()):          # 7 is content only for a regular file that exists (a garbage MP3); a vanished or non-regular path is a runtime fault
            raise                                                # a system or resource error (or our own misuse): runtime, resumable
        raise InputError(UNDECODABLE) from error
    samples = np.asarray(samples)
    if (isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0 or samples.ndim not in (1, 2) or samples.size == 0 or 0 in samples.shape
            or not np.isfinite(samples).all()):
        raise InputError(UNDECODABLE)
    waveform = np.ascontiguousarray(_package_load_audio((samples, int(rate))), dtype=np.float32)       # outside every try: any failure here is a runtime error
    if waveform.ndim != 1 or len(waveform) == 0 or not np.isfinite(waveform).all():
        raise InputError(UNDECODABLE)
    return AudioRecording(job_name, waveform, digest, int(rate))
