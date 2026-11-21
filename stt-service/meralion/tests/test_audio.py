import struct

import numpy as np
import pytest
import soundfile

from meralion_worker import audio
from meralion_worker.audio import InputError, ReplayRecording, load_recording


@pytest.fixture(autouse=True)
def package_loader(monkeypatch):
    """Stands in for meralion_3_asr._audio_io.load_audio: mono by averaging, no resampling (the real one is not installed here)."""
    def fake(x):
        samples, rate = x
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        return samples / peak if peak > 1.0 else samples
    monkeypatch.setattr(audio, "_package_load_audio", fake)


def write(path, samples, rate=16000, subtype="PCM_16"):
    soundfile.write(str(path), samples, rate, subtype=subtype)
    return path


def test_a_wav_is_loaded_with_its_hash_length_and_source_rate(tmp_path):
    samples = (0.1 * np.sin(np.linspace(0, 200, 24000))).astype(np.float32)
    path = write(tmp_path / "call.wav", samples, rate=24000)
    recording = load_recording("job-1", path)
    assert recording.job_name == "job-1" and recording.source_sample_rate == 24000
    assert recording.num_samples == 24000 and recording.seconds == pytest.approx(24000 / 16000)
    assert recording.sha256 == audio.sha256_file(path) and len(recording.sha256) == 64
    piece = recording.get(100, 200)
    assert piece.dtype == np.float32 and piece.flags["C_CONTIGUOUS"] and len(piece) == 100


@pytest.mark.parametrize("extension", [".wav", ".WAV", ".flac", ".ogg"])
def test_supported_extensions_are_case_insensitive(tmp_path, extension):
    soundfile_format = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG"}[extension.lower()]
    path = tmp_path / f"call{extension}"
    soundfile.write(str(path), (0.1 * np.ones(1600)).astype(np.float32), 16000, format=soundfile_format)
    assert load_recording("j", path).num_samples == 1600


def test_an_mp3_is_loaded(tmp_path):
    path = tmp_path / "call.mp3"
    try:
        soundfile.write(str(path), (0.1 * np.sin(np.linspace(0, 400, 16000))).astype(np.float32), 16000, format="MP3")
    except Exception as error:
        pytest.skip(f"this libsndfile cannot write mp3: {error}")
    assert load_recording("j", path).num_samples > 0


@pytest.mark.parametrize("name", ["call.m4a", "call.amr", "call.webm", "call.txt", "call", "call.wav.bak"])
def test_an_unsupported_extension_is_a_final_input_failure_with_the_exact_text(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"x")
    with pytest.raises(InputError) as info:
        load_recording("j", path)
    assert str(info.value) == "Unsupported or unknown media format"


def test_an_empty_audio_file_is_a_final_input_failure(tmp_path):
    path = write(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))
    with pytest.raises(InputError) as info:
        load_recording("j", path)
    assert str(info.value) == "Audio could not be decoded"


def test_garbage_with_a_supported_extension_is_a_final_input_failure(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is not audio at all" * 10)
    with pytest.raises(InputError) as info:
        load_recording("j", path)
    assert str(info.value) == "Audio could not be decoded"


def test_non_finite_audio_is_a_final_input_failure(tmp_path, monkeypatch):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(audio, "_package_load_audio", lambda x: np.array([0.1, np.nan, 0.2], dtype=np.float32))
    with pytest.raises(InputError, match="could not be decoded"):
        load_recording("j", path)


def test_a_missing_file_is_a_runtime_error_not_an_input_failure(tmp_path):
    with pytest.raises(OSError):
        load_recording("j", tmp_path / "missing.wav")


def test_the_replay_recording_has_a_length_and_silent_windows():
    recording = ReplayRecording("job", 1920000)
    assert recording.num_samples == 1920000 and recording.seconds == 120.0 and recording.source_sample_rate == 16000
    window = recording.get(0, 480000)
    assert window.dtype == np.float32 and len(window) == 480000 and not window.any()
    assert ReplayRecording("job", 5).sha256 == ReplayRecording("job", 5).sha256 != ReplayRecording("job", 6).sha256


# ---- what is a content failure and what is not (a temporary fault must never become a permanent failed marker) -------------------------------------------------------------

CONTENT_CODES = [1, 3, 4, 5, 6, 7, 12, 18, 24, 32, 33, 34, 61, 64, 69, 71, 90, 92, 100, 143, 155, 158, 159, 160, 165, 177, 182]
RUNTIME_CODES = [0, 2, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 30, 35, 45, 60,
                 91, 95, 120, 148, 156, 157, 161, 164, 168, 173, 174, 175, 176, 183,            # the reviewed non-content codes from 61 up (156: FLAC decoder creation, can be an allocation failure)
                 184, 185, 200, 400, 599, 9999, -1]                                              # 184 is "No Error"; nothing above 183 is defined, and an unknown code is never a content failure


@pytest.mark.parametrize("code", CONTENT_CODES)
def test_a_content_libsndfile_error_is_classified_as_content(code):
    assert audio.is_content_error(soundfile.LibsndfileError(code)) is True


@pytest.mark.parametrize("code", RUNTIME_CODES)
def test_a_system_resource_or_misuse_libsndfile_error_is_not_content(code):
    assert audio.is_content_error(soundfile.LibsndfileError(code)) is False


@pytest.mark.parametrize("error", [MemoryError("temporary"), OSError("disk"), ImportError("numba"), ValueError("bad"), RuntimeError("engine"), KeyError("k"), TimeoutError("t"),
                                   soundfile.SoundFileError("generic")])
def test_no_other_exception_is_ever_classified_as_content(error):
    assert audio.is_content_error(error) is False


def failing_read(error):
    def read(*args, **kwargs):
        raise error
    return read


@pytest.mark.parametrize("code", CONTENT_CODES)
def test_a_content_error_from_soundfile_is_a_final_input_failure(tmp_path, monkeypatch, code):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(soundfile, "read", failing_read(soundfile.LibsndfileError(code)))
    with pytest.raises(InputError) as info:
        load_recording("j", path)
    assert str(info.value) == "Audio could not be decoded" and info.value.__cause__.code == code


@pytest.mark.parametrize("code", RUNTIME_CODES)
def test_a_libsndfile_system_or_resource_error_propagates_and_is_not_an_input_failure(tmp_path, monkeypatch, code):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(soundfile, "read", failing_read(soundfile.LibsndfileError(code)))
    with pytest.raises(soundfile.LibsndfileError) as info:
        load_recording("j", path)
    assert not isinstance(info.value, InputError) and info.value.code == code


@pytest.mark.parametrize("error", [MemoryError("temporary"), OSError("disk hiccup"), ImportError("no module"), RuntimeError("engine"), ValueError("bad argument"), TimeoutError("slow")])
def test_other_soundfile_failures_propagate_unchanged(tmp_path, monkeypatch, error):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(soundfile, "read", failing_read(error))
    with pytest.raises(type(error)):
        load_recording("j", path)


@pytest.mark.parametrize("error", [MemoryError("temporary"), OSError("disk"), ImportError("no module"), ValueError("resample failed"), RuntimeError("engine"), KeyError("k")])
def test_any_failure_of_the_package_loader_propagates_and_is_not_an_input_failure(tmp_path, monkeypatch, error):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(audio, "_package_load_audio", failing_read(error))
    with pytest.raises(type(error)) as info:
        load_recording("j", path)
    assert not isinstance(info.value, InputError)


def real_bad_files(directory):
    """Real files libsndfile cannot use: each is a content problem and must become a final input failure (the codes observed with libsndfile 1.2.2 are in the comments)."""
    good = write(directory / "good.wav", np.full(16000, 0.1, dtype=np.float32))
    data = good.read_bytes()
    cases = {
        "garbage.wav": b"this is not audio at all" * 20,                                    # code 1
        "empty.wav": b"",                                                                    # code 1
        "truncated_header.wav": data[:20],                                                   # code 64
        "wrong_format_tag.wav": data[:20] + struct.pack("<H", 0x0050) + data[22:],           # code 64
        "zero_sample_rate.wav": data[:24] + struct.pack("<I", 0) + data[28:],                # code 24
        "garbage.flac": b"fLaC" + b"\x00" * 40,                                              # code 18
        "garbage.ogg": b"OggS" + b"\x00" * 40,                                               # code 18
        "garbage.mp3": b"ID3" + b"\x00" * 60,                                                # code 7
    }
    for name, blob in cases.items():
        (directory / name).write_bytes(blob)
    return list(cases)


def test_real_bad_content_files_are_final_input_failures(tmp_path):
    for name in real_bad_files(tmp_path):
        with pytest.raises(InputError) as info:
            load_recording("j", tmp_path / name)
        assert str(info.value) == "Audio could not be decoded", name
        assert audio.is_content_error(info.value.__cause__), (name, info.value.__cause__)


def test_a_missing_file_and_a_directory_are_runtime_errors_not_input_failures(tmp_path):
    with pytest.raises(OSError):
        load_recording("j", tmp_path / "missing.wav")
    (tmp_path / "folder.wav").mkdir()
    with pytest.raises(OSError):
        load_recording("j", tmp_path / "folder.wav")


@pytest.mark.parametrize("label, samples, rate", [
    ("zero rate", np.full(1600, 0.1, dtype=np.float32), 0),
    ("negative rate", np.full(1600, 0.1, dtype=np.float32), -16000),
    ("float rate", np.full(1600, 0.1, dtype=np.float32), 16000.5),
    ("bool rate", np.full(1600, 0.1, dtype=np.float32), True),
    ("three dimensions", np.zeros((10, 2, 2), dtype=np.float32), 16000),
    ("zero dimensions", np.float32(0.1), 16000),
    ("no frames", np.zeros((0, 2), dtype=np.float32), 16000),
    ("no channels", np.zeros((100, 0), dtype=np.float32), 16000),
    ("nan", np.array([0.1, np.nan, 0.2], dtype=np.float32), 16000),
    ("inf", np.array([0.1, np.inf], dtype=np.float32), 16000),
])
def test_samples_that_fail_our_own_checks_are_final_input_failures_before_any_package_call(tmp_path, monkeypatch, label, samples, rate):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(soundfile, "read", lambda *a, **k: (samples, rate))
    monkeypatch.setattr(audio, "_package_load_audio", failing_read(AssertionError("the package loader must not see bad content")))
    with pytest.raises(InputError) as info:
        load_recording("j", path)
    assert str(info.value) == "Audio could not be decoded", label


def test_stereo_samples_pass_the_checks_and_reach_the_package_loader(tmp_path, monkeypatch):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    seen = {}

    def loader(x):
        seen["input"] = x
        return np.full(1600, 0.1, dtype=np.float32)

    monkeypatch.setattr(soundfile, "read", lambda *a, **k: (np.full((1600, 2), 0.1, dtype=np.float32), 44100))
    monkeypatch.setattr(audio, "_package_load_audio", loader)
    assert load_recording("j", path).source_sample_rate == 44100 and seen["input"][1] == 44100 and seen["input"][0].shape == (1600, 2)


# ---- the allow-list itself: explicit, reviewed, and checked against the library's own table ---------------------------------------------------------------------------

def table_message(code):
    return soundfile._ffi.string(soundfile._snd.sf_error_number(code)).decode()


def test_the_allow_list_is_exactly_the_reviewed_set():
    expected = ({1, 3, 4, 5, 6, 7, 12, 18, 24, 32, 33, 34} | set(range(61, 91)) | set(range(92, 95)) | set(range(98, 120)) | set(range(121, 144)) | {146, 147}
                | set(range(149, 156)) | {158, 159, 160, 162, 165, 166, 167} | set(range(177, 183)))
    assert audio.CONTENT_CODES == expected and len(audio.CONTENT_CODES) == 112
    assert not hasattr(audio, "FORMAT_SPECIFIC_FLOOR")                                      # no numeric threshold any more


def test_every_listed_code_is_defined_by_the_library_and_none_describes_a_resource_fault():
    for code in sorted(audio.CONTENT_CODES):
        message = table_message(code)
        assert not message.startswith("Not a valid error number") and message != "No Error."
        lowered = message.lower()
        for resource in ("malloc", "memory", "temp", "creating", "initialization", "unknown error", "pipe", "supplied filename", "negative"):
            assert resource not in lowered or code == 7, (code, message)                      # 7 is the one reviewed exception: see test_code_7_is_content_only_for_an_existing_regular_file
    for code in (156, 157, 161, 173):
        assert code not in audio.CONTENT_CODES


def test_no_code_the_library_does_not_define_is_content():
    for code in range(184, 700):
        assert audio.is_content_error(soundfile.LibsndfileError(code)) is False


def test_real_bad_files_still_give_codes_in_the_list(tmp_path):
    """Garbage and a truncated WAV header are content failures. A FLAC truncated inside the stream gives 161, which is deliberately NOT content (it is ambiguous), so it propagates."""
    import io
    garbage = {"g.wav": b"x" * 500, "g.flac": b"y" * 500, "g.ogg": b"z" * 500, "empty.wav": b""}
    for name, data in garbage.items():
        path = tmp_path / name
        path.write_bytes(data)
        with pytest.raises(soundfile.LibsndfileError) as info:
            soundfile.read(str(path))
        assert audio.is_content_error(info.value), (name, info.value.code)
    buffer = io.BytesIO()
    soundfile.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    (tmp_path / "t.wav").write_bytes(buffer.getvalue()[:30])
    with pytest.raises(soundfile.LibsndfileError) as info:
        soundfile.read(str(tmp_path / "t.wav"))
    assert info.value.code == 71 and audio.is_content_error(info.value)
    flac = io.BytesIO()
    soundfile.write(flac, np.zeros(1600, dtype=np.float32), 16000, format="FLAC")
    (tmp_path / "t.flac").write_bytes(flac.getvalue()[:60])
    with pytest.raises(soundfile.LibsndfileError) as info:
        soundfile.read(str(tmp_path / "t.flac"))
    assert info.value.code == 161 and not audio.is_content_error(info.value)


@pytest.mark.parametrize("code", [156, 157, 161, 9999])
def test_a_resource_or_unknown_error_on_valid_flac_audio_propagates_and_is_not_an_input_failure(tmp_path, monkeypatch, code):
    path = tmp_path / "call.flac"
    soundfile.write(str(path), np.full(1600, 0.1, dtype=np.float32), 16000, format="FLAC")
    monkeypatch.setattr(soundfile, "read", failing_read(soundfile.LibsndfileError(code)))
    with pytest.raises(soundfile.LibsndfileError) as info:
        load_recording("j", path)
    assert info.value.code == code and not isinstance(info.value, InputError)


def test_code_7_is_content_only_for_an_existing_regular_file(tmp_path, monkeypatch):
    path = write(tmp_path / "call.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(soundfile, "read", failing_read(soundfile.LibsndfileError(7)))
    with pytest.raises(InputError):                                                         # the file is there: a garbage-MP3-style failure is about its content
        load_recording("j", path)

    def vanish(*args, **kwargs):
        path.unlink()                                                                       # the path disappears while libsndfile is opening it, then libsndfile reports 7
        raise soundfile.LibsndfileError(7)

    path2 = write(tmp_path / "call2.wav", np.full(1600, 0.1, dtype=np.float32))
    monkeypatch.setattr(soundfile, "read", lambda *a, **k: (path2.unlink(), (_ for _ in ()).throw(soundfile.LibsndfileError(7)))[1])
    with pytest.raises(soundfile.LibsndfileError):
        load_recording("j", path2)
