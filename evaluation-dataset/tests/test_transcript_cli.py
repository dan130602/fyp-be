"""CLI regression tests with ARTIFICIAL files in pytest temporary directories
only; never touches real pilot call data."""

import json

import pytest

from compilation_fixtures import SCHEMAS
from transcript_fixtures import minimal_ground_truth
from fyp_evaluation.transcripts.cli import main


def _write_call(calls_dir, call_id, turns):
    call_dir = calls_dir / call_id
    call_dir.mkdir(parents=True)
    (call_dir / "ground_truth.json").write_text(
        json.dumps(minimal_ground_truth(call_id, turns)), encoding="utf-8",
    )
    return call_dir


@pytest.fixture
def calls_dir(tmp_path):
    return tmp_path / "calls"


def run(monkeypatch, calls_dir, *arguments):
    # monkeypatch is accepted (unused) to keep this helper's signature
    # consistent with the rest of the suite's CLI-runner fixtures.
    del monkeypatch
    return main(["--calls-dir", str(calls_dir), "--schema-dir", str(SCHEMAS), *map(str, arguments)])


# --- Single-call and batch build ------------------------------------------

def test_build_single_call_writes_both_files(calls_dir, monkeypatch, capsys):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    assert (call_dir / "transcript.json").exists()
    assert (call_dir / "speaker_roles.json").exists()
    out = capsys.readouterr().out
    assert "CALL_001" in out and "2 turns" in out


def test_build_all_discovers_every_call_directory(calls_dir, monkeypatch):
    _write_call(calls_dir, "CALL_002", [("CLIENT", "Hi.")])
    _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi.")])
    assert run(monkeypatch, calls_dir, "--all") == 0
    for call_id in ("CALL_001", "CALL_002"):
        assert (calls_dir / call_id / "transcript.json").exists()
        assert (calls_dir / call_id / "speaker_roles.json").exists()


def test_build_all_with_no_calls_is_a_clear_usage_error(calls_dir, monkeypatch):
    calls_dir.mkdir()
    assert run(monkeypatch, calls_dir, "--all") == 2


# --- Output contract on real files ----------------------------------------

def test_written_transcript_has_no_results_wrapper_or_speaker_roles(calls_dir, monkeypatch):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    run(monkeypatch, calls_dir, "--call", "CALL_001")
    transcript = json.loads((call_dir / "transcript.json").read_text(encoding="utf-8"))
    assert set(transcript) == {"transcripts", "audio_segments"}
    speaker_roles = json.loads((call_dir / "speaker_roles.json").read_text(encoding="utf-8"))
    assert speaker_roles == {"spk_0": "RM", "spk_1": "Customer"}


# --- Overwrite protection --------------------------------------------------

def test_refuses_to_overwrite_existing_output_without_flag(calls_dir, monkeypatch):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    original = (call_dir / "transcript.json").read_text(encoding="utf-8")
    # Simulate authoring drift: the dialogue changed, but the run has no --overwrite.
    ground = json.loads((call_dir / "ground_truth.json").read_text(encoding="utf-8"))
    ground["dialogue"][0]["text"] = "ARTIFICIAL changed text."
    (call_dir / "ground_truth.json").write_text(json.dumps(ground), encoding="utf-8")
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 2
    assert (call_dir / "transcript.json").read_text(encoding="utf-8") == original


def test_overwrite_flag_explicitly_regenerates(calls_dir, monkeypatch):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    ground = json.loads((call_dir / "ground_truth.json").read_text(encoding="utf-8"))
    ground["dialogue"][0]["text"] = "ARTIFICIAL changed text."
    (call_dir / "ground_truth.json").write_text(json.dumps(ground), encoding="utf-8")
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--overwrite") == 0
    updated = json.loads((call_dir / "transcript.json").read_text(encoding="utf-8"))
    assert updated["audio_segments"][0]["transcript"] == "ARTIFICIAL changed text."


# --- Batch failure safety --------------------------------------------------

def test_one_malformed_call_aborts_the_whole_batch_without_writing_any_output(calls_dir, monkeypatch):
    _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    bad_dir = calls_dir / "CALL_002"
    bad_dir.mkdir(parents=True)
    (bad_dir / "ground_truth.json").write_text(json.dumps({"call_id": "CALL_002"}), encoding="utf-8")
    assert run(monkeypatch, calls_dir, "--all") == 2
    assert not (calls_dir / "CALL_001" / "transcript.json").exists()
    assert not (calls_dir / "CALL_001" / "speaker_roles.json").exists()


def test_malformed_call_error_message_names_the_call(calls_dir, monkeypatch, capsys):
    bad_dir = calls_dir / "CALL_003"
    bad_dir.mkdir(parents=True)
    (bad_dir / "ground_truth.json").write_text(json.dumps({"call_id": "CALL_003"}), encoding="utf-8")
    assert run(monkeypatch, calls_dir, "--call", "CALL_003") == 2
    assert "CALL_003" in capsys.readouterr().err


def test_missing_ground_truth_is_a_clear_error_not_a_silent_skip(calls_dir, monkeypatch, capsys):
    calls_dir.mkdir()
    assert run(monkeypatch, calls_dir, "--call", "CALL_404") == 2
    assert "CALL_404" in capsys.readouterr().err


# --- Determinism ------------------------------------------------------

def test_rebuilding_with_overwrite_is_byte_identical(calls_dir, monkeypatch):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    first = {
        name: (call_dir / name).read_bytes()
        for name in ("transcript.json", "speaker_roles.json")
    }
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--overwrite") == 0
    second = {
        name: (call_dir / name).read_bytes()
        for name in ("transcript.json", "speaker_roles.json")
    }
    assert first == second


# --- Read-only check mode --------------------------------------------------

def test_check_passes_on_freshly_built_files(calls_dir, monkeypatch):
    _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--check") == 0


def test_check_fails_and_writes_nothing_when_outputs_are_missing(calls_dir, monkeypatch, capsys):
    _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    # Missing output files are a usage problem the check cannot even attempt
    # (exit 2), distinct from a loaded-but-invalid pair (exit 1).
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--check") == 2
    assert not (calls_dir / "CALL_001" / "transcript.json").exists()
    assert "CALL_001" in capsys.readouterr().err


def test_check_fails_when_generated_pair_was_hand_tampered(calls_dir, monkeypatch, capsys):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    speaker_roles = json.loads((call_dir / "speaker_roles.json").read_text(encoding="utf-8"))
    speaker_roles["spk_0"] = "Customer"
    (call_dir / "speaker_roles.json").write_text(json.dumps(speaker_roles), encoding="utf-8")
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--check") == 1
    assert "does not match expected" in capsys.readouterr().err


def test_check_does_not_modify_any_file(calls_dir, monkeypatch):
    call_dir = _write_call(calls_dir, "CALL_001", [("EMPLOYEE", "Hi."), ("CLIENT", "Hello.")])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0
    before = {name: (call_dir / name).read_bytes() for name in ("transcript.json", "speaker_roles.json")}
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--check") == 0
    after = {name: (call_dir / name).read_bytes() for name in ("transcript.json", "speaker_roles.json")}
    assert before == after


# --- Audio-derived times from a render (--render-id) -----------------------

def _dialogue(count):
    return [("EMPLOYEE" if index % 2 == 0 else "CLIENT", f"Turn {index}.") for index in range(count)]


def _write_timings(call_dir, call_id, rows, render_id="clean_voice_A"):
    render = call_dir / "renders" / render_id
    render.mkdir(parents=True)
    (render / "timings.json").write_text(json.dumps({"call_id": call_id, "render_id": render_id, "turns": rows}), encoding="utf-8")


def _turns_of(call_dir):
    return json.loads((call_dir / "transcript.json").read_text(encoding="utf-8"))["audio_segments"]


def test_render_id_stitches_the_audio_derived_times_and_leaves_the_ground_truth_alone(calls_dir, monkeypatch):
    call_dir = _write_call(calls_dir, "CALL_001", _dialogue(3))
    before = (call_dir / "ground_truth.json").read_bytes()
    rows = [{"turn_id": i, "start_seconds": i * 2.0, "end_seconds": i * 2.0 + 1.5} for i in range(3)]
    _write_timings(call_dir, "CALL_001", rows)
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--render-id", "clean_voice_A") == 0
    assert [(s["start_time"], s["end_time"]) for s in _turns_of(call_dir)] == [("0.000", "1.500"), ("2.000", "3.500"), ("4.000", "5.500")]
    assert (call_dir / "ground_truth.json").read_bytes() == before
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--render-id", "clean_voice_A", "--check") == 0


def test_check_with_a_render_id_flags_placeholder_times(calls_dir, monkeypatch, capsys):
    call_dir = _write_call(calls_dir, "CALL_001", _dialogue(2))
    _write_timings(call_dir, "CALL_001", [{"turn_id": 0, "start_seconds": 0, "end_seconds": 1}, {"turn_id": 1, "start_seconds": 1.4, "end_seconds": 3}])
    assert run(monkeypatch, calls_dir, "--call", "CALL_001") == 0  # placeholder times (0.0)
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--check") == 0
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--render-id", "clean_voice_A", "--check") == 1
    assert "do not match the times of render clean_voice_A" in capsys.readouterr().err


@pytest.mark.parametrize("rows", [
    [{"turn_id": 0, "start_seconds": 0, "end_seconds": 1}],  # a turn is missing
    [{"turn_id": 0, "start_seconds": 0, "end_seconds": 2}, {"turn_id": 1, "start_seconds": 1, "end_seconds": 3}],  # overlapping
])
def test_bad_timings_abort_the_build_without_writing(calls_dir, monkeypatch, rows):
    call_dir = _write_call(calls_dir, "CALL_001", _dialogue(2))
    _write_timings(call_dir, "CALL_001", rows)
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--render-id", "clean_voice_A") == 2
    assert not (call_dir / "transcript.json").exists()


def test_a_missing_render_is_a_clear_error(calls_dir, monkeypatch, capsys):
    _write_call(calls_dir, "CALL_001", _dialogue(2))
    assert run(monkeypatch, calls_dir, "--call", "CALL_001", "--render-id", "nope") == 2
    assert "no timings at" in capsys.readouterr().err
