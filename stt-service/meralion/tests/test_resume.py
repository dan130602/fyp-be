"""Progress, resume and crash behaviour of one invocation (plan: Resume and progress). Every crash scenario must end identical to an uninterrupted run under stable_view (C3)."""
import json
from pathlib import Path

import pytest

import synthetic
from harness import CrashingDecoder, FlakyStorage, SimulatedCrash, raw_files, run, snapshot, two_jobs
from meralion_worker import pipeline, strategy_fg
from meralion_worker.decoders.base import Decoder
from meralion_worker.state import ResumeError, sha256_bytes
from meralion_worker.storage import LocalStorage

FIRST_PASS_CALLS = 14                  # two recordings x 7 windows


@pytest.fixture
def reference(tmp_path):
    """An uninterrupted run of the two-job world, in its own directory."""
    world = two_jobs()
    summary = run(world, tmp_path, base="reference")
    assert summary.exit_code == 0
    return {"summary": summary, "out": snapshot(tmp_path / "reference" / "out"), "state": snapshot(tmp_path / "reference" / "state")}


def finished(tmp_path, base="run"):
    return {"out": snapshot(tmp_path / base / "out"), "state": snapshot(tmp_path / base / "state")}


# ---- the uninterrupted run ---------------------------------------------------------------------------------------------------------------------------------------------

def test_an_uninterrupted_run_writes_everything_and_reports_the_plan(tmp_path):
    world = two_jobs()
    summary = run(world, tmp_path)
    assert (summary.exit_code, summary.members, summary.done, summary.failed, summary.resumed) == (0, 2, 2, 0, False)
    assert (summary.windows_total, summary.run_budget, summary.run_budget_bound, summary.retries_selected) == (14, None, False, 2)
    assert summary.jobs == {"alpha": "done", "beta": "done"}
    files = set(raw_files(tmp_path / "run" / "state"))
    assert files == {"inv-1/invocation.json", "inv-1/plan.json", "inv-1/recordings/alpha.json", "inv-1/recordings/beta.json", "inv-1/first_pass/alpha.json", "inv-1/first_pass/beta.json",
                     "inv-1/retries/alpha_r000.json", "inv-1/retries/beta_r001.json", "inv-1/done/alpha.json", "inv-1/done/beta.json"}
    assert set(raw_files(tmp_path / "run" / "out")) == {"transcripts/alpha.json", "transcripts/beta.json"}


def test_the_plan_records_members_ids_budget_and_the_first_pass_hashes(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    state_dir = tmp_path / "run" / "state" / "inv-1"
    plan = json.loads((state_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["members"] == ["alpha", "beta"] and plan["windows_total"] == 14 and plan["run_budget"] is None and plan["run_budget_bound"] is False
    assert [(g["retry_id"], g["call_id"], g["window"]) for g in plan["selected"]] == [("alpha_r000", "alpha", [600000, 1080000]), ("beta_r001", "beta", [1080000, 1560000])]
    assert plan["skipped"] == []
    for name in ("alpha", "beta"):
        assert plan["first_pass_sha256"][name] == sha256_bytes((state_dir / "first_pass" / f"{name}.json").read_bytes())


def test_recordings_are_opened_once_per_phase_that_needs_audio(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    assert world.opened == ["alpha", "beta", "alpha", "beta"]                      # first pass, then the retry phase (both have a retry)


def test_two_independent_runs_are_identical_under_stable_view(tmp_path, reference):
    run(two_jobs(), tmp_path, base="second")
    assert finished(tmp_path, "second") == {"out": reference["out"], "state": reference["state"]}


def test_rerunning_a_finished_invocation_does_nothing(tmp_path, reference):
    world = two_jobs()
    before_out, before_state = raw_files(tmp_path / "reference" / "out"), raw_files(tmp_path / "reference" / "state")
    decoder = world.decoder()
    summary = run(world, tmp_path, decoder=decoder, base="reference")
    assert summary.exit_code == 0 and summary.resumed is True and decoder.calls == []
    assert raw_files(tmp_path / "reference" / "out") == before_out and raw_files(tmp_path / "reference" / "state") == before_state


# ---- crash scenarios -----------------------------------------------------------------------------------------------------------------------------------------------------

def test_a_crash_partway_through_the_first_pass_resumes_without_redoing_finished_recordings(tmp_path, reference):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=10))
    state_dir = tmp_path / "run" / "state" / "inv-1"
    assert (state_dir / "first_pass" / "alpha.json").is_file() and not (state_dir / "first_pass" / "beta.json").exists() and not (state_dir / "plan.json").exists()
    assert not (tmp_path / "run" / "out").exists() or not list((tmp_path / "run" / "out").rglob("*.json"))
    decoder = world.decoder()
    summary = run(world, tmp_path, decoder=decoder)
    assert summary.exit_code == 0 and summary.resumed is True
    assert [c for c in decoder.calls if c[0] == "alpha"] == [("alpha", 600000, 1080000)]          # alpha: only its retry; its first pass was kept
    assert len([c for c in decoder.calls if c[0] == "beta"]) == 7 + 1
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_a_crash_after_the_plan_resumes_with_the_same_plan_and_never_recomputes_it(tmp_path, reference, monkeypatch):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=FIRST_PASS_CALLS))
    state_dir = tmp_path / "run" / "state" / "inv-1"
    plan_before = (state_dir / "plan.json").read_bytes()
    assert not list((state_dir / "retries").glob("*.json")) if (state_dir / "retries").exists() else True

    def forbidden(*args, **kwargs):
        raise AssertionError("the plan must not be recomputed on resume")

    monkeypatch.setattr(pipeline, "select_retries", forbidden)
    decoder = world.decoder()
    assert run(world, tmp_path, decoder=decoder).exit_code == 0
    assert (state_dir / "plan.json").read_bytes() == plan_before
    assert sorted(decoder.calls) == [("alpha", 600000, 1080000), ("beta", 1080000, 1560000)]       # exactly the two retry windows
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_a_crash_partway_through_the_retries_resumes_with_the_missing_retry_only(tmp_path, reference):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=FIRST_PASS_CALLS + 1))
    state_dir = tmp_path / "run" / "state" / "inv-1"
    assert (state_dir / "retries" / "alpha_r000.json").is_file() and not (state_dir / "retries" / "beta_r001.json").exists()
    assert (tmp_path / "run" / "out" / "transcripts" / "alpha.json").is_file() and (state_dir / "done" / "alpha.json").is_file()      # alpha is finished before beta's retry is attempted
    assert not (tmp_path / "run" / "out" / "transcripts" / "beta.json").exists()
    decoder = world.decoder()
    assert run(world, tmp_path, decoder=decoder).exit_code == 0
    assert decoder.calls == [("beta", 1080000, 1560000)]
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_a_crash_between_two_outputs_keeps_the_first_and_finishes_the_second(tmp_path, reference):
    world = two_jobs()
    state = LocalStorage(tmp_path / "run" / "state")
    flaky = FlakyStorage(tmp_path / "run" / "out", lambda key: key.startswith("transcripts/"), nth=2)
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, output=flaky, state=state)
    first = (tmp_path / "run" / "out" / "transcripts" / "alpha.json").read_bytes()
    assert not (tmp_path / "run" / "out" / "transcripts" / "beta.json").exists()
    decoder = world.decoder()
    summary = run(world, tmp_path, decoder=decoder)
    assert summary.exit_code == 0 and decoder.calls == []                           # every decode was persisted before the crash
    assert (tmp_path / "run" / "out" / "transcripts" / "alpha.json").read_bytes() == first   # the finished output was not touched
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_a_crash_between_an_output_and_its_done_marker_is_repaired_not_redone(tmp_path, reference):
    world = two_jobs()
    out = LocalStorage(tmp_path / "run" / "out")
    flaky_state = FlakyStorage(tmp_path / "run" / "state", lambda key: "/done/" in key, nth=1)
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, output=out, state=flaky_state)
    state_dir = tmp_path / "run" / "state" / "inv-1"
    assert (tmp_path / "run" / "out" / "transcripts" / "alpha.json").is_file() and not (state_dir / "done" / "alpha.json").exists()
    written = (tmp_path / "run" / "out" / "transcripts" / "alpha.json").read_bytes()
    assert run(world, tmp_path).exit_code == 0
    assert (tmp_path / "run" / "out" / "transcripts" / "alpha.json").read_bytes() == written          # not rewritten
    done = json.loads((state_dir / "done" / "alpha.json").read_text(encoding="utf-8"))
    assert done["output_sha256"] == sha256_bytes(written) and done["output_key"] == "transcripts/alpha.json"
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_an_output_that_does_not_match_this_run_is_not_repaired(tmp_path):
    world = two_jobs()
    flaky_state = FlakyStorage(tmp_path / "run" / "state", lambda key: "/done/" in key, nth=1)
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, output=LocalStorage(tmp_path / "run" / "out"), state=flaky_state)
    path = tmp_path / "run" / "out" / "transcripts" / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["meralion"]["audio"]["sha256"] = "0" * 64                                                # an output for different audio
    path.write_text(json.dumps(document), encoding="utf-8")
    before = raw_files(tmp_path / "run")
    with pytest.raises(ResumeError, match="did not write"):
        run(world, tmp_path)
    assert raw_files(tmp_path / "run") == before


@pytest.mark.parametrize("after", [0, 3, 7, 10, 14, 15])          # 16 decodes in total (14 first-pass windows, 2 retries): every one of these dies before the end
def test_a_crash_at_any_decode_resumes_to_the_same_result(tmp_path, reference, after):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=after))
    assert run(world, tmp_path).exit_code == 0
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_repeated_crashes_still_converge(tmp_path, reference):
    world = two_jobs()
    for after in (2, 2, 9, 3, 1):
        with pytest.raises(SimulatedCrash):
            run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=after))
    assert run(world, tmp_path).exit_code == 0
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


# ---- incompatible restarts ----------------------------------------------------------------------------------------------------------------------------------------------

class OtherIdentity(Decoder):
    def __init__(self, inner, **changes):
        self.inner, self.changes = inner, changes

    def decode(self, piece, **kwargs):
        return self.inner.decode(piece, **kwargs)

    def describe(self):
        return {**self.inner.describe(), **self.changes}


def partially_done(tmp_path):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=10))
    return world


@pytest.mark.parametrize("label", ["formatter", "legacy config", "job order", "extra job", "different input", "decoder backend", "decoder digest", "vendored sources", "model manifest", "worker version"])
def test_an_incompatible_restart_is_refused_and_changes_nothing(tmp_path, monkeypatch, label):
    world = synthetic.World({"alpha": dict(break_at=2), "beta": dict(break_at=4, replace_first="half a percent"), "gamma": dict(break_at=None)})
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, names=["alpha", "beta"], decoder=CrashingDecoder(world.decoder(), after=10))
    before = raw_files(tmp_path / "run")
    kwargs = {"names": ["alpha", "beta"]}
    if label == "formatter":
        kwargs["effective"] = synthetic.effective(formatter="none")
    elif label == "legacy config":
        kwargs["effective"] = synthetic.legacy_effective()                           # amt1 state resumed under the preserved legacy config
    elif label == "job order":
        kwargs["names"] = ["beta", "alpha"]
    elif label == "extra job":
        kwargs["names"] = ["alpha", "beta", "gamma"]
    elif label == "different input":
        kwargs["jobs"] = [pipeline.Job("alpha", "/elsewhere/alpha.wav"), pipeline.Job("beta", "/synthetic/beta.wav")]
    elif label == "decoder backend":
        kwargs["decoder"] = OtherIdentity(world.decoder(), backend="vllm")
    elif label == "decoder digest":
        kwargs["decoder"] = OtherIdentity(world.decoder(), frozen_digest="something else")
    elif label == "vendored sources":
        context = synthetic.context()
        context.vendored_sources_sha256 = "0" * 64
        kwargs["context"] = context
    elif label == "model manifest":
        context = synthetic.context()
        context.model_manifest_sha256 = "0" * 64
        kwargs["context"] = context
    elif label == "worker version":
        monkeypatch.setattr(pipeline, "__version__", "9.9.9")
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(world, tmp_path, **kwargs)
    assert raw_files(tmp_path / "run") == before


def test_a_legacy_config_state_cannot_be_resumed_under_the_amt1_default(tmp_path):
    world = synthetic.World({"alpha": dict(break_at=2), "beta": dict(break_at=4, replace_first="half a percent")})
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, names=["alpha", "beta"], effective=synthetic.legacy_effective(), decoder=CrashingDecoder(world.decoder(), after=10))
    before = raw_files(tmp_path / "run")
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(world, tmp_path, names=["alpha", "beta"])                                  # the default is now amt1
    assert raw_files(tmp_path / "run") == before


def test_a_changed_input_is_detected_by_its_hash_when_the_recording_is_reloaded(tmp_path):
    world = partially_done(tmp_path)
    before = raw_files(tmp_path / "run")

    def changed(job):
        from meralion_worker.audio import ReplayRecording
        return ReplayRecording(job.job_name, world.specs[job.job_name].get("num_samples", synthetic.SAMPLES_120S) + 16000)       # same job, different audio

    with pytest.raises(ResumeError, match="changed since it was first read"):
        run(world, tmp_path, open_recording=changed)
    assert raw_files(tmp_path / "run") == before


def test_first_pass_results_that_changed_after_the_plan_was_made_are_refused(tmp_path):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=FIRST_PASS_CALLS))
    path = tmp_path / "run" / "state" / "inv-1" / "first_pass" / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["windows"][0]["result"]["text"] += " tampered"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ResumeError, match="plan.json was made from different"):
        run(world, tmp_path)


# ---- outputs the run did not write ---------------------------------------------------------------------------------------------------------------------------------------

def test_an_existing_output_blocks_a_new_invocation_before_any_state_is_written(tmp_path):
    world = two_jobs()
    foreign = tmp_path / "run" / "out" / "transcripts" / "beta.json"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(ResumeError, match="did not write"):
        run(world, tmp_path)
    assert not (tmp_path / "run" / "state").exists() or not list((tmp_path / "run" / "state").rglob("*"))
    assert foreign.read_text(encoding="utf-8") == "{}" and world.opened == []


def test_another_invocation_cannot_overwrite_a_finished_ones_outputs(tmp_path, reference):
    world = two_jobs()
    before = raw_files(tmp_path / "reference")
    with pytest.raises(ResumeError, match="did not write"):
        run(world, tmp_path, invocation_id="inv-2", base="reference")
    assert raw_files(tmp_path / "reference") == before and not (tmp_path / "reference" / "state" / "inv-2").exists()


def test_a_foreign_output_appearing_mid_resume_blocks_the_run(tmp_path):
    world = partially_done(tmp_path)
    foreign = tmp_path / "run" / "out" / "transcripts" / "beta.json"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text(json.dumps({"meralion": {"invocation": {"id": "someone-else"}}}), encoding="utf-8")
    before = raw_files(tmp_path / "run")
    with pytest.raises(ResumeError, match="did not write"):
        run(world, tmp_path)
    assert raw_files(tmp_path / "run") == before


# ---- final input failures and runtime errors -----------------------------------------------------------------------------------------------------------------------------

def test_a_final_input_failure_is_excluded_from_the_plan_and_stays_excluded_on_restart(tmp_path):
    world = two_jobs()
    world.failing["beta"] = "Audio could not be decoded"
    summary = run(world, tmp_path)
    assert (summary.exit_code, summary.members, summary.done, summary.failed) == (2, 1, 1, 1) and summary.jobs == {"alpha": "done", "beta": "failed"}
    assert (summary.windows_total, summary.run_budget) == (7, None)                     # beta's windows are not counted; the default strategy has no budget (the budgeted one gives 1)
    state_dir = tmp_path / "run" / "state" / "inv-1"
    assert json.loads((state_dir / "failed" / "beta.json").read_text(encoding="utf-8"))["reason"] == "Audio could not be decoded"
    assert not (tmp_path / "run" / "out" / "transcripts" / "beta.json").exists() and (tmp_path / "run" / "out" / "transcripts" / "alpha.json").is_file()
    plan_before, before = (state_dir / "plan.json").read_bytes(), raw_files(tmp_path / "run")
    world.failing.clear()                                                                # even if the input were fixed, the failure is final within this invocation
    world.opened.clear()
    again = run(world, tmp_path)
    assert again.exit_code == 2 and "beta" not in world.opened and (state_dir / "plan.json").read_bytes() == plan_before
    assert raw_files(tmp_path / "run") == before


def test_all_recordings_failing_gives_exit_2_with_no_plan(tmp_path):
    world = two_jobs()
    world.failing.update(alpha="Unsupported or unknown media format", beta="Audio could not be decoded")
    summary = run(world, tmp_path)
    assert (summary.exit_code, summary.members, summary.done, summary.failed) == (2, 0, 0, 2)
    assert not (tmp_path / "run" / "state" / "inv-1" / "plan.json").exists()


def test_a_download_or_network_error_is_a_runtime_error_not_a_final_failure_and_resumes(tmp_path, reference):
    world = two_jobs()
    healthy = world.open_recording

    def flaky(job):
        if job.job_name == "beta":
            raise ConnectionError("network down")
        return healthy(job)

    with pytest.raises(ConnectionError):
        run(world, tmp_path, open_recording=flaky)
    state_dir = tmp_path / "run" / "state" / "inv-1"
    assert not (state_dir / "failed").exists()                                          # nothing was marked failed
    assert run(world, tmp_path).exit_code == 0
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_duplicate_job_names_are_rejected(tmp_path):
    world = two_jobs()
    with pytest.raises(ValueError, match="unique"):
        run(world, tmp_path, jobs=[pipeline.Job("alpha", "/a"), pipeline.Job("alpha", "/b")])


# ---- a temporary resource error on valid audio is resumable and never a permanent failed marker (review finding 1) ------------------------------------------------------------

SHORT_SAMPLES = 20 * 16000


def short_recording(tmp_path, monkeypatch, name="short"):
    """A real 20 s silent WAV (one FG window), the package's audio loader replaced by a passthrough, and the scripted text the replay decoder will answer."""
    import numpy as np
    import soundfile
    from meralion_worker import audio
    monkeypatch.setattr(audio, "_package_load_audio", lambda x: np.asarray(x[0], dtype=np.float32))
    path = tmp_path / "in" / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.zeros(SHORT_SAMPLES, dtype=np.float32), 16000, subtype="PCM_16")
    return pipeline.Job(name, str(path)), {(name, 0, SHORT_SAMPLES): "hello world"}


def real_opener():
    from meralion_worker import audio
    return lambda job: audio.load_recording(job.job_name, job.input)


@pytest.mark.parametrize("fault", ["soundfile read", "package loader"])
@pytest.mark.parametrize("error", [MemoryError("temporary"), OSError("disk hiccup"), ImportError("numba not ready"),
                                   *[__import__("soundfile").LibsndfileError(code) for code in (156, 157, 161, 17, 2, 8, 9, 9999)]],
                         ids=lambda e: f"{type(e).__name__}-{getattr(e, 'code', '')}")
def test_a_resource_error_on_valid_audio_is_resumable_and_never_a_permanent_failure(tmp_path, monkeypatch, fault, error):
    import soundfile
    from meralion_worker import audio
    from meralion_worker.decoders.replay_decoder import ReplayDecoder, build_table
    job, entries = short_recording(tmp_path, monkeypatch)
    table = build_table(entries.items())
    armed = {"on": True}

    def failing_once(real):
        def wrapper(*args, **kwargs):
            if armed["on"]:
                armed["on"] = False
                raise error
            return real(*args, **kwargs)
        return wrapper

    if fault == "soundfile read":
        monkeypatch.setattr(soundfile, "read", failing_once(soundfile.read))
    else:
        monkeypatch.setattr(audio, "_package_load_audio", failing_once(audio._package_load_audio))
    with pytest.raises(type(error)):
        run(None, tmp_path, jobs=[job], decoder=ReplayDecoder(table), open_recording=real_opener())
    state_dir = tmp_path / "run" / "state" / "inv-1"
    assert not (state_dir / "failed").exists() and not (state_dir / "recordings").exists()          # no permanent marker, nothing half-recorded
    summary = run(None, tmp_path, jobs=[job], decoder=ReplayDecoder(table), open_recording=real_opener())       # the fault is gone: the SAME invocation finishes the recording
    assert (summary.exit_code, summary.done, summary.failed, summary.resumed) == (0, 1, 0, True)
    document = json.loads((tmp_path / "run" / "out" / "transcripts" / "short.json").read_text(encoding="utf-8"))
    assert document["results"]["transcripts"][0]["transcript"] == "hello world"


def test_a_genuinely_bad_file_is_still_a_permanent_final_failure_while_valid_audio_proceeds(tmp_path, monkeypatch):
    from meralion_worker.decoders.replay_decoder import ReplayDecoder, build_table
    good, entries = short_recording(tmp_path, monkeypatch, "good")
    bad = tmp_path / "in" / "bad.wav"
    bad.write_bytes(b"this is not audio at all" * 20)
    jobs = [pipeline.Job("bad", str(bad)), good]
    table = build_table(entries.items())
    summary = run(None, tmp_path, jobs=jobs, decoder=ReplayDecoder(table), open_recording=real_opener())
    assert (summary.exit_code, summary.done, summary.failed) == (2, 1, 1) and summary.jobs == {"bad": "failed", "good": "done"}
    marker = json.loads((tmp_path / "run" / "state" / "inv-1" / "failed" / "bad.json").read_text(encoding="utf-8"))
    assert marker["reason"] == "Audio could not be decoded"
    again = run(None, tmp_path, jobs=jobs, decoder=ReplayDecoder(table), open_recording=real_opener())          # the failure is final within the invocation
    assert again.exit_code == 2 and again.jobs == {"bad": "failed", "good": "done"}


# ---- a ready recording is saved before the next recording is touched (review finding 2) -------------------------------------------------------------------------------------------

def test_a_recording_that_is_ready_is_written_even_when_another_recordings_retry_then_fails(tmp_path, reference):
    """The reproduced scenario: alpha's retries finish, then beta's retry crashes. alpha's transcript must already be on disk, finished and marked done."""
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=FIRST_PASS_CALLS + 1))          # call 15 = alpha's retry (succeeds), call 16 = beta's retry (dies)
    out, state_dir = tmp_path / "run" / "out" / "transcripts", tmp_path / "run" / "state" / "inv-1"
    assert (out / "alpha.json").is_file() and (state_dir / "done" / "alpha.json").is_file()
    assert not (out / "beta.json").exists() and not (state_dir / "done" / "beta.json").exists() and not (state_dir / "retries" / "beta_r001.json").exists()
    alpha_on_disk = (out / "alpha.json").read_bytes()
    decoder = world.decoder()
    assert run(world, tmp_path, decoder=decoder).exit_code == 0
    assert decoder.calls == [("beta", 1080000, 1560000)]                                                     # only beta's missing retry; alpha is never decoded again
    assert (out / "alpha.json").read_bytes() == alpha_on_disk
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_recordings_without_retries_are_finished_in_their_turn_too(tmp_path):
    """alpha (one retry), gamma (no retry), beta (one retry), in that order: a crash on beta's retry leaves alpha AND gamma finished."""
    world = synthetic.World({"alpha": dict(break_at=2), "gamma": dict(break_at=None), "beta": dict(break_at=4)})
    total_first_pass = 21
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, names=["alpha", "gamma", "beta"], decoder=CrashingDecoder(world.decoder(), after=total_first_pass + 1))
    out = tmp_path / "run" / "out" / "transcripts"
    assert (out / "alpha.json").is_file() and (out / "gamma.json").is_file() and not (out / "beta.json").exists()
    decoder = world.decoder()
    assert run(world, tmp_path, names=["alpha", "gamma", "beta"], decoder=decoder).exit_code == 0
    assert decoder.calls == [("beta", 1080000, 1560000)]


def expected_resume_calls(tmp_path, plan):
    """What a resume must decode, computed from the crashed state and the plan: the first-pass windows of recordings that have no first pass, plus the planned retries of recordings that
    are not finished and lack their retry file. Finished recordings (a done marker) contribute nothing."""
    state_dir = tmp_path / "run" / "state" / "inv-1"
    calls = []
    for name in ("alpha", "beta"):
        if (state_dir / "done" / f"{name}.json").exists():
            continue
        if not (state_dir / "first_pass" / f"{name}.json").exists():
            calls += [(name, a, b) for a, b in strategy_fg.windows_for(synthetic.SAMPLES_120S)]
        for group in plan["selected"]:
            if group["call_id"] == name and not (state_dir / "retries" / f"{group['retry_id']}.json").exists():
                calls.append((name, group["window"][0], group["window"][1]))
    return sorted(calls)


@pytest.mark.parametrize("after", range(16))
def test_after_a_crash_at_any_decode_only_unfinished_work_is_decoded_and_finished_recordings_never_again(tmp_path, reference, after):
    world = two_jobs()
    with pytest.raises(SimulatedCrash):
        run(world, tmp_path, decoder=CrashingDecoder(world.decoder(), after=after))
    state_dir = tmp_path / "run" / "state" / "inv-1"
    finished_before = {name for name in ("alpha", "beta") if (state_dir / "done" / f"{name}.json").exists()}
    reference_plan = json.loads((Path(tmp_path) / "reference" / "state" / "inv-1" / "plan.json").read_text(encoding="utf-8"))
    expected = expected_resume_calls(tmp_path, reference_plan)
    decoder = world.decoder()
    assert run(world, tmp_path, decoder=decoder).exit_code == 0
    assert sorted(decoder.calls) == expected
    assert not any(call[0] in finished_before for call in decoder.calls)                          # a finished recording is never decoded again
    assert finished(tmp_path) == {"out": reference["out"], "state": reference["state"]}


def test_a_decode_error_leaves_no_failed_marker_and_the_same_invocation_finishes_after_recovery(tmp_path):
    """A vLLM abort or error surfaces as DecodeError: a runtime failure (resumable), never a permanent input failure and never an output."""
    from meralion_worker.decoders.base import DecodeError

    class AbortingOnce(CrashingDecoder):
        def decode(self, piece, *, recording_id, start_sample, end_sample):
            self.count += 1
            if self.count == 5:
                raise DecodeError("vLLM did not complete request 4 of 7: finish_reason='abort', stop_reason=None")
            return self.inner.decode(piece, recording_id=recording_id, start_sample=start_sample, end_sample=end_sample)

    world = two_jobs()
    with pytest.raises(DecodeError):
        run(world, tmp_path, decoder=AbortingOnce(world.decoder(), after=10 ** 6))
    state = tmp_path / "run" / "state" / "inv-1"
    assert not (state / "failed").exists() and not any((tmp_path / "run" / "out").rglob("*.json"))
    summary = run(world, tmp_path)
    assert (summary.exit_code, summary.done, summary.failed) == (0, 2, 0)
