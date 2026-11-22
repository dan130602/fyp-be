"""The `run` command end to end. run_checks and make_decoder are replaced (the real ones need the 6 GB checkpoint, the installed package and a GPU); everything else is the real command line."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

import synthetic
from harness import CrashingDecoder, SimulatedCrash, raw_files
from meralion_worker import audio, cli
from meralion_worker import model_check
from meralion_worker.decoders.base import DecodeError

SECONDS = 120


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Real 120 s silent WAV files, the packaged decoder replaced by a replay decoder, the identity checks replaced by a stub, and the package's audio loader replaced by a mono passthrough."""
    monkeypatch.setattr(audio, "_package_load_audio", lambda x: np.asarray(x[0], dtype=np.float32) if np.asarray(x[0]).ndim == 1 else np.asarray(x[0], dtype=np.float32).mean(axis=1))
    world = synthetic.World({"alpha": dict(break_at=2), "beta": dict(break_at=4, replace_first="half a percent")})
    (tmp_path / "in").mkdir()
    for name in ("alpha", "beta"):
        soundfile.write(str(tmp_path / "in" / f"{name}.wav"), np.zeros(SECONDS * 16000, dtype=np.float32), 16000, subtype="PCM_16")
    decoders = []

    def fake_decoder(effective):
        decoder = world.decoder()
        decoders.append(decoder)
        return decoder

    monkeypatch.setattr(cli, "make_decoder", fake_decoder)
    monkeypatch.setattr(cli, "run_checks", lambda effective: {"manifest": {"id": "test-model", "revision": "rev"}, "model": {"manifest_sha256": "m" * 64, "seconds": 0.5},
                                                              "vendored": {"commit": "c", "files": {}, "sources_sha256": "v" * 64}, "package": {}})
    monkeypatch.delenv("TRANSCRIBE_OUTPUT_BUCKET", raising=False)
    monkeypatch.delenv("MERALION_BACKEND", raising=False)
    return type("Env", (), {"root": tmp_path, "world": world, "decoders": decoders})


def jobs_file(env, names=("alpha", "beta"), suffix=".wav"):
    path = env.root / "jobs.json"
    path.write_text(json.dumps({"jobs": [{"job_name": n, "input": str(env.root / "in" / f"{n}{suffix}")} for n in names]}), encoding="utf-8")
    return path


def argv(env, *extra, jobs=None, invocation="inv-1", out="out", state="state"):
    return ["run", "--invocation-id", invocation, "--jobs", str(jobs or jobs_file(env)), "--output-dir", str(env.root / out), "--state-dir", str(env.root / state), *extra]


def last_json(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# ---- success, partial failure ------------------------------------------------------------------------------------------------------------------------------------------

def test_run_exits_0_prints_the_summary_last_and_writes_it_to_a_file(env, capsys):
    summary_file = env.root / "summary.json"
    assert cli.main(argv(env, "--summary-output", str(summary_file))) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip().splitlines()[-1])
    assert summary["ok"] is True and summary["exit_code"] == 0 and summary["members"] == 2 and summary["done"] == 2 and summary["failed"] == 0
    assert summary["windows_total"] == 14 and summary["run_budget"] is None and summary["retries_selected"] == 2 and summary["jobs"] == {"alpha": "done", "beta": "done"}
    assert json.loads(summary_file.read_text(encoding="utf-8")) == summary
    assert "alpha: first pass" in captured.err and "beta: written" in captured.err            # progress goes to stderr, the summary to stdout
    assert (env.root / "out" / "transcripts" / "alpha.json").is_file()
    document = json.loads((env.root / "out" / "transcripts" / "beta.json").read_text(encoding="utf-8"))
    assert "0.5 percent" in document["results"]["transcripts"][0]["transcript"]


def test_run_exits_2_when_one_recording_is_unusable_and_the_others_are_done(env, capsys):
    (env.root / "in" / "beta.m4a").write_bytes(b"not supported")
    jobs = env.root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "alpha", "input": str(env.root / "in" / "alpha.wav")}, {"job_name": "beta", "input": str(env.root / "in" / "beta.m4a")}]}), encoding="utf-8")
    assert cli.main(argv(env, jobs=jobs)) == 2
    summary = last_json(capsys)
    assert summary["ok"] is False and summary["exit_code"] == 2 and summary["jobs"] == {"alpha": "done", "beta": "failed"} and summary["done"] == 1 and summary["failed"] == 1
    assert set(raw_files(env.root / "out")) == {"transcripts/alpha.json"}
    failure = json.loads((env.root / "state" / "inv-1" / "failed" / "beta.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "Unsupported or unknown media format"


def test_a_single_job_can_be_given_without_a_jobs_file(env, capsys):
    code = cli.main(["run", "--invocation-id", "one", "--job-name", "alpha", "--input", str(env.root / "in" / "alpha.wav"), "--output-dir", str(env.root / "out"), "--state-dir", str(env.root / "state")])
    assert code == 0 and last_json(capsys)["jobs"] == {"alpha": "done"}


def test_the_effective_decoder_is_the_one_make_decoder_returns_and_the_config_fingerprint_is_reported(env, capsys):
    assert cli.main(argv(env)) == 0
    assert len(env.decoders) == 1 and last_json(capsys)["config_fingerprint"] == synthetic.effective().fingerprint


# ---- runtime error and resume ------------------------------------------------------------------------------------------------------------------------------------------

def test_a_runtime_failure_exits_1_and_the_same_invocation_id_then_finishes(env, capsys, monkeypatch):
    monkeypatch.setattr(cli, "make_decoder", lambda effective: CrashingDecoder(env.world.decoder(), after=3))
    # the pipeline does not catch the injected crash type, but main() reports any runtime failure of `run` as exit 1
    assert cli.main(argv(env)) == 1
    failure = last_json(capsys)
    assert failure["ok"] is False and failure["exit_code"] == 1 and failure["invocation_id"] == "inv-1" and "SimulatedCrash" in failure["error"]
    monkeypatch.setattr(cli, "make_decoder", lambda effective: env.world.decoder())
    assert cli.main(argv(env)) == 0
    assert last_json(capsys)["resumed"] is True


def test_a_decode_error_is_a_runtime_failure(env, capsys, monkeypatch):
    class Failing(CrashingDecoder):
        def decode(self, *args, **kwargs):
            raise DecodeError("engine died")

    monkeypatch.setattr(cli, "make_decoder", lambda effective: Failing(env.world.decoder(), after=0))
    assert cli.main(argv(env)) == 1 and "engine died" in last_json(capsys)["error"]


# ---- exit 3: usage, configuration and compatibility -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bad id", "", "a/b", "id\n", "x" * 129])
def test_a_bad_invocation_id_exits_3(env, capsys, bad):
    assert cli.main(argv(env, invocation=bad)) == 3
    assert last_json(capsys)["exit_code"] == 3 and not (env.root / "state").exists()


def test_duplicate_job_names_exit_3(env, capsys):
    jobs = env.root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "alpha", "input": "/a.wav"}, {"job_name": "alpha", "input": "/b.wav"}]}), encoding="utf-8")
    assert cli.main(argv(env, jobs=jobs)) == 3 and "unique" in last_json(capsys)["error"]


@pytest.mark.parametrize("content, text", [
    ("not json", "cannot read the jobs file"),
    ("{}", "at least one job"),
    ('{"jobs": []}', "at least one job"),
    ('{"jobs": [{"job_name": "a"}]}', "each job must be"),
    ('{"jobs": [{"job_name": "a", "input": "/x", "extra": 1}]}', "each job must be"),
    ('{"jobs": [{"job_name": "bad name", "input": "/x"}]}', "invalid job name"),
    ('{"jobs": [{"job_name": "a\\n", "input": "/x"}]}', "invalid job name"),
    ('{"jobs": [{"job_name": "a", "input": ""}]}', "each job must be"),
])
def test_a_bad_jobs_file_exits_3(env, capsys, content, text):
    path = env.root / "jobs.json"
    path.write_text(content, encoding="utf-8")
    assert cli.main(argv(env, jobs=path)) == 3 and text in last_json(capsys)["error"]


def test_jobs_file_and_single_job_arguments_cannot_be_mixed_and_one_must_be_given(env, capsys):
    assert cli.main(argv(env, "--job-name", "x")) == 3 and "not both" in last_json(capsys)["error"]
    assert cli.main(["run", "--invocation-id", "i", "--output-dir", "o", "--state-dir", "s"]) == 3 and "give --jobs" in last_json(capsys)["error"]


@pytest.mark.parametrize("extra", [
    ["--output-dir", "o"], ["--state-dir", "s"],                                       # one of the pair missing
    ["--output-dir", "o", "--state-uri", "s3://b/p"],                                  # a mix of local and S3
    ["--state-dir", "s", "--output-bucket", "b"],
    ["--output-bucket", "b"],                                                          # S3 without a state location
    ["--state-uri", "s3://b/p"],                                                       # S3 without an output bucket (and no environment default)
    [],
])
def test_the_location_arguments_must_be_one_complete_pair(env, capsys, extra):
    base = ["run", "--invocation-id", "i", "--jobs", str(jobs_file(env))]
    assert cli.main(base + extra) == 3
    assert last_json(capsys)["exit_code"] == 3 and not (env.root / "out").exists()


def test_an_incompatible_resume_exits_3_and_changes_nothing(env, capsys):
    assert cli.main(argv(env)) == 0
    capsys.readouterr()
    before = raw_files(env.root / "state"), raw_files(env.root / "out")
    assert cli.main(argv(env, jobs=jobs_file(env, names=("beta", "alpha")))) == 3          # same id, different job order
    assert "incompatible resume" in last_json(capsys)["error"]
    assert (raw_files(env.root / "state"), raw_files(env.root / "out")) == before


def test_a_failing_identity_check_exits_3_before_any_work(env, capsys, monkeypatch):
    def failing(effective):
        raise model_check.CheckError("model file hash differs: model-00001-of-00002.safetensors")

    monkeypatch.setattr(cli, "run_checks", failing)
    assert cli.main(argv(env)) == 3
    assert "hash differs" in last_json(capsys)["error"] and not (env.root / "out").exists() and not (env.root / "state").exists() and env.decoders == []


def test_a_decoder_that_cannot_be_set_up_as_frozen_exits_3(env, capsys, monkeypatch):
    from meralion_worker.decoders.base import DecodingSettingsError

    def refusing(effective):
        raise DecodingSettingsError("decoding settings differ from the frozen table")

    monkeypatch.setattr(cli, "make_decoder", refusing)
    assert cli.main(argv(env)) == 3 and "frozen table" in last_json(capsys)["error"]
    assert not (env.root / "out").exists()


def test_a_foreign_output_exits_3(env, capsys):
    foreign = env.root / "out" / "transcripts" / "alpha.json"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("{}", encoding="utf-8")
    assert cli.main(argv(env)) == 3 and "did not write" in last_json(capsys)["error"] and foreign.read_text(encoding="utf-8") == "{}"


def test_the_run_command_has_no_option_that_skips_the_checks():
    parser = cli.build_parser()
    run = next(action for action in parser._subparsers._group_actions[0].choices["run"]._actions)
    options = {opt for action in parser._subparsers._group_actions[0].choices["run"]._actions for opt in action.option_strings}
    assert not any("skip" in o or "no-check" in o or "unsafe" in o for o in options)
    assert {"--invocation-id", "--jobs", "--job-name", "--input", "--output-dir", "--state-dir", "--output-bucket", "--state-uri", "--config", "--backend", "--model-dir", "--summary-output"} <= options


# ---- S3 mode (a fake bucket store; no network) -------------------------------------------------------------------------------------------------------------------------

class FakeBuckets:
    """An in-memory stand-in for S3Storage, shared by every instance: {bucket: {key: bytes}}."""

    def __init__(self):
        self.data = {}
        self.created = []

    def factory(self):
        store = self

        class FakeS3Storage:
            def __init__(self, bucket, prefix="", *, region=None, client=None):
                self.bucket, self.prefix, self.region = bucket, prefix.strip("/"), region
                store.created.append((bucket, self.prefix, region))

            def _key(self, key):
                return f"{self.prefix}/{key}" if self.prefix else key

            def exists(self, key):
                return self._key(key) in store.data.get(self.bucket, {})

            def read(self, key):
                from meralion_worker.storage import ObjectNotFound
                try:
                    return store.data[self.bucket][self._key(key)]
                except KeyError:
                    raise ObjectNotFound(key)

            def write_exclusive(self, key, data, content_type="application/json"):
                from meralion_worker.storage import AlreadyExists
                bucket = store.data.setdefault(self.bucket, {})
                if self._key(key) in bucket:
                    raise AlreadyExists(key)
                bucket[self._key(key)] = data

            def download(self, uri, dest_dir):
                from meralion_worker.storage import ObjectNotFound, parse_s3_uri
                bucket, key = parse_s3_uri(uri)
                try:
                    body = store.data[bucket][key]
                except KeyError:
                    raise ObjectNotFound(uri)
                path = Path(dest_dir) / Path(key).name
                path.write_bytes(body)
                return path

        return FakeS3Storage


def test_s3_mode_reads_s3_inputs_and_writes_outputs_and_state_to_their_buckets(env, capsys, monkeypatch):
    buckets = FakeBuckets()
    for name in ("alpha", "beta"):
        buckets.data.setdefault("in-bucket", {})[f"uploads/u1/{name}.wav"] = (env.root / "in" / f"{name}.wav").read_bytes()
    monkeypatch.setattr(cli, "S3Storage", buckets.factory())
    jobs = env.root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": n, "input": f"s3://in-bucket/uploads/u1/{n}.wav"} for n in ("alpha", "beta")]}), encoding="utf-8")
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_BUCKET", "out-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    code = cli.main(["run", "--invocation-id", "nightly-1", "--jobs", str(jobs), "--state-uri", "s3://state-bucket/jobs/nightly"])
    assert code == 0 and last_json(capsys)["done"] == 2
    assert set(buckets.data["out-bucket"]) == {"transcripts/alpha.json", "transcripts/beta.json"}
    assert "jobs/nightly/nightly-1/plan.json" in buckets.data["state-bucket"] and "jobs/nightly/nightly-1/done/beta.json" in buckets.data["state-bucket"]
    assert any(r == "ap-southeast-1" for _, _, r in buckets.created)
    document = json.loads(buckets.data["out-bucket"]["transcripts/beta.json"].decode("utf-8"))
    assert "0.5 percent" in document["results"]["transcripts"][0]["transcript"] and document["meralion"]["audio"]["input"] == "s3://in-bucket/uploads/u1/beta.wav"


def test_the_output_bucket_option_beats_the_environment_default(env, capsys, monkeypatch):
    buckets = FakeBuckets()
    monkeypatch.setattr(cli, "S3Storage", buckets.factory())
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_BUCKET", "env-bucket")
    code = cli.main(["run", "--invocation-id", "i", "--jobs", str(jobs_file(env, names=("alpha",))), "--output-bucket", "cli-bucket", "--state-uri", "s3://st/p"])
    assert code == 0 and "transcripts/alpha.json" in buckets.data["cli-bucket"] and "env-bucket" not in buckets.data


def test_a_missing_s3_input_is_a_runtime_error_exit_1(env, capsys, monkeypatch):
    buckets = FakeBuckets()
    monkeypatch.setattr(cli, "S3Storage", buckets.factory())
    jobs = env.root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "alpha", "input": "s3://in-bucket/missing.wav"}]}), encoding="utf-8")
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_BUCKET", "out-bucket")
    assert cli.main(["run", "--invocation-id", "i", "--jobs", str(jobs), "--state-uri", "s3://st/p"]) == 1
    assert "ObjectNotFound" in last_json(capsys)["error"] and "failed" not in buckets.data.get("st", {})


def test_a_missing_local_input_is_a_runtime_error_exit_1(env, capsys):
    jobs = env.root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "alpha", "input": str(env.root / "in" / "nope.wav")}]}), encoding="utf-8")
    assert cli.main(argv(env, jobs=jobs)) == 1 and "ObjectNotFound" in last_json(capsys)["error"]


# ---- a temporary resource error reading valid audio: exit 1, no permanent marker, the rerun finishes (review finding 1) ---------------------------------------------------------

@pytest.mark.parametrize("error", [MemoryError("temporary"), soundfile.LibsndfileError(156), soundfile.LibsndfileError(9999)], ids=["MemoryError", "flac-156", "unknown-9999"])
def test_a_temporary_memory_error_reading_valid_audio_exits_1_and_the_rerun_exits_0(env, capsys, monkeypatch, error):
    armed = {"on": True}
    real_read = soundfile.read

    def flaky_read(*args, **kwargs):
        if armed["on"]:
            armed["on"] = False
            raise error
        return real_read(*args, **kwargs)

    monkeypatch.setattr(soundfile, "read", flaky_read)
    assert cli.main(argv(env)) == 1
    failure = last_json(capsys)
    assert failure["ok"] is False and failure["exit_code"] == 1 and type(error).__name__ in failure["error"]
    assert not (env.root / "state" / "inv-1" / "failed").exists()
    assert cli.main(argv(env)) == 0
    summary = last_json(capsys)
    assert summary["done"] == 2 and summary["failed"] == 0 and summary["resumed"] is True
