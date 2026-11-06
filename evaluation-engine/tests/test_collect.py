"""The collect phase against fake S3 and Transcribe clients: the REAL production handler starts the jobs, nothing touches AWS."""
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from transcribe_eval import collect as engine
from transcribe_eval.config import load_settings
from transcribe_eval.handler_loader import load_handler

BUCKET = "fyp-eval-bucket"


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"Metadata": self.objects[Key]["meta"]}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.uploads.append(key)
        self.objects[key] = {"body": open(filename, "rb").read(), "meta": (ExtraArgs or {}).get("Metadata", {})}

    def copy_object(self, Bucket, Key, CopySource, MetadataDirective):
        self.objects[Key] = dict(self.objects[CopySource["Key"]])

    def get_object(self, Bucket, Key):
        return {"Body": SimpleNamespace(read=lambda: self.objects[Key]["body"])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


class FakeTranscribe:
    """start_transcription_job is called by the real handler; a job finishes after `polls` status checks."""

    def __init__(self, s3, polls=1, failing=(), never=()):
        self.s3, self.polls, self.failing, self.never, self.jobs, self.checks = s3, polls, failing, never, {}, {}

    def start_transcription_job(self, **request):
        self.jobs[request["TranscriptionJobName"]] = request
        self.checks[request["TranscriptionJobName"]] = 0

    def get_transcription_job(self, TranscriptionJobName):
        request = self.jobs[TranscriptionJobName]
        self.checks[TranscriptionJobName] += 1
        key = request["Media"]["MediaFileUri"].split("/", 3)[3]
        job = {"TranscriptionJobName": TranscriptionJobName, "Media": request["Media"], "Settings": request["Settings"],
               "LanguageCode": request["LanguageCode"], "TranscriptionJobStatus": "IN_PROGRESS"}
        if any(tag in key for tag in self.never):
            return {"TranscriptionJob": job}
        if self.checks[TranscriptionJobName] > self.polls:
            if any(tag in key for tag in self.failing):
                job.update(TranscriptionJobStatus="FAILED", FailureReason="The media format is not valid")
            else:
                job["TranscriptionJobStatus"] = "COMPLETED"
                self.s3.objects[request["OutputKey"]] = {"body": json.dumps({"results": {"transcripts": [{"transcript": key}]}}).encode(), "meta": {}}
        return {"TranscriptionJob": job}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root).mkdir()
    (root / "dataset_manifest.json").write_text(json.dumps({"dataset_id": "toy_v1"}), encoding="utf-8")
    for number in (1, 2):
        render = root / "calls" / f"CALL_00{number}" / "renders" / "clean_voice_A"
        render.mkdir(parents=True)
        (render / "audio.mp3").write_bytes(b"audio-%d" % number)
        (render / "timings.json").write_text(json.dumps({"duration_seconds": 60}), encoding="utf-8")
    return root


@pytest.fixture
def world(monkeypatch, tmp_path):
    settings = load_settings({"EVAL_BUCKET": BUCKET}, tmp_path / "none")
    s3 = FakeS3()
    box = SimpleNamespace(settings=settings, s3=s3, transcribe=None)

    def build(**kwargs):
        box.transcribe = FakeTranscribe(s3, **kwargs)
        monkeypatch.setattr("boto3.client", lambda *a, **k: box.transcribe)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
        for name, value in settings.handler_environment().items():
            monkeypatch.setenv(name, value)
        box.handler = load_handler(settings.handler_path)
        return box

    box.build = build
    return box


def run(world, dataset, tmp_path, calls=("CALL_001", "CALL_002"), **kwargs):
    return engine.collect(world.settings, world.handler, world.s3, dataset_root=dataset, render_id="clean_voice_A", call_ids=list(calls),
                          runs_root=tmp_path / "runs", sleep=lambda seconds: None, progress=lambda message: None, **kwargs)


def test_a_successful_run_saves_transcripts_and_metadata_and_cleans_up(world, dataset, tmp_path):
    world.build()
    metadata = run(world, dataset, tmp_path, run_id="r1")
    folder = tmp_path / "runs" / "r1"
    assert metadata["status"] == "completed"
    for call in ("CALL_001", "CALL_002"):
        saved = json.loads((folder / "raw" / "transcripts" / f"{call}.json").read_text(encoding="utf-8"))
        assert call in saved["results"]["transcripts"][0]["transcript"]
    saved_meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    first = saved_meta["calls"][0]
    assert first["status"] == "COMPLETED" and first["job"]["Settings"] == {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2}
    assert first["run_key"] == "uploads/eval-r1/toy_v1/CALL_001.mp3" and first["master_key"] == "eval-audio/toy_v1/clean_voice_A/CALL_001.mp3"
    assert len(first["audio_sha256"]) == 64 and saved_meta["handler"]["sha256"] and saved_meta["handler_environment"]["TRANSCRIBE_OUTPUT_BUCKET"] == BUCKET
    assert not any(key.startswith("uploads/") for key in world.s3.objects)  # temporary copies deleted
    assert "eval-audio/toy_v1/clean_voice_A/CALL_001.mp3" in world.s3.objects  # master audio kept
    assert "AWS_SECRET" not in json.dumps(saved_meta)


def test_the_master_audio_is_uploaded_once_and_again_only_when_it_changes(world, dataset, tmp_path):
    world.build()
    run(world, dataset, tmp_path, run_id="r1")
    assert len(world.s3.uploads) == 2
    run(world, dataset, tmp_path, run_id="r2")
    assert len(world.s3.uploads) == 2  # reused
    (dataset / "calls" / "CALL_001" / "renders" / "clean_voice_A" / "audio.mp3").write_bytes(b"re-rendered")
    run(world, dataset, tmp_path, run_id="r3")
    assert world.s3.uploads[-1].endswith("CALL_001.mp3") and len(world.s3.uploads) == 3


def test_a_failed_job_is_recorded_not_raised_and_the_other_calls_still_complete(world, dataset, tmp_path):
    world.build(failing=("CALL_002",))
    metadata = run(world, dataset, tmp_path, run_id="r1")
    states = {e["call_id"]: (e["status"], e["failure_reason"]) for e in metadata["calls"]}
    assert metadata["status"] == "completed_with_failures"
    assert states == {"CALL_001": ("COMPLETED", None), "CALL_002": ("FAILED", "The media format is not valid")}
    assert (tmp_path / "runs" / "r1" / "raw" / "transcripts" / "CALL_001.json").exists()
    assert not (tmp_path / "runs" / "r1" / "raw" / "transcripts" / "CALL_002.json").exists()


def test_a_timeout_is_recorded_and_keeps_the_uploads_a_running_job_still_needs(world, dataset, tmp_path):
    world.build(never=("CALL_001",))
    ticks = iter(range(0, 10_000, 100))
    metadata = engine.collect(world.settings, world.handler, world.s3, dataset_root=dataset, render_id="clean_voice_A", call_ids=["CALL_001"],
                              run_id="r1", runs_root=tmp_path / "runs", sleep=lambda s: None, clock=lambda: next(ticks), timeout_seconds=300,
                              progress=lambda message: None)
    assert metadata["calls"][0]["status"] == "TIMEOUT" and metadata["status"] == "completed_with_failures"
    assert "uploads/eval-r1/toy_v1/CALL_001.mp3" in world.s3.objects


def test_missing_audio_or_an_existing_run_folder_stops_before_anything_is_uploaded(world, dataset, tmp_path):
    world.build()
    with pytest.raises(engine.CollectError, match="no audio"):
        run(world, dataset, tmp_path, calls=("CALL_001", "CALL_009"))
    assert not world.s3.objects
    run(world, dataset, tmp_path, run_id="r1")
    with pytest.raises(engine.CollectError, match="already exists"):
        run(world, dataset, tmp_path, run_id="r1")


def test_the_plan_needs_no_aws_and_reports_sizes_and_minutes(dataset):
    planned = engine.plan(dataset, "clean_voice_A", ["CALL_001", "CALL_002"], BUCKET)
    assert planned["dataset_id"] == "toy_v1" and planned["audio_minutes"] == 2.0
    assert planned["calls"][0]["master_key"] == "eval-audio/toy_v1/clean_voice_A/CALL_001.mp3"
