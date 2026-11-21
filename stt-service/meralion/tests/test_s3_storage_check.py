"""tools/ec2/s3_storage_check.py against the fake S3 client of test_storage.py (no network, no AWS, no boto3, no decoder)."""
import importlib.util
from pathlib import Path

import pytest

from meralion_worker import storage as st
from test_storage import FakeClientError, FakeS3

TOOL = Path(__file__).resolve().parent.parent / "tools" / "ec2" / "s3_storage_check.py"
spec = importlib.util.spec_from_file_location("s3_storage_check_under_test", TOOL)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

INPUT = "s3://test-bucket/inputs/CALL_064.wav"


def rig(client):
    return st.S3Storage("test-bucket", "", client=client), st.S3Storage("test-bucket", "meralion-state/s3-check-1", client=client)


def seeded():
    client = FakeS3()
    client.objects[("test-bucket", "inputs/CALL_064.wav")] = b"RIFFxxxxWAVE"
    return client


def test_a_healthy_bucket_passes_every_check_and_writes_only_under_the_state_prefix():
    client = seeded()
    results = check.run_checks(*[rig(client)[0], INPUT, rig(client)[1], "check.json"])
    assert {name: item["ok"] for name, item in results.items()} == {"download_input": True, "missing_key_is_absent": True, "conditional_write": True, "read_back_equals": True,
                                                                     "exists_after_write": True, "second_write_refused": True}
    puts = [call for call in client.calls if call[0] == "put"]
    assert puts and all(call[2].startswith("meralion-state/s3-check-1/") and call[4] == "*" for call in puts)             # only the state prefix, always conditional
    assert not any(call[0] in ("delete", "delete_object") for call in client.calls) and not hasattr(client, "delete_object")


def test_a_missing_input_is_reported_not_raised_and_the_other_checks_still_run():
    client = FakeS3()
    results = check.run_checks(*[rig(client)[0], INPUT, rig(client)[1], "check.json"])
    assert results["download_input"]["ok"] is False and "ObjectNotFound" in results["download_input"]["detail"]
    assert results["conditional_write"]["ok"] and results["second_write_refused"]["ok"]


def test_access_denied_on_the_existence_check_is_a_failure_with_the_listbucket_hint_not_a_false_absent():
    class NoList(FakeS3):
        def head_object(self, Bucket, Key):
            raise FakeClientError("403")

    client = NoList()
    client.objects[("test-bucket", "inputs/CALL_064.wav")] = b"x"
    results = check.run_checks(*[rig(client)[0], INPUT, rig(client)[1], "check.json"])
    assert results["missing_key_is_absent"]["ok"] is False and "ListBucket" in results["missing_key_is_absent"]["detail"]
    assert results["exists_after_write"]["ok"] is False


def test_a_store_that_ignores_the_conditional_write_is_caught():
    class Overwrites(FakeS3):
        def put_object(self, Bucket, Key, Body, ContentType, IfNoneMatch):
            self.calls.append(("put", Bucket, Key, ContentType, IfNoneMatch))
            self.objects[(Bucket, Key)] = Body

    client = Overwrites()
    client.objects[("test-bucket", "inputs/CALL_064.wav")] = b"x"
    results = check.run_checks(*[rig(client)[0], INPUT, rig(client)[1], "check.json"])
    assert results["second_write_refused"]["ok"] is False and "ACCEPTED" in results["second_write_refused"]["detail"]


def test_a_denied_write_is_a_recorded_failure_and_the_command_exits_one(monkeypatch, tmp_path):
    import json

    class NoWrite(FakeS3):
        def put_object(self, **kwargs):
            raise FakeClientError("AccessDenied")

    client = NoWrite()
    client.objects[("test-bucket", "inputs/CALL_064.wav")] = b"x"
    real = st.S3Storage
    monkeypatch.setattr(st, "S3Storage", lambda bucket, prefix="", **kwargs: real(bucket, prefix, client=client))
    out = tmp_path / "report.json"
    assert check.main(["--input-uri", INPUT, "--state-bucket", "test-bucket", "--state-prefix", "meralion-state/s3-check-2", "--out", str(out)]) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is False and report["results"]["conditional_write"]["ok"] is False and "decoder" in report["scope"]


def test_the_tool_never_builds_a_decoder_or_imports_the_heavy_stack():
    source = TOOL.read_text(encoding="utf-8")
    for forbidden in ("make_decoder", "VllmDecoder", "import torch", "import vllm", "delete_object", "transcripts/"):
        assert forbidden not in source.replace('writes nothing under transcripts/ and', ""), forbidden


def test_a_read_back_that_differs_from_what_was_written_is_caught():
    class Corrupts(FakeS3):
        def get_object(self, Bucket, Key):
            result = super().get_object(Bucket, Key)
            if Key.startswith("meralion-state/"):
                return {"Body": type("Body", (), {"read": lambda self_: b"something else"})()}
            return result

    client = Corrupts()
    client.objects[("test-bucket", "inputs/CALL_064.wav")] = b"x"
    results = check.run_checks(rig(client)[0], INPUT, rig(client)[1], "check.json")
    assert results["read_back_equals"]["ok"] is False and results["conditional_write"]["ok"] is True
