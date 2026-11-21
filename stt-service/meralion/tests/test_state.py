import json

import pytest

from meralion_worker import state as st
from meralion_worker.storage import LocalStorage

RECORD = {"invocation_id": "inv-1", "jobs": [{"job_name": "a", "input": "/x/a.wav"}, {"job_name": "b", "input": "/x/b.wav"}], "config": {"strategy": {"id": "s"}, "formatter": "none"},
          "config_fingerprint": "f" * 64, "model_manifest_sha256": "m" * 64, "vendored_sources_sha256": "v" * 64,
          "decoder": {"backend": "replay", "settings_id": "replay", "frozen_digest": "replay"}, "worker_version": "0.1.0", "created_at": "2026-11-15T00:00:00+00:00"}


def make(tmp_path, invocation_id="inv-1"):
    return st.InvocationState(LocalStorage(tmp_path), invocation_id)


def test_keys_live_under_the_invocation_id(tmp_path):
    state = make(tmp_path)
    assert state.key("first_pass", "job.json") == "inv-1/first_pass/job.json"
    state.put_once("recordings", "job.json", document={"x": 1})
    assert (tmp_path / "inv-1" / "recordings" / "job.json").is_file()


@pytest.mark.parametrize("bad", ["", "has space", "a/b", "über", "x" * 129, "id\n", "\nid", "id\r\n"])
def test_invalid_invocation_ids_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="invocation id"):
        make(tmp_path, bad)


@pytest.mark.parametrize("good", ["inv-1", "A.b_c-9", "x" * 128])
def test_valid_invocation_ids(tmp_path, good):
    assert make(tmp_path, good).invocation_id == good


def test_put_once_writes_the_schema_and_never_overwrites(tmp_path):
    state = make(tmp_path)
    created, first_hash = state.put_once("plan.json", document={"members": ["a"]})
    assert created and state.read("plan.json") == {"schema": st.SCHEMA, "members": ["a"]}
    created, again_hash = state.put_once("plan.json", document={"members": ["DIFFERENT"]})
    assert not created and again_hash == first_hash                                   # the stored file's hash is returned, the stored file is untouched
    assert state.read("plan.json")["members"] == ["a"]


def test_the_hash_is_of_the_bytes_on_disk(tmp_path):
    state = make(tmp_path)
    _, digest = state.put_once("plan.json", document={"a": 1})
    assert digest == st.sha256_bytes(state.read_bytes("plan.json"))


def test_the_canonical_bytes_do_not_depend_on_key_order_and_keep_unicode():
    assert st.canonical_bytes({"b": 1, "a": [2, 3]}) == st.canonical_bytes({"a": [2, 3], "b": 1})
    assert "好的".encode("utf-8") in st.canonical_bytes({"text": "好的"})
    assert st.canonical_bytes({"a": 1}).endswith(b"\n")


def test_stable_view_drops_volatile_fields_at_every_depth_and_keeps_the_rest():
    document = {"timing": {"total_s": 1}, "created_at": "x", "windows": [{"text": "a", "decode_seconds": 0.5, "native": {"batch_seconds": 2.0, "finish_reason": "stop"}}], "model_load_seconds": 3, "kept": 0}
    assert st.stable_view(document) == {"windows": [{"text": "a", "native": {"finish_reason": "stop"}}], "kept": 0}
    assert document["timing"] == {"total_s": 1}                                        # the input is not modified


def test_the_first_begin_creates_and_a_second_with_the_same_identity_resumes(tmp_path):
    state = make(tmp_path)
    assert state.begin(RECORD) is False
    assert state.begin({**RECORD, "created_at": "a different time"}) is True           # created_at is not part of the identity
    assert state.read("invocation.json")["created_at"] == RECORD["created_at"]         # and the stored file is not overwritten


@pytest.mark.parametrize("field, value", [
    ("invocation_id", "inv-2"),
    ("jobs", [{"job_name": "b", "input": "/x/b.wav"}, {"job_name": "a", "input": "/x/a.wav"}]),      # the order matters
    ("jobs", [{"job_name": "a", "input": "/x/a.wav"}]),
    ("jobs", [{"job_name": "a", "input": "/other/a.wav"}, {"job_name": "b", "input": "/x/b.wav"}]),
    ("config", {"strategy": {"id": "s"}, "formatter": "percentages-v2"}),
    ("config_fingerprint", "0" * 64),
    ("model_manifest_sha256", "0" * 64),
    ("vendored_sources_sha256", "0" * 64),
    ("decoder", {"backend": "vllm", "settings_id": "stock-vllm-package-v1", "frozen_digest": "x"}),
    ("decoder", {"backend": "replay", "settings_id": "replay", "frozen_digest": "other"}),
    ("worker_version", "0.2.0"),
])
def test_each_identity_field_makes_a_restart_incompatible(tmp_path, field, value):
    state = make(tmp_path)
    state.begin(RECORD)
    before = state.read_bytes("invocation.json")
    with pytest.raises(st.ResumeError, match=field):
        state.begin({**RECORD, field: value})
    assert state.read_bytes("invocation.json") == before                                # nothing was overwritten


def test_every_listed_identity_field_is_tested():
    assert set(st.IDENTITY_FIELDS) == {"invocation_id", "jobs", "config", "config_fingerprint", "model_manifest_sha256", "vendored_sources_sha256", "decoder", "worker_version"}


def test_state_files_survive_a_new_state_object_over_the_same_storage(tmp_path):
    make(tmp_path).put_once("first_pass", "a.json", document={"windows": []})
    assert make(tmp_path).exists("first_pass", "a.json") and not make(tmp_path).exists("first_pass", "b.json")
    assert json.loads((tmp_path / "inv-1" / "first_pass" / "a.json").read_text(encoding="utf-8"))["windows"] == []
