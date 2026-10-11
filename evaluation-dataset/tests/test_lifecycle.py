"""G3/G4 lifecycle: version transitions, historical reconstruction, stale and held responses, cross-stage write lock.

Every scenario runs on an isolated copy of the release with fake transports (no network, no credentials). The copy holds the
real accepted G3 configs; G3 is then reopened to a new accepted version while G4 plans exist, exactly as a G5-directed
regeneration would do.
"""

import json
from pathlib import Path

import pytest

from fyp_evaluation.call_configs.feasibility import synthesize_response as g3_synthesize
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.check_plans.feasibility import synthesize_response as g4_synthesize
from fyp_evaluation.check_plans.inputs import load_inputs as load_g4_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner, RunError
from fyp_evaluation.llm import stage_runner
from fyp_evaluation.llm.fake_transport import FakeTransport, response_body
from fyp_evaluation.llm.store import LockHeldError, StageWriteLockBusy, dataset_write_lock, write_lock_path
from g4_support import CALL_IDS, SCHEMAS, Clock, copy_release


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class World:
    """An isolated dataset copy with a G3 runner and a G4 runner sharing it (separate audit directories)."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.root = copy_release(tmp_path / "dataset")
        self.inputs = load_g4_inputs(self.root, SCHEMAS)
        self.fee_call = next(c for c in CALL_IDS if self.g3_context(c).fee_required)
        self.other_call = next(c for c in CALL_IDS if c != self.fee_call and self.g3_context(c).fee_required)
        self.fake3 = FakeTransport(self._g3_respond)
        self.fake4 = FakeTransport(self._g4_respond)
        self.g3 = ConfigRunner(self.root, audit_root=tmp_path / "audit_g3", schema_dir=SCHEMAS, transport=self.fake3,
                               clock=Clock(), inputs=self.inputs.upstream)
        self.g4 = self.new_g4(inputs=self.inputs)
        self.g3_rate = 0.75

    def g3_context(self, call_id):
        from fyp_evaluation.call_configs.slots import CallContext
        return CallContext(self.inputs.upstream, call_id)

    def _g3_respond(self, body):
        call_id = body["metadata"]["call_id"]
        response = g3_synthesize(self.g3.builder.context(call_id))
        response["fee_disclosure"] = {"basis": "RATE", "value": self.g3_rate, "currency": None}
        return response_body(json.dumps(response))

    def _g4_respond(self, body):
        call_id = body["metadata"]["call_id"]
        return response_body(json.dumps(g4_synthesize(self.g4.builder.context(call_id))))

    def new_g4(self, inputs=None, audit="audit_g4"):
        """A G4 runner; without `inputs` it performs the full upstream gate against the files on disk right now."""
        return CheckPlanRunner(self.root, audit_root=self.tmp / audit, schema_dir=SCHEMAS, transport=self.fake4, clock=Clock(),
                               inputs=inputs or load_g4_inputs(self.root, SCHEMAS))

    def config_text(self, call_id):
        return (self.root / "authoring/call_configs" / f"{call_id}.json").read_bytes().decode("utf-8").replace("\r\n", "\n")

    def plan_path(self, call_id):
        return self.root / "authoring/check_plans" / f"{call_id}.json"

    def plan_text(self, call_id):
        return self.plan_path(call_id).read_bytes().decode("utf-8").replace("\r\n", "\n")

    def g4_provenance(self, call_id):
        return parse_json_text((self.root / "authoring/check_plan_provenance" / f"{call_id}.json").read_text(encoding="utf-8"))

    def reopen_g3(self, call_id, rate=0.75):
        """Accept a materially different G3 config (new scenario content and a new fee rate) as the next G3 version."""
        self.g3_rate = rate
        attempt = self.g3.run_sync(call_id, reopen_reason="material change for a lifecycle test")
        assert attempt["state"] == "ACCEPTED", attempt
        return attempt


@pytest.fixture()
def world(tmp_path):
    return World(tmp_path)


def count_reloads(runner, monkeypatch):
    counter = {"n": 0}
    original = runner._load_inputs

    def counting():
        counter["n"] += 1
        return original()
    monkeypatch.setattr(runner, "_load_inputs", counting)
    return counter


# ============================================================================== A. version transition
def test_g3_v1_g4_v1_to_g3_v2_g4_v2_keeps_both_versions_verifiable(world):
    call = world.fee_call
    assert world.g4.run_sync(call)["state"] == "ACCEPTED"
    v1_text, g3_v1 = world.plan_text(call), world.config_text(call)
    v1_hash, g3_v1_hash = text_hash(v1_text), text_hash(g3_v1)
    assert parse_json_text(v1_text)["call_config_sha256"] == g3_v1_hash
    world.reopen_g3(call)
    g3_v2_hash = text_hash(world.config_text(call))
    assert g3_v2_hash != g3_v1_hash

    fresh = world.new_g4()  # a new process after the G3 change
    assert fresh.dependency_status([call, world.other_call]) == {
        call: {"status": "stale", "changed": ["call_config"], "error": None},
        world.other_call: {"status": "none", "changed": [], "error": None}}
    assert fresh.status()["calls"][call]["status"] == "ACCEPTED_STALE"
    # historical reproduction still succeeds; current dependency validity fails until G4 is regenerated
    errors = fresh.validate()
    assert any("stale call_config_sha256" in e for e in errors)
    assert not any("cannot be reproduced" in e or "cannot be rebuilt" in e for e in errors)
    difference = fresh.check()
    assert "authoring/check_plans/" + call + ".json" not in difference["changed"] + difference["extra"]
    assert difference["changed"] == ["authoring/check_plan_record.json"]
    assert world.plan_text(call) == v1_text  # nothing was rewritten

    v2 = fresh.run_sync(call, reopen_reason="G3 changed")
    assert v2["state"] == "ACCEPTED" and v2["attempt_id"] == "A002"
    v2_text = world.plan_text(call)
    assert parse_json_text(v2_text)["call_config_sha256"] == g3_v2_hash and "0.75%" in v2_text and v2_text != v1_text
    assert fresh.validate() == [] and fresh.check() == {"missing": [], "changed": [], "extra": []}
    assert fresh.dependency_status([call])[call]["status"] == "current"
    provenance = world.g4_provenance(call)
    (history,) = provenance["accepted_history"]
    assert history["artifact_sha256"] == v1_hash and provenance["accepted"]["artifact_sha256"] == text_hash(v2_text)
    # the historical plan is rebuilt against G3 v1 (not the current G3 v2) and reproduces its recorded hash
    rebuilt = fresh._rebuild_text(call, history)
    assert text_hash(rebuilt) == v1_hash and rebuilt == v1_text
    assert world.g4_provenance(call)["attempts"][0]["upstream_sha256"]["call_config"] == g3_v1_hash
    assert provenance["attempts"][1]["upstream_sha256"]["call_config"] == g3_v2_hash


def test_historical_response_rules_are_checked_against_the_matching_version(world):
    call = world.fee_call
    world.g4.run_sync(call)
    world.reopen_g3(call)
    fresh = world.new_g4()
    fresh.run_sync(call, reopen_reason="again")
    assert fresh.validate() == []
    # break the v1 history entry's response so that it violates a rule of ITS OWN (G3 v1) context
    path = world.root / "authoring/check_plan_provenance" / f"{call}.json"
    provenance = parse_json_text(path.read_text(encoding="utf-8"))
    entry = provenance["accepted_history"][0]
    entry["parsed_response"]["events"][0]["text"] = "We have 500 units."
    from fyp_evaluation.llm.canonical import response_hash
    entry["parsed_response_sha256"] = response_hash(entry["parsed_response"], fresh.builder.response_schema)
    path.write_text(serialize_json(provenance), encoding="utf-8")
    errors = fresh.validate()
    assert any("history[0]" in e and "digits" in e for e in errors)  # reported explicitly, whichever step trips first


def test_the_sync_path_rebuilds_requests_from_fresh_upstream_after_a_g3_change(world, monkeypatch):
    call = world.fee_call
    world.g4.run_sync(call)  # the G4 runner now caches G3 v1
    world.reopen_g3(call)
    reloads = count_reloads(world.g4, monkeypatch)
    attempt = world.g4.run_sync(call, reopen_reason="after G3")
    assert reloads["n"] == 1 and attempt["state"] == "ACCEPTED"
    assert attempt["upstream_sha256"]["call_config"] == text_hash(world.config_text(call))


def test_a_request_after_unreadable_upstream_sends_nothing(world):
    call = world.fee_call
    (world.root / "authoring/call_configs" / f"{call}.json").unlink()
    with pytest.raises(RunError, match="no request was sent"):
        world.g4.run_sync(call)
    with pytest.raises(RunError, match="no request was sent"):
        world.g4.submit([world.other_call])  # a batch checks every call's upstream
    assert world.fake4.calls == [] and world.g4.store.attempt_ids(call) == [] and world.g4.store.submission_ids() == []


# ============================================================================= B. stale / held responses
def test_pending_batch_response_built_on_g3_v1_is_not_accepted_after_g3_v2(world):
    call, other = world.fee_call, world.other_call
    report = world.g4.submit([call, other])
    world.reopen_g3(call)
    world.fake4.finish_batch(report["batch_id"])
    result = world.g4.collect()  # the SAME runner instance that still caches G3 v1
    assert result["collected"] == ["SUB_0001"] and not result["fatal"]
    stale = world.g4.store.read_attempt(call, "A001")
    assert stale["state"] == "STALE_UPSTREAM" and stale["stale_upstream"] == {"changed": ["call_config"]}
    assert "call_config" in stale["rejection_reason"]
    assert not world.plan_path(call).exists() and world.g4.store.read_raw_response(call, "A001")["response"]["status"] == "completed"
    assert world.g4.store.read_attempt(other, "A001")["state"] == "ACCEPTED" and world.plan_path(other).is_file()  # unrelated result still processed
    summary = world.g4_provenance(call)["attempts"][0]
    assert summary["state"] == "STALE_UPSTREAM" and summary["stale_upstream"] == {"changed": ["call_config"]}
    state = world.g4.call_state(call)
    assert state["rejected"] == 0 and state["transport_failed"] == 0 and not state["in_flight"]
    downloads = len([c for c in world.fake4.calls if c[0] == "download_file"])
    assert world.g4.collect()["collected"] == [] and len([c for c in world.fake4.calls if c[0] == "download_file"]) == downloads
    assert len([c for c in world.fake4.calls if c[0] == "create_batch"]) == 1  # never resubmitted automatically
    assert world.g4.validate() == [] and world.g4.status()["calls"][call]["status"] == "STALE_UPSTREAM"

    # an explicit new attempt works, is budget-neutral and is built on G3 v2
    ok, reason = world.g4.eligibility(call)
    assert ok, reason
    again = world.g4.submit([call])
    assert again["submission_id"] == "SUB_0002"
    world.fake4.finish_batch(again["batch_id"])
    world.g4.collect()
    new = world.g4.store.read_attempt(call, "A002")
    assert new["state"] == "ACCEPTED" and new["upstream_sha256"]["call_config"] == text_hash(world.config_text(call))
    assert parse_json_text(world.plan_text(call))["call_config_sha256"] == text_hash(world.config_text(call))
    assert world.g4.validate() == [] and world.g4.check() == {"missing": [], "changed": [], "extra": []}


def test_a_stale_reopen_response_keeps_the_older_accepted_version_and_needs_an_explicit_reopen(world):
    call = world.fee_call
    world.g4.run_sync(call)
    v1_text = world.plan_text(call)
    report = world.g4.submit([call], reopen_reason="rework")
    world.reopen_g3(call)
    world.fake4.finish_batch(report["batch_id"])
    world.g4.collect()
    assert world.g4.store.read_attempt(call, "A002")["state"] == "STALE_UPSTREAM" and world.plan_text(call) == v1_text
    ok, reason = world.g4.eligibility(call)
    assert not ok and "already accepted" in reason  # an explicit --reopen is required for an accepted call
    assert world.g4.eligibility(call, reopen=True)[0]
    assert world.g4.run_sync(call, reopen_reason="after G3 v2")["state"] == "ACCEPTED"
    assert world.g4.validate() == [] and len(world.g4_provenance(call)["accepted_history"]) == 1


def test_a_response_stored_before_a_change_is_stale_for_a_long_running_runner(world):
    call = world.fee_call
    attempt, _ = world.g4._new_attempt(call, "sync")
    world.g4._store_response(attempt, world._g4_respond({"metadata": {"call_id": call}}), {"transport": "sync"})
    world.reopen_g3(call)
    (done,) = world.g4._finalize_responded()
    assert done["state"] == "STALE_UPSTREAM" and not world.plan_path(call).exists()


def test_same_hashes_accept_normally_without_reloading(world, monkeypatch):
    reloads = count_reloads(world.g4, monkeypatch)
    for call in (world.fee_call, world.other_call):
        assert world.g4.run_sync(call)["state"] == "ACCEPTED"
    assert reloads["n"] == 0 and world.g4.validate() == []


def test_inputs_cached_by_a_lagging_runner_are_reloaded_not_reused(world, monkeypatch):
    call = world.fee_call
    lagging = world.g4  # caches G3 v1
    world.reopen_g3(call)
    current = world.new_g4()  # built after the change: records G3 v2 in its request
    attempt, _ = current._new_attempt(call, "sync", reopen_reason=None)
    current._store_response(attempt, world._g4_respond({"metadata": {"call_id": call}}), {"transport": "sync"})
    reloads = count_reloads(lagging, monkeypatch)
    (done,) = lagging._finalize_responded()
    assert reloads["n"] == 1 and done["state"] == "ACCEPTED"
    assert parse_json_text(world.plan_text(call))["call_config_sha256"] == text_hash(world.config_text(call))
    assert lagging.dependency_status([call])[call]["status"] == "current"


def test_unreadable_upstream_blocks_acceptance_keeps_the_response_and_resumes_after_repair(world):
    call = world.fee_call
    config = world.root / "authoring/call_configs" / f"{call}.json"
    saved = config.read_bytes()
    attempt, _ = world.g4._new_attempt(call, "sync")
    world.g4._store_response(attempt, world._g4_respond({"metadata": {"call_id": call}}), {"transport": "sync"})
    config.unlink()
    (held,) = world.g4._finalize_responded()
    assert held["state"] == "RESPONDED" and held["blocked"]["kind"] == "UPSTREAM_INVALID" and "cannot read the upstream inputs" in held["blocked"]["reason"]
    assert world.g4.store.read_raw_response(call, "A001") is not None and not world.plan_path(call).exists()
    assert world.g4.eligibility(call)[0] is False  # still in flight: nothing can be resubmitted over it
    config.write_bytes(saved)
    (done,) = world.g4._finalize_responded()
    assert done["state"] == "ACCEPTED" and "blocked" not in done and world.g4.validate() == []


def test_upstream_that_fails_validation_blocks_with_an_explicit_status(world):
    call = world.fee_call
    lagging = world.g4
    world.reopen_g3(call)
    current = world.new_g4()
    attempt, _ = current._new_attempt(call, "sync")
    current._store_response(attempt, world._g4_respond({"metadata": {"call_id": call}}), {"transport": "sync"})
    record = world.root / "authoring/call_config_record.json"  # not one of the hashed upstream files, but part of the G3 gate
    saved = record.read_bytes()
    record.unlink()
    (held,) = lagging._finalize_responded()
    assert held["state"] == "RESPONDED" and held["blocked"]["kind"] == "UPSTREAM_INVALID" and "failed validation" in held["blocked"]["reason"]
    assert lagging.status()["calls"][call]["status"] == "BLOCKED_UPSTREAM_INVALID"
    record.write_bytes(saved)
    (done,) = lagging._finalize_responded()
    assert done["state"] == "ACCEPTED"


# ================================================================================ C. cross-stage write lock
def test_lock_contention_defers_the_response_and_a_later_collect_completes_it(world):
    call = world.fee_call
    world.g4.write_lock_wait = 0.05
    report = world.g4.submit([call, world.other_call])
    world.fake4.finish_batch(report["batch_id"])
    with dataset_write_lock(world.root):
        result = world.g4.collect()
        assert result["collected"] == ["SUB_0001"] and any("still unresolved" in a for a in result["anomalies"])
        assert any("WRITE_LOCK_BUSY" in a for a in result["anomalies"])
        assert world.g4.store.read_submission("SUB_0001")["state"] == "SUBMITTED"
        assert world.g4.store.read_attempt(call, "A001")["state"] == "RESPONDED"
        assert world.g4.status()["calls"][call]["status"] == "DEFERRED_WRITE_LOCK"
        assert not world.plan_path(call).exists()
    result = world.g4.collect()
    assert result["anomalies"] == [] and world.g4.store.read_submission("SUB_0001")["state"] == "COLLECTED"
    assert world.plan_path(call).is_file() and world.plan_path(world.other_call).is_file()
    downloads = [c for c in world.fake4.calls if c[0] == "download_file"]
    assert len(downloads) == 2  # the first collect downloaded; the second only re-checked the batch


def test_the_write_lock_is_keyed_by_the_resolved_dataset_root_not_the_audit_directory(world, tmp_path):
    assert write_lock_path(world.root) == write_lock_path(world.root / ".." / "dataset") == world.root.resolve() / ".stage_write.lock"
    assert write_lock_path(world.root) == write_lock_path(str(world.root))
    other_audit = world.new_g4(inputs=world.inputs, audit="somewhere_else")
    other_audit.write_lock_wait = 0.05
    call = world.fee_call
    attempt, _ = other_audit._new_attempt(call, "sync")
    other_audit._store_response(attempt, world._g4_respond({"metadata": {"call_id": call}}), {"transport": "sync"})
    with dataset_write_lock(world.root):  # e.g. a G3 acceptance running with yet another --audit-dir
        (held,) = other_audit._finalize_responded()
        assert held["blocked"]["kind"] == "WRITE_LOCK_BUSY"
    (done,) = other_audit._finalize_responded()
    assert done["state"] == "ACCEPTED"


def test_g3_acceptance_uses_the_same_lock_with_a_different_audit_directory(world):
    world.g3.write_lock_wait = 0.05
    call = world.fee_call
    with dataset_write_lock(world.root):
        attempt = world.g3.run_sync(call, reopen_reason="x")
        assert attempt["state"] == "RESPONDED" and attempt["blocked"]["kind"] == "WRITE_LOCK_BUSY"
    (done,) = world.g3._finalize_responded()
    assert done["state"] == "ACCEPTED"


def test_independent_dataset_roots_do_not_block_each_other(tmp_path):
    from g4_support import make_runner
    first, _ = make_runner(tmp_path / "a")
    second, _ = make_runner(tmp_path / "b")
    first.write_lock_wait = second.write_lock_wait = 0.05
    with dataset_write_lock(first.root):
        assert second.run_sync("CALL_001")["state"] == "ACCEPTED"
        assert first.run_sync("CALL_002")["state"] == "RESPONDED"  # same root as the held lock: deferred
    assert first._finalize_responded()[0]["state"] == "ACCEPTED"


def test_a_leftover_lock_fails_explicitly_and_is_never_removed_silently(world):
    leftover = write_lock_path(world.root)
    leftover.write_text("pid 1\n")
    with pytest.raises(StageWriteLockBusy, match="Confirm that no G3/G4/G5 process is active") as raised:
        with dataset_write_lock(world.root, wait_seconds=0.05):
            pass
    assert str(leftover) in str(raised.value) and isinstance(raised.value, LockHeldError)
    assert leftover.exists()
    leftover.unlink()
    with dataset_write_lock(world.root, wait_seconds=0.05):
        assert leftover.exists()
    assert not leftover.exists()


def test_no_lock_is_held_across_provider_calls_or_batch_processing(world):
    lock = write_lock_path(world.root)
    seen = []

    def watch(fake, name):
        original = getattr(fake, name)

        def wrapped(*args, **kwargs):
            seen.append((name, lock.exists()))
            return original(*args, **kwargs)
        setattr(fake, name, wrapped)
    for name in ("create_response", "upload_batch_file", "create_batch", "retrieve_batch", "download_file"):
        watch(world.fake4, name)
    world.g4.run_sync(world.fee_call)
    report = world.g4.submit([world.other_call])
    world.fake4.finish_batch(report["batch_id"])
    world.g4.collect()
    assert {name for name, _ in seen} == {"create_response", "upload_batch_file", "create_batch", "retrieve_batch", "download_file"}
    assert not any(held for _, held in seen) and not lock.exists()


def test_an_interrupted_write_releases_the_lock_and_is_repaired_idempotently(world, monkeypatch):
    call = world.fee_call
    attempt, _ = world.g4._new_attempt(call, "sync")
    world.g4._store_response(attempt, world._g4_respond({"metadata": {"call_id": call}}), {"transport": "sync"})
    real = stage_runner.write_text_atomic

    def crash_on_artifact(path, text):
        if str(path).endswith(f"check_plans{'/' if '/' in str(path) else chr(92)}{call}.json"):
            raise OSError("simulated crash while writing the artifact")
        return real(path, text)
    monkeypatch.setattr(stage_runner, "write_text_atomic", crash_on_artifact)
    with pytest.raises(OSError, match="simulated crash"):
        world.g4._finalize_responded()
    assert not write_lock_path(world.root).exists()
    assert world.g4.store.read_attempt(call, "A001")["state"] == "RESPONDED"
    assert world.g4_provenance(call)["accepted"]["attempt_id"] == "A001" and not world.plan_path(call).exists()  # half-written
    monkeypatch.setattr(stage_runner, "write_text_atomic", real)
    (done,) = world.g4._finalize_responded()
    assert done["state"] == "ACCEPTED" and world.g4.validate() == [] and world.g4_provenance(call)["accepted_history"] == []


# ======================================================================== D. historical context failures
def accepted_then_g3_changed(world):
    call = world.fee_call
    world.g4.run_sync(call)
    world.reopen_g3(call)
    return call, world.new_g4()


def test_a_missing_committed_attempt_summary_fails_explicitly_and_never_falls_back(world):
    call, fresh = accepted_then_g3_changed(world)
    path = world.root / "authoring/check_plan_provenance" / f"{call}.json"
    provenance = parse_json_text(path.read_text(encoding="utf-8"))
    provenance["attempts"] = []
    path.write_text(serialize_json(provenance), encoding="utf-8")
    errors = fresh.validate()
    assert any("no committed attempt summary records the G3 config" in e for e in errors)
    difference = fresh.check()
    assert any(call in item for item in difference["unreproducible"]) and not difference["changed"][:-1]
    with pytest.raises(ValueError, match="no committed attempt summary"):
        fresh._rebuild_text(call, provenance["accepted"])


def test_an_artifact_disagreeing_with_its_attempt_summary_is_rejected(world):
    call, fresh = accepted_then_g3_changed(world)
    path = world.root / "authoring/check_plan_provenance" / f"{call}.json"
    provenance = parse_json_text(path.read_text(encoding="utf-8"))
    provenance["attempts"][0]["upstream_sha256"]["call_config"] = "0" * 64
    path.write_text(serialize_json(provenance), encoding="utf-8")
    assert any("disagree about the G3 config version" in e for e in fresh.validate())


def test_corrupted_or_missing_g3_history_fails_explicitly(world):
    call, fresh = accepted_then_g3_changed(world)
    g3_path = world.root / "authoring/call_config_provenance" / f"{call}.json"
    original = g3_path.read_bytes()
    provenance = parse_json_text(original.decode("utf-8"))
    from fyp_evaluation.call_configs.request import RequestBuilder as G3RequestBuilder
    from fyp_evaluation.llm.canonical import response_hash
    broken = json.loads(json.dumps(provenance))
    entry = broken["accepted_history"][0]
    entry["parsed_response"]["client"]["name"] = "Someone Else"  # consistent response hash, but no longer the accepted artifact
    entry["parsed_response_sha256"] = response_hash(entry["parsed_response"], G3RequestBuilder(fresh.inputs.upstream, SCHEMAS).response_schema)
    g3_path.write_text(serialize_json(broken), encoding="utf-8")
    errors = fresh.validate()
    assert any("cannot be reconstructed from G3's accepted history" in e and "corrupted history" in e for e in errors)
    assert not any("differs from the rebuild: authoring/check_plans" in e for e in errors)
    missing = json.loads(json.dumps(provenance))
    missing["accepted_history"] = []
    g3_path.write_text(serialize_json(missing), encoding="utf-8")
    assert any("cannot be reconstructed from G3's accepted history" in e for e in fresh.validate())
    g3_path.write_bytes(original)
    assert fresh.dependency_status([call])[call]["status"] == "stale"


# ================================================================ E. dependency status is never assumed
def test_g3_dependency_status_is_computed_from_the_files_not_assumed(world):
    call = world.fee_call
    assert world.g3.dependency_status([call, "CALL_001"]) == {
        call: {"status": "current", "changed": [], "error": None}, "CALL_001": {"status": "current", "changed": [], "error": None}}
    checklist = world.root / "checklist.json"
    saved = checklist.read_bytes()
    checklist.write_bytes(saved + b" ")  # a frozen input silently altered
    status = world.g3.dependency_status([call])[call]
    assert status["status"] == "stale" and status["changed"] == ["checklist_sha256"]
    assert world.g3.status()["calls"][call]["status"] == "ACCEPTED_STALE"
    checklist.write_bytes(saved)
    plan = world.root / "authoring/generation_plan.json"
    plan_saved = plan.read_bytes()
    plan.unlink()
    broken = world.g3.dependency_status([call])[call]
    assert broken["status"] == "upstream_invalid" and "generation_plan" in broken["error"]
    assert world.g3.status()["calls"][call]["status"] == "ACCEPTED_UPSTREAM_INVALID"
    plan.write_bytes(plan_saved)
    assert world.g3.dependency_status([call])[call]["status"] == "current"


def test_the_cli_status_lists_the_dependency_state(world, capsys):
    from fyp_evaluation.check_plans.cli import main
    world.g4.run_sync(world.fee_call)
    world.reopen_g3(world.fee_call)
    args = ["--dataset-root", str(world.root), "--audit-dir", str(world.tmp / "audit_g4"), "--schema-dir", str(SCHEMAS), "--status"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert f"{world.fee_call} ACCEPTED_STALE" in out and "dependency=stale" in out and "counts: " in out and "ACCEPTED_STALE" in out.splitlines()[-1]
