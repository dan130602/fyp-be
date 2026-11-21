"""Targeted sampling pool: the corrected default draws extras only from clips that contain entity spans (never from compliance-evidence-only clips); the original rule stays selectable."""
import hashlib
import json

import pytest

import sft_core as C
import sft_data as D
import sft_train as T
import test_sft as H
import test_sft_run as R


@pytest.fixture(autouse=True)
def small_speech_token(monkeypatch):
    monkeypatch.setattr(C, "SPEECH_TOKEN_ID", H.SPEECH)


def package_with_evidence_only_clips(tmp_path):
    """The standard test package, then 3 of the 6 odd (entity-free) training clips become compliance-evidence-only clips (important = True, no entity span)."""
    data, batch, n, m = R.make_package(tmp_path)
    clips = [json.loads(l) for l in (data / "clips_train.jsonl").read_text().splitlines()]
    odd = [c for c in clips if not c["entity_spans"]]
    for c in odd[:3]:
        c["evidence_turns"], c["important"] = [1], True
    (data / "clips_train.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["files"]["clips_train.jsonl"] = hashlib.sha256((data / "clips_train.jsonl").read_bytes()).hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data, batch, clips


def shares(tmp_path, run):
    return [json.loads(l)["important_draw_share"] for l in (tmp_path / "runs" / run / "sampling_summary.jsonl").read_text().splitlines()]


def test_entity_pool_excludes_evidence_only_clips_and_extends_past_the_pool(tmp_path):
    data, batch, clips = package_with_evidence_only_clips(tmp_path)
    n = len(clips)
    entity = sum(1 for c in clips if c["entity_spans"])
    important = sum(1 for c in clips if c["important"])
    assert (n, entity, important) == (15, 9, 12)
    k = 12                                                              # more extras than entity clips: a second round is needed
    T.run(R.args_for(tmp_path, data, batch, "ENT", k=k, objective="ce", sampling="targeted", extra=("--skip-baseline-validation",)), R.stack())
    T.run(R.args_for(tmp_path, data, batch, "OLD", k=k, objective="ce", sampling="targeted", extra=("--skip-baseline-validation", "--target-pool", "important")), R.stack())
    assert shares(tmp_path, "ENT") == [round((entity + k) / (n + k), 4)] * 2        # every extra copy is an entity clip: 9 once + 3 twice-extra
    assert shares(tmp_path, "OLD") == [round((important + k) / (n + k), 4)] * 2     # the original rule also doubles the 3 evidence-only clips
    cfg = json.loads((tmp_path / "runs" / "ENT" / "config.json").read_text())["config"]
    assert cfg["target_pool"] == "entity" and json.loads((tmp_path / "runs" / "OLD" / "config.json").read_text())["config"]["target_pool"] == "important"


def test_entity_pool_plan_never_gives_an_evidence_only_clip_an_extra_copy(tmp_path):
    data, batch, clips = package_with_evidence_only_clips(tmp_path)
    ids = [c["clip_id"] for c in clips]
    entity_ids = [c["clip_id"] for c in clips if c["entity_spans"]]
    evidence_only = {c["clip_id"] for c in clips if c["important"] and not c["entity_spans"]}
    from collections import Counter
    for plan in D.plan_epochs(ids, entity_ids, 12, 3, 42, "targeted"):
        counts = Counter(plan)
        assert all(counts[c] == 1 for c in evidence_only) and all(counts[c] >= 2 for c in entity_ids)
