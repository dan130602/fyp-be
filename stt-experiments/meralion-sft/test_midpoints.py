"""Mid-epoch (half-epoch) validation: it never changes training, writes its own log and adapter checkpoints, and survives an interrupted/resumed run exactly once per point."""
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


def rows(tmp_path, run, name):
    path = tmp_path / "runs" / run / name
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def expected_updates(n, m, accumulation=4, epochs=2, per_epoch=2):
    steps = len(D.group_boundaries(n + m, accumulation))
    mid = round(steps * 1 / per_epoch)
    return steps, [e * steps + mid for e in range(epochs)]


def test_midpoint_validation_does_not_change_training_and_writes_its_own_files(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    T.run(R.args_for(tmp_path, data, batch, "plain", k=m, extra=("--validate-per-epoch", "1", "--skip-baseline-validation")), R.stack())
    T.run(R.args_for(tmp_path, data, batch, "half", k=m, extra=("--skip-baseline-validation",)), R.stack())            # default: every half epoch
    for epoch in (1, 2):
        assert R.digest_of(tmp_path, "plain", epoch) == R.digest_of(tmp_path, "half", epoch)
    steps, updates = expected_updates(n, m)
    mids = rows(tmp_path, "half", "val_midpoints.jsonl")
    assert [r["update"] for r in mids] == updates and [r["epoch"] for r in mids] == [1, 2] and all(r["epoch_fraction"] == round((u % steps or steps) / steps, 4) for r, u in zip(mids, updates))
    assert rows(tmp_path, "plain", "val_midpoints.jsonl") == []
    # the epoch-level logs are untouched by the midpoints (selection reads epoch_summary by epoch number)
    assert [e["epoch"] for e in rows(tmp_path, "half", "epoch_summary.jsonl")] == [1, 2] and [e["epoch"] for e in rows(tmp_path, "half", "val_metrics.jsonl")] == [1, 2]
    for r in mids:                                                                                                       # the saved adapter is exactly the validated one
        state, meta = C.load_lora_safetensors(tmp_path / "runs" / "half" / "checkpoints" / r["checkpoint"])
        assert C.state_digest(state) == r["checkpoint_digest"] and int(meta["update"]) == r["update"] and r["validation"]["ce"] is not None
    # same training log for both runs (validation consumed no random numbers and changed no weights)
    plain, half = rows(tmp_path, "plain", "train_metrics.jsonl"), rows(tmp_path, "half", "train_metrics.jsonl")
    assert [r["ce"] for r in plain] == [r["ce"] for r in half] and [r["grad_norm"] for r in plain] == [r["grad_norm"] for r in half]


@pytest.mark.parametrize("interrupt_call", [2, 6])                                                                       # before / after the first midpoint (it falls at update 3)
def test_interrupted_run_keeps_each_midpoint_exactly_once(tmp_path, monkeypatch, interrupt_call):
    data, batch, n, m = R.make_package(tmp_path)
    T.run(R.args_for(tmp_path, data, batch, "A", k=m, extra=("--skip-baseline-validation",)), R.stack())
    calls, original = {"n": 0}, T.train_group

    def interrupting(*a, **k):
        calls["n"] += 1
        if calls["n"] == interrupt_call:
            raise KeyboardInterrupt
        return original(*a, **k)
    monkeypatch.setattr(T, "train_group", interrupting)
    with pytest.raises(KeyboardInterrupt):
        T.run(R.args_for(tmp_path, data, batch, "B", k=m, extra=("--skip-baseline-validation",)), R.stack())
    monkeypatch.setattr(T, "train_group", original)
    T.run(R.args_for(tmp_path, data, batch, "B", k=m, extra=("--skip-baseline-validation", "--resume")), R.stack())
    steps, updates = expected_updates(n, m)
    assert [r["update"] for r in rows(tmp_path, "B", "val_midpoints.jsonl")] == updates
    assert R.digest_of(tmp_path, "A", 2) == R.digest_of(tmp_path, "B", 2)
    assert [r["validation"]["ce"] for r in rows(tmp_path, "A", "val_midpoints.jsonl")] == [r["validation"]["ce"] for r in rows(tmp_path, "B", "val_midpoints.jsonl")]


def test_trim_logs_drops_only_midpoints_after_the_resumed_state(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoints").mkdir()
    (run / "val_midpoints.jsonl").write_text("".join(json.dumps({"update": u}) + "\n" for u in (3, 9, 12)))
    removed = T.trim_logs(run, {"epoch": 2, "group": 2, "update": 9})
    assert [json.loads(l)["update"] for l in (run / "val_midpoints.jsonl").read_text().splitlines()] == [3, 9] and removed == {"val_midpoints.jsonl": 1}


def test_four_validations_per_epoch(tmp_path):
    data, batch, n, m = R.make_package(tmp_path)
    T.run(R.args_for(tmp_path, data, batch, "Q", k=m, extra=("--skip-baseline-validation", "--validate-per-epoch", "4")), R.stack())
    steps = len(D.group_boundaries(n + m, 4))
    expected = sorted({round(steps * k / 4) for k in (1, 2, 3)} - {0, steps})
    got = [r["step_in_epoch"] for r in rows(tmp_path, "Q", "val_midpoints.jsonl") if r["epoch"] == 1]
    assert got == expected
