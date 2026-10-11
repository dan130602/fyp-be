"""Atomic writes survive transient Windows sharing violations but never swallow a permanent failure."""

import os

import pytest

from fyp_evaluation.llm import fsutil


def test_a_transient_permission_error_is_retried_and_the_write_completes(tmp_path, monkeypatch):
    real, calls = os.replace, {"n": 0}

    def flaky(source, target):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "Access is denied")
        return real(source, target)
    monkeypatch.setattr(fsutil.os, "replace", flaky)
    monkeypatch.setattr(fsutil.time, "sleep", lambda _: None)
    target = tmp_path / "out.json"
    fsutil.write_text_atomic(target, "hello\n")
    assert target.read_bytes() == b"hello\n" and calls["n"] == 4 and not (tmp_path / "out.json.tmp").exists()


def test_a_permanent_permission_error_is_raised_after_bounded_retries_and_leaves_the_target_intact(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    target.write_bytes(b"old\n")
    calls = {"n": 0}

    def denied(source, target):
        calls["n"] += 1
        raise PermissionError(5, "Access is denied")
    monkeypatch.setattr(fsutil.os, "replace", denied)
    monkeypatch.setattr(fsutil.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        fsutil.write_text_atomic(target, "new\n")
    assert calls["n"] == fsutil.REPLACE_ATTEMPTS and target.read_bytes() == b"old\n"
