"""The runtime (package versions and tracked patches) is part of the invocation identity, so a different runtime cannot resume a state (Stage 3 follow-up)."""
import hashlib
import json

import pytest

import synthetic
from harness import run
from meralion_worker import runtime_identity as ri
from meralion_worker.decoders.base import Decoder
from meralion_worker.state import ResumeError


def test_package_versions_are_reported_and_a_missing_package_is_none():
    versions = ri.package_versions(("pytest", "definitely-not-installed-package"))
    assert versions["pytest"] and versions["definitely-not-installed-package"] is None


def test_without_a_manifest_there_are_no_patches(tmp_path):
    assert ri.patches_identity(tmp_path / "missing.json") == [] and ri.runtime_identity(packages=("pytest",), manifest_path=tmp_path / "missing.json")["patches"] == []


def make_package(tmp_path, monkeypatch, body):
    package = tmp_path / "pkgfake" / "inner"
    package.mkdir(parents=True)
    (tmp_path / "pkgfake" / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text(body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return package / "mod.py"


def manifest(tmp_path, original, patched):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"patches": [{"target": "pkgfake/inner/mod.py", "target_version": "1", "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                                             "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(), "patch_sha256": "p" * 64}]}), encoding="utf-8")
    return path


@pytest.mark.parametrize("installed, state", [("ORIGINAL", "original"), ("PATCHED", "patched"), ("SOMETHING ELSE", "unknown")])
def test_the_installed_file_is_classified_against_the_manifest(tmp_path, monkeypatch, installed, state):
    make_package(tmp_path, monkeypatch, installed)
    found = ri.patches_identity(manifest(tmp_path, "ORIGINAL", "PATCHED"))
    assert len(found) == 1 and found[0]["state"] == state and found[0]["installed_sha256"] == hashlib.sha256(installed.encode()).hexdigest()


def test_a_patched_and_an_unpatched_runtime_have_different_identities(tmp_path, monkeypatch):
    path = make_package(tmp_path, monkeypatch, "ORIGINAL")
    m = manifest(tmp_path, "ORIGINAL", "PATCHED")
    before = ri.runtime_identity(packages=("pytest",), manifest_path=m)
    path.write_text("PATCHED", encoding="utf-8")
    after = ri.runtime_identity(packages=("pytest",), manifest_path=m)
    assert before != after and before["patches"][0]["state"] == "original" and after["patches"][0]["state"] == "patched"


class WithRuntime(Decoder):
    def __init__(self, inner, runtime):
        self.inner, self.runtime = inner, runtime

    def decode(self, piece, **kwargs):
        return self.inner.decode(piece, **kwargs)

    def describe(self):
        return {**self.inner.describe(), "runtime": self.runtime}


BASE = {"packages": {"vllm": "0.16.0", "torch": "2.9.1"}, "patches": []}
PATCHED = {"packages": {"vllm": "0.16.0", "torch": "2.9.1"}, "patches": [{"target": "vllm/platforms/cuda.py", "state": "patched", "installed_sha256": "b" * 64}]}
NEWER = {"packages": {"vllm": "0.17.0", "torch": "2.9.1"}, "patches": []}


@pytest.mark.parametrize("first, second", [(BASE, PATCHED), (PATCHED, BASE), (BASE, NEWER)], ids=["unpatched-then-patched", "patched-then-unpatched", "other-vllm-version"])
def test_a_state_cannot_be_resumed_with_a_different_runtime(tmp_path, first, second):
    world = synthetic.World({"alpha": dict(break_at=2)})
    run(world, tmp_path, decoder=WithRuntime(world.decoder(), first))
    with pytest.raises(ResumeError, match="incompatible resume"):
        run(world, tmp_path, decoder=WithRuntime(world.decoder(), second))


def test_the_same_runtime_resumes(tmp_path):
    world = synthetic.World({"alpha": dict(break_at=2)})
    run(world, tmp_path, decoder=WithRuntime(world.decoder(), PATCHED))
    assert run(world, tmp_path, decoder=WithRuntime(world.decoder(), PATCHED)).exit_code == 0
