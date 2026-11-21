"""REV 3.2 Amendment 6: the controlled logits test never runs where a device UUID is visible, and its report is bound to the commit, package files and runtime it ran with."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ORIGAMI = Path(__file__).resolve().parent.parent / "tools" / "origami"
SOURCE = (ORIGAMI / "plugin_logits_test.py").read_text(encoding="utf-8")
HEAVY = ("torch", "vllm", "meralion_3_asr")


def load():
    spec = importlib.util.spec_from_file_location("plugin_logits_test_under_test", ORIGAMI / "plugin_logits_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load()


class Recorder(importlib.abc.MetaPathFinder):
    """Records every import request of a heavy module; never provides one."""

    def __init__(self):
        self.requested = []

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            self.requested.append(name)
        return None


@pytest.fixture
def recorder(monkeypatch):
    finder = Recorder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for name in [m for m in sys.modules if m.split(".")[0] in HEAVY]:
        monkeypatch.delitem(sys.modules, name)
    return finder


@pytest.mark.parametrize("value, visible", [("MIG-abc", True), ("GPU-abc", True), ("0", False), ("0,1", False), ("", False), (None, False), ("1,MIG-x", True), (" GPU-x", True)])
def test_uuid_device_visible(value, visible):
    env = {} if value is None else {"CUDA_VISIBLE_DEVICES": value}
    assert tool.uuid_device_visible(env) is visible


@pytest.mark.parametrize("value", ["MIG-22222222-bbbb", "GPU-33333333-cccc"])
def test_a_visible_device_uuid_refuses_before_any_heavy_import_and_writes_no_report(tmp_path, monkeypatch, capsys, recorder, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    out = tmp_path / "report.json"
    assert tool.main(["--out", str(out)]) == 3
    assert tool.REFUSAL in capsys.readouterr().err and not out.exists() and recorder.requested == []


@pytest.mark.parametrize("value", ["0", None])
def test_a_numeric_or_unset_device_passes_the_guard_and_then_needs_a_clean_recorded_commit(tmp_path, monkeypatch, capsys, recorder, value):
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    monkeypatch.delenv("MERALION_WORKER_GIT_COMMIT", raising=False)
    monkeypatch.setattr(tool, "CODE_ROOT", tmp_path / "no-code-folder")
    out = tmp_path / "report.json"
    assert tool.main(["--out", str(out)]) == 3                       # not the device guard: the refusal names the missing commit
    assert "no clean recorded worker commit" in capsys.readouterr().err and not out.exists() and recorder.requested == []


@pytest.mark.parametrize("content, env, expected", [
    ("a" * 40, {}, "a" * 40), ("unknown", {}, None), ("a" * 40 + "-dirty", {}, None), ("", {}, None), (None, {"MERALION_WORKER_GIT_COMMIT": "b" * 40}, "b" * 40),
    (None, {}, None), (None, {"MERALION_WORKER_GIT_COMMIT": "unknown"}, None), ("c" * 40, {"MERALION_WORKER_GIT_COMMIT": "b" * 40}, "c" * 40)])
def test_the_worker_commit_comes_from_code_commit_and_refuses_unknown_and_dirty(tmp_path, content, env, expected):
    if content is not None:
        (tmp_path / "CODE_COMMIT").write_text(content + "\n", encoding="utf-8")
    assert tool.worker_commit(tmp_path, env) == expected


def test_the_guard_is_the_first_statement_of_main_and_no_heavy_module_is_imported_at_module_level():
    tree = ast.parse(SOURCE)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first = main.body[0]
    assert isinstance(first, ast.If) and isinstance(first.test, ast.Call) and first.test.func.id == "uuid_device_visible" and isinstance(first.body[-1], ast.Return)
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name.split(".")[0] for node in top_level if isinstance(node, ast.Import) for alias in node.names} | {node.module.split(".")[0] for node in top_level if isinstance(node, ast.ImportFrom)}
    assert not names & set(HEAVY)
    for function in ("uuid_device_visible", "worker_commit", "collect_evidence"):
        body = ast.get_source_segment(SOURCE, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function))
        assert "import torch" not in body and "import vllm" not in body and "from vllm" not in body                  # the evidence is read from metadata and files only


def test_the_evidence_hashes_the_exercised_package_files_the_tool_and_the_versions(tmp_path, monkeypatch, recorder):
    package = tmp_path / "pkgroot" / "meralion_3_asr"
    (package / "_vllm_plugin" / "transformers_utils").mkdir(parents=True)
    contents = {"_vllm_plugin/__init__.py": b"init", "_vllm_plugin/transformers_utils/no_repeat_logits_processor.py": b"processor"}
    for name, data in contents.items():
        (package / name).write_bytes(data)
    (package / "__init__.py").write_bytes(b"")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    evidence = tool.collect_evidence("d" * 40, {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_ID": "55"})
    assert evidence["package"]["files_sha256"] == {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    assert evidence["worker_git_commit"] == "d" * 40 and evidence["tool_sha256"] == hashlib.sha256((ORIGAMI / "plugin_logits_test.py").read_bytes()).hexdigest()
    assert evidence["uuid_device_visible"] is False and evidence["cuda_visible_devices"] == "0" and evidence["slurm_job_id"] == "55" and set(evidence["versions"]) == {"vllm", "torch", "meralion-3-asr"}
    assert not [m for m in sys.modules if m.split('.')[0] in HEAVY]      # the spec lookup is not an import: nothing heavy was executed


def test_the_evidence_of_a_missing_package_has_no_file_hashes_rather_than_invented_ones(recorder):
    evidence = tool.collect_evidence("d" * 40, {})
    assert evidence["package"]["files_sha256"] == {} or all(len(v) == 64 for v in evidence["package"]["files_sha256"].values())
    assert evidence["uuid_device_visible"] is False


def test_the_report_carries_the_bound_evidence_in_its_source():
    assert '"evidence": evidence' in SOURCE and "evidence[\"platform_class\"]" in SOURCE and "worker_commit()" in SOURCE


# ---- REV 4.1 A1: complete cases, a derived result, and a tool hash bound to the COMMITTED file ----------------------------------------------------------------------------------------

def load_decision():
    spec = importlib.util.spec_from_file_location("decision_under_plugin_test", ORIGAMI.parent / "decision.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ORIGAMI.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ORIGAMI.parent))
    return module


decision = load_decision()

LINES_LF = b"line one" + bytes([10]) + b"line two" + bytes([10])
LINES_CRLF = b"line one" + bytes([13, 10]) + b"line two" + bytes([13, 10])
MIXED = b"a" + bytes([13, 10]) + b"b" + bytes([10])


def test_the_evaluators_required_cases_are_the_tools_declared_cases():
    assert tuple(decision.REQUIRED_PLUGIN_CASES) == tuple(tool.CASE_NAMES) and len(set(tool.CASE_NAMES)) == len(tool.CASE_NAMES) == 12


def test_run_cases_assigns_exactly_the_declared_case_names_and_checks_them_at_run_time():
    tree = ast.parse(SOURCE)
    run_cases = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_cases")
    assigned = [node.slice.value for node in ast.walk(run_cases) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "results"
                and isinstance(node.slice, ast.Constant) and isinstance(node.ctx, ast.Store)]
    assert set(assigned) == set(tool.CASE_NAMES) and len(set(assigned)) == 12         # none dropped, none added (the swap case is assigned in both try branches; the ORDER is checked at run time)
    assert "check_declared(results)" in ast.get_source_segment(SOURCE, run_cases)    # and the run-time check that raises if they drift


@pytest.mark.parametrize("results, passed, failed, untested", [
    ({name: True for name in tool.CASE_NAMES}, True, [], []),
    ({**{name: True for name in tool.CASE_NAMES}, tool.CASE_NAMES[0]: False}, False, [tool.CASE_NAMES[0]], []),
    ({**{name: True for name in tool.CASE_NAMES}, tool.CASE_NAMES[8]: None}, False, [], [tool.CASE_NAMES[8]]),                       # a case that could not run is not a pass
    ({}, False, [], []),                                                                                                           # no cases at all is not a pass
    ({name: True for name in tool.CASE_NAMES[:-1]}, False, [], []),                                                                # a missing case is not a pass
])
def test_the_summary_is_derived_from_the_cases(results, passed, failed, untested):
    assert tool.summarise(results) == (passed, failed, untested)


def test_the_report_summary_is_derived_in_main():
    assert "summarise(results)" in SOURCE and '"passed": passed' in SOURCE and '"passed": not failed' not in SOURCE


def make_git_repo(path, content=b"def main(): pass"):
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True).stdout.strip()
    path.mkdir(parents=True)
    git("init", "-q")
    target = path / decision.PLUGIN_TOOL_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    git("add", "-f", ".")
    git("commit", "-q", "-m", "tool")
    return path, git("rev-parse", "HEAD"), target


def documents():
    packages = {"vllm": "0.16.0", "torch": "2.9.1+cu128", "meralion-3-asr": "0.1.0"}
    return {"alpha": {"meralion": {"decoder": {"runtime": {"packages": packages}}}}}


def report_for(commit, tool_sha256, manifest):
    cases = {name: True for name in decision.REQUIRED_PLUGIN_CASES}
    return {"test": "plugin_logits_test", "passed": True, "failed": [], "not_testable_here": [], "cases": cases,
            "evidence": {"worker_git_commit": commit, "tool_sha256": tool_sha256, "package": {"files_sha256": {name: manifest["files"][name] for name in decision.PLUGIN_FILES}},
                         "versions": {"vllm": "0.16.0", "torch": "2.9.1+cu128", "meralion-3-asr": "0.1.0"}, "cuda_visible_devices": None, "uuid_device_visible": False}}


def test_the_tool_hash_is_checked_against_the_committed_blob_in_a_real_repository(tmp_path):
    repo, commit, target = make_git_repo(tmp_path / "repo")
    manifest = json.loads((ORIGAMI.parent.parent / "configs" / "third_party" / "meralion-3-asr-0.1.0.json").read_text(encoding="utf-8"))
    real = hashlib.sha256(target.read_bytes()).hexdigest()
    assert decision.committed_blob_sha256(repo, commit, decision.PLUGIN_TOOL_PATH) == (real, None)
    assert decision.plugin_problems(report_for(commit, real, manifest), manifest, documents(), repo=repo) == []
    for label, report, fragment in (
            ("zero hash", report_for(commit, "0" * 64, manifest), "tool_sha256 is"),
            ("the hash of another file", report_for(commit, hashlib.sha256(b"another file").hexdigest(), manifest), "tool_sha256 is"),
            ("an unknown commit", report_for("f" * 40, real, manifest), "cannot be established"),
            ("no commit", report_for(None, real, manifest), "names no worker commit")):
        problems = decision.plugin_problems(report, manifest, documents(), repo=repo)
        assert any(fragment in problem for problem in problems), (label, problems)
    missing_path = decision.plugin_problems(report_for(commit, real, manifest), manifest, documents(), repo=tmp_path)           # not a repository at all
    assert any("cannot be established" in problem for problem in missing_path)


def test_the_committed_blob_wins_over_a_changed_working_tree_file(tmp_path):
    repo, commit, target = make_git_repo(tmp_path / "repo", content=LINES_LF)
    manifest = json.loads((ORIGAMI.parent.parent / "configs" / "third_party" / "meralion-3-asr-0.1.0.json").read_text(encoding="utf-8"))
    committed = hashlib.sha256(LINES_LF).hexdigest()
    target.write_bytes(LINES_CRLF)                                      # the laptop's working copy differs (CRLF), the commit does not
    assert decision.plugin_problems(report_for(commit, committed, manifest), manifest, documents(), repo=repo) == []
    working_tree_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    assert any("tool_sha256 is" in p for p in decision.plugin_problems(report_for(commit, working_tree_hash, manifest), manifest, documents(), repo=repo))


def test_a_blob_is_read_as_bytes_without_text_conversion(tmp_path):
    repo, commit, target = make_git_repo(tmp_path / "repo", content=MIXED)                  # a blob that really contains CRLF
    assert decision.committed_blob_sha256(repo, commit, decision.PLUGIN_TOOL_PATH)[0] == hashlib.sha256(MIXED).hexdigest()


def test_the_declared_case_names_are_enforced_when_the_cases_are_produced():
    full = {name: True for name in tool.CASE_NAMES}
    assert tool.check_declared(dict(full)) == full
    dropped = {name: True for name in tool.CASE_NAMES[:-1]}
    renamed = {**{name: True for name in tool.CASE_NAMES[:-1]}, "something_else": True}
    reordered = dict(reversed(list(full.items())))
    for bad in (dropped, renamed, reordered, {}, {**full, "extra": True}):
        with pytest.raises(RuntimeError, match="declared CASE_NAMES"):
            tool.check_declared(bad)
    tree = ast.parse(SOURCE)
    run_cases = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_cases")
    assert "check_declared(results)" in ast.get_source_segment(SOURCE, run_cases)
