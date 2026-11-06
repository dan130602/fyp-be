"""The shared fixture copy leaves rendered audio out by default and can include it on request.

The inclusion behaviour is proven with a tiny artificial release (a few bytes of fake "audio"), not with the production
recordings. One real pilot copy checks that the default really is light.
"""

import inspect

import pytest

import g3_support
import g4_support
import g5_support
import g6_support
import g7_support
from release_copy import copy_tree, is_render_path


def tiny_release(root):
    files = {
        "checklist.json": "[]", "authoring/call_specs/CALL_001.json": "{}", "authoring/plan_verification/CALL_001.json": "{}",
        "authoring/.stage_write.lock": "", "calls/CALL_001/ground_truth.json": "{}", "calls/CALL_001/transcript.json": "{}",
        "calls/CALL_001/renders/clean_voice_A/audio.mp3": "fake-mp3", "calls/CALL_001/renders/clean_voice_A/timings.json": "{}",
        "calls/CALL_001/renders/clean_voice_A/segments/0.mp3": "fake", "calls/CALL_002/renders/noisy/audio.mp3": "fake",
        "calls/.gitkeep": "",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def tree(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("relative,expected", [
    ("calls/CALL_001/renders", True), ("calls/CALL_001/renders/clean_voice_A/audio.mp3", True), ("calls/CALL_014/renders/x/segments/0.mp3", True),
    ("calls/CALL_001/ground_truth.json", False), ("calls/CALL_001", False), ("calls/.gitkeep", False), ("authoring/renders/x.json", False),
    ("renders/audio.mp3", False), ("calls/CALL_001/renders_notes.json", False),
])
def test_is_render_path(relative, expected):
    assert is_render_path(relative) is expected


def test_default_copy_leaves_rendered_audio_out(tmp_path):
    source = tiny_release(tmp_path / "source")
    copied = tree(copy_tree(source, tmp_path / "light"))
    assert not [p for p in copied if "/renders/" in "/" + p] and "calls/CALL_001/ground_truth.json" in copied and "calls/CALL_001/transcript.json" in copied
    assert not (tmp_path / "light/calls/CALL_001/renders").exists()


def test_keep_renders_includes_the_audio_files_exactly(tmp_path):
    source = tiny_release(tmp_path / "source")
    full = tmp_path / "full"
    copy_tree(source, full, keep_renders=True)
    assert tree(full) == tree(source)
    assert (full / "calls/CALL_001/renders/clean_voice_A/audio.mp3").read_text(encoding="utf-8") == "fake-mp3"


def test_prefix_and_suffix_skips_still_apply_with_and_without_renders(tmp_path):
    source = tiny_release(tmp_path / "source")
    for keep in (False, True):
        copied = tree(copy_tree(source, tmp_path / f"skip_{keep}", skip_prefixes=("authoring/plan_verification",),
                                skip_suffixes=(".stage_write.lock",), keep_renders=keep))
        assert "authoring/plan_verification/CALL_001.json" not in copied and "authoring/.stage_write.lock" not in copied
        assert "authoring/call_specs/CALL_001.json" in copied
        assert any("/renders/" in "/" + p for p in copied) is keep


@pytest.mark.parametrize("module", [g3_support, g4_support, g5_support, g6_support, g7_support])
def test_every_support_copy_helper_can_keep_renders(module):
    parameter = inspect.signature(module.copy_release).parameters["keep_renders"]
    assert parameter.default is False and parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_real_pilot_copy_is_light_and_still_complete_for_g3_to_g7(tmp_path):
    copied = g7_support.copy_release(tmp_path / "pilot")
    files = tree(copied)
    assert not [p for p in files if "/renders/" in "/" + p]
    assert "calls/CALL_001/ground_truth.json" in files and "authoring/call_specs/CALL_001.json" in files
    assert sum((copied / p).stat().st_size for p in files) < 60 * 1024 * 1024
