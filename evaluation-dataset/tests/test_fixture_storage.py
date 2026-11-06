"""Guard: test fixtures must never copy the real recordings by accident.

The pilot release is about 630 MB, 613 MB of it rendered mp3 audio under `calls/<id>/renders`. A single full test run used
to write tens of gigabytes of those copies. Two checks keep it from coming back:

* every shared copy helper, run on the real pilot, produces a light copy (no mp3, no large file, small total);
* no test module copies the pilot release wholesale: each local release-copy helper delegates to `release_copy.copy_tree`,
  and any direct `copytree` of the release leaves renders/mp3 out.

Tests that genuinely need audio opt in with `keep_renders=True` and should use tiny artificial files (see test_release_copy).
"""

import re
from pathlib import Path

import pytest

import g3_support
import g4_support
import g5_support
import g6_support
import g7_support
from fyp_evaluation.plan_verifications import controls as g5_controls
from fyp_evaluation.dialogue_verifications import controls as g7_controls

TESTS = Path(__file__).resolve().parent
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 60 * 1024 * 1024


@pytest.mark.parametrize("module", [g3_support, g4_support, g5_support, g6_support, g7_support, g5_controls, g7_controls], ids=lambda m: m.__name__)
def test_shared_copy_helpers_produce_light_copies_of_the_real_pilot(module, tmp_path):
    copied = (module.copy_release(g3_support.RELEASE, tmp_path / "pilot") if module in (g5_controls, g7_controls)
              else module.copy_release(tmp_path / "pilot"))
    files = [p for p in copied.rglob("*") if p.is_file()]
    assert files, "the copy is empty"
    assert not [p for p in files if p.suffix.lower() in {".mp3", ".wav", ".ogg"}]
    assert not [p for p in files if "renders" in p.relative_to(copied).parts]
    assert max(p.stat().st_size for p in files) < MAX_FILE_BYTES
    assert sum(p.stat().st_size for p in files) < MAX_TOTAL_BYTES


def _sources():
    return [p for p in sorted(TESTS.glob("*.py")) if p.name not in {"test_fixture_storage.py", "release_copy.py"}]


def test_local_release_copy_helpers_delegate_to_the_shared_lightweight_copy():
    offenders = []
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^def (_?copy_release)\(.*?(?=^def |\Z)", text, re.S | re.M):
            body = match.group(0)
            lightweight = "copy_tree(" in body or ("copytree(" in body and "ignore_patterns" in body and ("renders" in body or "mp3" in body))
            if not lightweight:
                offenders.append(f"{path.name}: {match.group(1)}")
    assert not offenders, "release-copy helpers that would copy the recordings: " + ", ".join(offenders)


def test_no_test_copies_the_whole_release_directly():
    offenders = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"copytree\(\s*(RELEASE|PILOT|SOURCE)\b", line) and not ("ignore_patterns" in line and ("renders" in line or "mp3" in line)):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, "use release_copy.copy_tree instead of copying the recordings: " + ", ".join(offenders)
