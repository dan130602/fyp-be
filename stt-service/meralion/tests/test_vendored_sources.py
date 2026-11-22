import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from meralion_worker import vendored
from meralion_worker.model_check import CheckError, blob_id, verify_vendored

# Each file is pinned to its OWN commit. Written out here so a wrong SOURCES.json cannot validate itself.
#   overlap, retry_plan, retry_merge: the frozen merge and retry logic at 2f982b2 (unchanged by the amt1 promotion)
#   formatting: the formatter with amount-v1 at 6feb3fa (adds 130 lines to the 2f982b2 blob, deletes none)
EXPECTED = {
    "overlap.py": ("8c1c1537b44050f7", "2f982b2"),
    "retry_plan.py": ("e663dde56300426b", "2f982b2"),
    "retry_merge.py": ("12e5d0c2d7e330c8", "2f982b2"),
    "formatting.py": ("d7601b00a1568dd8", "6feb3fa"),
}
FROZEN_FORMATTING_BLOB = "981c033743188894"          # the 2f982b2 version of formatting.py (percentages-v2 only), kept for the comparison below
SOURCE_PATHS = {
    "overlap.py": "stt-experiments/meralion-runner/overlap.py",
    "retry_plan.py": "stt-experiments/meralion-runner/retry_plan.py",
    "retry_merge.py": "stt-experiments/meralion-runner/retry_merge.py",
    "formatting.py": "stt-experiments/meralion/formatting.py",
}


def sources():
    return json.loads((vendored.DIRECTORY / "SOURCES.json").read_text(encoding="utf-8"))


def test_the_sources_file_is_schema_v2_with_a_commit_per_file_and_no_single_commit():
    record = sources()
    assert record["schema"] == "vendored-sources-v2" and "commit" not in record and set(record) == {"schema", "files"}


def test_every_file_matches_its_recorded_identity_and_its_own_pin():
    record = sources()
    assert set(record["files"]) == set(EXPECTED)
    for name, entry in record["files"].items():
        data = (vendored.DIRECTORY / name).read_bytes()
        blob, pin = EXPECTED[name]
        assert blob_id(data) == entry["git_blob"] and entry["git_blob"].startswith(blob)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["bytes"]
        assert entry["source_path"] == SOURCE_PATHS[name]
        assert entry["commit"].startswith(pin), (name, entry["commit"])


def test_the_formatter_is_the_only_file_pinned_to_the_newer_commit():
    commits = {name: entry["commit"][:7] for name, entry in sources()["files"].items()}
    assert commits == {"overlap.py": "2f982b2", "retry_plan.py": "2f982b2", "retry_merge.py": "2f982b2", "formatting.py": "6feb3fa"}


def test_verify_vendored_passes_on_the_real_directory_and_reports_each_files_commit():
    report = verify_vendored()
    assert set(report["files"]) == set(EXPECTED) and set(report["commits"]) == set(EXPECTED)
    for name, (blob, pin) in EXPECTED.items():
        assert report["commits"][name].startswith(pin) and report["files"][name].startswith(blob)
    assert len(report["sources_sha256"]) == 64


def test_the_files_are_exact_git_blob_bytes_at_their_pins():
    """Compare with git itself when the repository and the pinned commits are available (skipped otherwise)."""
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=vendored.DIRECTORY, capture_output=True, check=True).stdout.decode().strip()
        for name, (_, pin) in EXPECTED.items():
            expected = subprocess.run(["git", "rev-parse", f"{pin}:{SOURCE_PATHS[name]}"], cwd=root, capture_output=True, check=True).stdout.decode().strip()
            assert blob_id((vendored.DIRECTORY / name).read_bytes()) == expected
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git or a pinned commit is not available")


def test_the_new_formatter_blob_only_adds_to_the_frozen_one():
    """6feb3fa adds the amount-v1 code to formatting.py and deletes nothing, so percentages-v2 is untouched (the legacy replay re-proves it on 175 transcripts)."""
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=vendored.DIRECTORY, capture_output=True, check=True).stdout.decode().strip()
        frozen = subprocess.run(["git", "rev-parse", f"2f982b2:{SOURCE_PATHS['formatting.py']}"], cwd=root, capture_output=True, check=True).stdout.decode().strip()
        numstat = subprocess.run(["git", "diff", "--numstat", "2f982b2", "6feb3fa", "--", SOURCE_PATHS["formatting.py"]], cwd=root, capture_output=True, check=True).stdout.decode().split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git or a pinned commit is not available")
    assert frozen.startswith(FROZEN_FORMATTING_BLOB) and numstat[:2] == ["130", "0"]


def test_the_vendored_formatter_offers_both_formatters_and_their_versions():
    vendored.ensure_on_path()
    import formatting
    assert formatting.FORMATTER_VERSION == "2" and formatting.AMOUNT_FORMATTER_VERSION == "amount-v1"
    assert callable(formatting.format_percentages) and callable(formatting.join_split_amounts)


def copy_vendored(target):
    for name in list(EXPECTED) + ["SOURCES.json"]:
        (target / name).write_bytes((vendored.DIRECTORY / name).read_bytes())


def test_a_modified_vendored_file_is_detected(tmp_path):
    copy_vendored(tmp_path)
    verify_vendored(tmp_path)                                         # intact copy passes
    for name in EXPECTED:
        copy_vendored(tmp_path)
        (tmp_path / name).write_bytes((vendored.DIRECTORY / name).read_bytes() + b"\n# edited\n")
        with pytest.raises(CheckError, match=name):
            verify_vendored(tmp_path)


def test_a_missing_vendored_file_is_detected(tmp_path):
    copy_vendored(tmp_path)
    (tmp_path / "retry_plan.py").unlink()
    with pytest.raises(CheckError, match="missing"):
        verify_vendored(tmp_path)


def test_swapping_the_formatter_for_the_frozen_blob_is_detected(tmp_path):
    """The record pins formatting.py to the newer blob: putting the older (percentages-only) bytes back must fail the check."""
    copy_vendored(tmp_path)
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=vendored.DIRECTORY, capture_output=True, check=True).stdout.decode().strip()
        older = subprocess.run(["git", "cat-file", "blob", f"2f982b2:{SOURCE_PATHS['formatting.py']}"], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git or the older commit is not available")
    (tmp_path / "formatting.py").write_bytes(older)
    with pytest.raises(CheckError, match="formatting.py"):
        verify_vendored(tmp_path)


def test_line_endings_are_untouched():
    """The frozen files are LF; a CRLF working copy would change every hash."""
    for name in EXPECTED:
        assert b"\r\n" not in (vendored.DIRECTORY / name).read_bytes()
