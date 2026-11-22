import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from meralion_worker import cli
from meralion_worker import config as cfg
from meralion_worker import model_check as mc


def write_model(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_files = {}
    for name, data in files.items():
        (directory / name).write_bytes(data)
        manifest_files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "source": "test"}
    return {"id": "test-model", "kind": "complete_checkpoint", "status": "complete", "files": manifest_files}


REAL_MODEL = "meralion-3-3b-asr-base-9f226d7"
FILES = {"model-00001-of-00002.safetensors": b"A" * 100, "model-00002-of-00002.safetensors": b"B" * 50, "config.json": b"{}"}


def test_a_complete_manifest_passes(tmp_path):
    manifest = write_model(tmp_path / "m", FILES)
    report = mc.check_model(tmp_path / "m", manifest)
    assert report["files_checked"] == 3 and len(report["manifest_sha256"]) == 64


def test_an_adapter_is_rejected(tmp_path):
    manifest = write_model(tmp_path / "m", FILES)
    (tmp_path / "m" / "adapter_config.json").write_text("{}")
    with pytest.raises(mc.CheckError, match="adapter"):
        mc.check_model(tmp_path / "m", manifest)


def test_a_hash_mismatch_is_rejected(tmp_path):
    manifest = write_model(tmp_path / "m", FILES)
    (tmp_path / "m" / "model-00001-of-00002.safetensors").write_bytes(b"C" * 100)    # same size, different content
    with pytest.raises(mc.CheckError, match="hash differs"):
        mc.check_model(tmp_path / "m", manifest)


def test_a_size_mismatch_and_a_missing_file_are_rejected(tmp_path):
    manifest = write_model(tmp_path / "m", FILES)
    (tmp_path / "m" / "config.json").write_bytes(b"{ }")
    with pytest.raises(mc.CheckError, match="size differs"):
        mc.check_model(tmp_path / "m", manifest)
    (tmp_path / "m" / "config.json").unlink()
    with pytest.raises(mc.CheckError, match="missing"):
        mc.check_model(tmp_path / "m", manifest)


def test_an_extra_safetensors_file_is_rejected(tmp_path):
    manifest = write_model(tmp_path / "m", FILES)
    (tmp_path / "m" / "model-00003-of-00003.safetensors").write_bytes(b"x")
    with pytest.raises(mc.CheckError, match="not in the manifest"):
        mc.check_model(tmp_path / "m", manifest)


def test_a_missing_directory_and_a_wrong_kind_are_rejected(tmp_path):
    manifest = write_model(tmp_path / "m", FILES)
    with pytest.raises(mc.CheckError, match="not found"):
        mc.check_model(tmp_path / "absent", manifest)
    with pytest.raises(mc.CheckError, match="complete_checkpoint"):
        mc.check_model(tmp_path / "m", {**manifest, "kind": "adapter"})


# ---- the installed package check (locates and hashes; never imports) ------------------------------------------------------------------------------------------------

def package_manifest(directory, files, verdict="identical_to_pypi"):
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    return {"id": "pkg", "status": "complete", "provenance_verdict": verdict, "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}}


PACKAGE_FILES = {"__init__.py": b"", "prompts.py": b"PROMPT = 1\n", "backends/transformers_backend.py": b"x = 1\n"}


def test_a_matching_package_passes(tmp_path):
    manifest = package_manifest(tmp_path / "pkg", PACKAGE_FILES)
    assert mc.verify_package(manifest, tmp_path / "pkg")["files_checked"] == 3


def test_a_modified_or_missing_or_extra_package_file_is_rejected(tmp_path):
    manifest = package_manifest(tmp_path / "pkg", PACKAGE_FILES)
    (tmp_path / "pkg" / "prompts.py").write_bytes(b"PROMPT = 2\n")
    with pytest.raises(mc.CheckError, match="differs"):
        mc.verify_package(manifest, tmp_path / "pkg")
    (tmp_path / "pkg" / "prompts.py").write_bytes(PACKAGE_FILES["prompts.py"])
    (tmp_path / "pkg" / "extra.py").write_bytes(b"import os\n")
    with pytest.raises(mc.CheckError, match="not in the manifest"):
        mc.verify_package(manifest, tmp_path / "pkg")
    (tmp_path / "pkg" / "extra.py").unlink()
    (tmp_path / "pkg" / "backends" / "transformers_backend.py").unlink()
    with pytest.raises(mc.CheckError, match="missing"):
        mc.verify_package(manifest, tmp_path / "pkg")


def test_a_pending_verdict_or_missing_install_is_rejected(tmp_path):
    manifest = package_manifest(tmp_path / "pkg", PACKAGE_FILES, verdict="pending")
    with pytest.raises(mc.CheckError, match="verdict"):
        mc.verify_package(manifest, tmp_path / "pkg")
    with pytest.raises(mc.CheckError, match="not installed"):
        mc.verify_package({**manifest, "provenance_verdict": "identical_to_pypi"}, tmp_path / "absent")


def test_the_package_is_never_imported_by_the_check():
    """This machine has no meralion_3_asr; locating it must not raise and must not import torch."""
    program = ("import sys; sys.path.insert(0, '.'); from meralion_worker import model_check as m; "
               "print(m.installed_package_dir()); assert 'torch' not in sys.modules and 'meralion_3_asr' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", program], cwd=cfg.PACKAGE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- the command line ------------------------------------------------------------------------------------------------------------------------------------------------

def test_cli_reports_a_configuration_error_with_exit_3(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert cli.main(["check-model", "--config", str(bad)]) == 3
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is False


def test_cli_summary_can_also_be_written_to_a_file(capsys, tmp_path):
    target = tmp_path / "summary.json"
    (tmp_path / "empty-model").mkdir()                                   # an empty directory: the result must not depend on whether a local checkpoint exists
    assert cli.main(["check-model", "--model-dir", str(tmp_path / "empty-model"), "--summary-output", str(target)]) == 3
    assert json.loads(target.read_text(encoding="utf-8"))["exit_code"] == 3


# ---- the real manifests (Stage 0B: established from the Origami checkpoint and the PyPI wheel) -----------------------------------------------------------------------

def test_the_real_model_manifest_is_complete_and_matches_the_recorded_values():
    model = mc.load_model_manifest(REAL_MODEL, cfg.CONFIG_ROOT)
    assert model["status"] == "complete" and model["files_listing_complete"] is True and "missing" not in model
    assert model["kind"] == "complete_checkpoint" and model["revision"] == "9f226d738dee62c60a2f43b0ab7b9a2585a132c3"
    files = model["files"]
    assert "adapter_config.json" not in files
    assert all({"sha256", "bytes", "source"} <= set(record) for record in files.values())
    # the weights, as recorded in every earlier run's metadata
    assert files["model-00001-of-00002.safetensors"] == {**files["model-00001-of-00002.safetensors"], "bytes": 4971858576,
                                                         "sha256": "ea836de00d15a94685d9b69189d0073c9248aa264790634cda38d2cf9b26baac"}
    assert files["model-00002-of-00002.safetensors"] == {**files["model-00002-of-00002.safetensors"], "bytes": 1642292496,
                                                         "sha256": "0c1472c987cf7360c628c7da8881ff1aad6a7fa00d42de3d07424c0862b7804f"}
    # the custom model code, as recorded at run time in evaluation-engine/runs/*/preflight.json
    assert files["modeling_meralion3.py"]["sha256"] == "9ec21babefabaf32b17d8802032805cf01aa6226ee2a0b492d3524dc4cd43bf9"
    assert files["configuration_meralion3.py"]["sha256"] == "c857fe1d26ca32166a78395d846f78aa604af4fa338fc83d37d9cf03e02ee42f"
    assert files["processing_meralion3.py"]["sha256"] == "5644fcd8badc62aa2bffa69207d702cec53677488094c2c2d7a4671ed18a06be"
    assert {"config.json", "generation_config.json", "tokenizer.json", "tokenizer.model", "model.safetensors.index.json"} <= set(files)


def test_the_real_package_manifest_is_complete_and_names_its_verified_artifact():
    package = mc.load_json(cfg.CONFIG_ROOT / "third_party" / "meralion-3-asr-0.1.0.json")
    assert package["status"] == "complete" and package["provenance_verdict"] == "identical_to_pypi"
    assert package["artifact"] == {"filename": "meralion_3_asr-0.1.0-py3-none-any.whl", "sha256": "939911f3d5856a307adf723d042eac21842afc40a008e94e5f74245af5455b9a"}
    assert len(package["files"]) == 20
    # the partial hash hints that were recorded earlier (handoff section 4) agree with the computed hashes
    for name, first8 in (("_audio_io.py", "c22d8f71"), ("prompts.py", "21df0032"), ("backends/transformers_backend.py", "54f420a5"),
                         ("meralion_3_asr.py", "9a240365"), ("backends/vllm_backend.py", "36d6ecca")):
        assert package["files"][name].startswith(first8)


@pytest.mark.skipif(mc.installed_package_dir() is not None, reason="meralion_3_asr is installed here")
def test_verify_package_reports_not_installed_where_the_package_is_absent():
    package = mc.load_json(cfg.CONFIG_ROOT / "third_party" / "meralion-3-asr-0.1.0.json")
    with pytest.raises(mc.CheckError, match="not installed"):
        mc.verify_package(package)


@pytest.mark.skipif(not os.environ.get("MERALION_TEST_REAL_MODEL"), reason="set MERALION_TEST_REAL_MODEL=1 to hash the local 6.6 GB checkpoint (about 30 s)")
def test_the_real_local_checkpoint_matches_its_manifest():
    model_dir = cfg.PACKAGE_ROOT / "model"
    if not model_dir.is_dir():
        pytest.skip("no local model directory")
    report = mc.check_model(model_dir, mc.load_model_manifest(REAL_MODEL, cfg.CONFIG_ROOT))
    assert report["files_checked"] == len(mc.load_model_manifest(REAL_MODEL, cfg.CONFIG_ROOT)["files"])


# ---- refusing a pending manifest (fixtures; the real manifests are complete now) ---------------------------------------------------------------------------------------

@pytest.fixture
def pending_configs(tmp_path, monkeypatch):
    root = tmp_path / "configs"
    shutil.copytree(cfg.CONFIG_ROOT, root)
    for relative, note in (("models/meralion-3-3b-asr-base-9f226d7.json", "model identity not established (test)"),
                           ("third_party/meralion-3-asr-0.1.0.json", "package provenance not established (test)")):
        path = root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(status="pending", status_note=note)
        path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(cfg, "CONFIG_ROOT", root)
    return root


def test_pending_manifests_are_refused_by_the_library(pending_configs):
    with pytest.raises(mc.CheckError, match="pending"):
        mc.check_model(pending_configs, mc.load_model_manifest(REAL_MODEL, pending_configs))
    with pytest.raises(mc.CheckError, match="pending"):
        mc.verify_package(mc.load_json(pending_configs / "third_party" / "meralion-3-asr-0.1.0.json"))


def test_cli_refuses_pending_manifests_with_exit_3(capsys, pending_configs):
    assert cli.main(["check-model"]) == 3
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["ok"] is False and summary["exit_code"] == 3 and "pending" in summary["error"]
    assert cli.main(["verify-environment"]) == 3
    assert "pending" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_cli_check_model_fails_on_an_incomplete_directory_with_the_real_manifest(capsys, tmp_path):
    (tmp_path / "model").mkdir()
    assert cli.main(["check-model", "--model-dir", str(tmp_path / "model")]) == 3
    error = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert "missing" in error or "not in the manifest" in error


@pytest.mark.skipif(mc.installed_package_dir() is not None, reason="meralion_3_asr is installed here")
def test_cli_verify_environment_passes_the_vendored_check_then_fails_on_the_absent_package(capsys):
    assert cli.main(["verify-environment"]) == 3
    assert "not installed" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
