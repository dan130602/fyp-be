import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_tool = load_tool("make_model_manifest")
package_tool = load_tool("make_package_manifest")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_base(files, missing):
    return {"schema": "model-manifest-v1", "id": "m", "repo": "r", "revision": "rev", "kind": "complete_checkpoint", "status": "pending", "status_note": "x",
            "files_listing_complete": False, "files": files, "missing": missing, "hint_source": "h"}


def checkpoint(tmp_path):
    folder = tmp_path / "ckpt"
    (folder / ".cache").mkdir(parents=True)
    contents = {"model-00001-of-00001.safetensors": b"weights", "config.json": b"{}", "tokenizer.json": b"tok"}
    for name, data in contents.items():
        (folder / name).write_bytes(data)
    (folder / ".cache" / "junk").write_bytes(b"ignored")
    base = make_base({"model-00001-of-00001.safetensors": {"sha256": sha(b"weights"), "bytes": 7, "source": "records"}},
                     [{"name": "config.json", "hint_first8": sha(b"{}")[:8], "hint_last3": sha(b"{}")[-3:]},
                      {"name": "tokenizer.json", "hint_first8": sha(b"tok")[:8], "hint_last3": sha(b"tok")[-3:]}])
    return folder, base


def test_model_manifest_is_completed_from_the_directory(tmp_path):
    folder, base = checkpoint(tmp_path)
    manifest, problems = model_tool.build(base, model_tool.from_directory(folder), "dir")
    assert problems == [] and manifest["status"] == "complete" and manifest["files_listing_complete"] is True
    assert set(manifest["files"]) == {"model-00001-of-00001.safetensors", "config.json", "tokenizer.json"}      # .cache ignored
    assert "missing" not in manifest and "status_note" not in manifest


def test_model_manifest_refuses_a_value_that_differs_from_the_record(tmp_path):
    folder, base = checkpoint(tmp_path)
    (folder / "model-00001-of-00001.safetensors").write_bytes(b"tampered")
    manifest, problems = model_tool.build(base, model_tool.from_directory(folder), "dir")
    assert manifest is None and any("differs" in p for p in problems)


def test_model_manifest_refuses_a_missing_file_or_wrong_hint_or_adapter(tmp_path):
    folder, base = checkpoint(tmp_path)
    (folder / "tokenizer.json").unlink()
    assert any("not present" in p for p in model_tool.build(base, model_tool.from_directory(folder), "d")[1])
    (folder / "tokenizer.json").write_bytes(b"other")
    assert any("hints disagree" in p for p in model_tool.build(base, model_tool.from_directory(folder), "d")[1])
    (folder / "tokenizer.json").write_bytes(b"tok")
    (folder / "adapter_config.json").write_text("{}")
    assert any("adapter" in p for p in model_tool.build(base, model_tool.from_directory(folder), "d")[1])


def test_model_manifest_from_a_sha256sum_listing(tmp_path):
    listing, sizes = tmp_path / "sums.txt", tmp_path / "sizes.txt"
    listing.write_text(f"{sha(b'weights')}  ./model-00001-of-00001.safetensors\n{sha(b'{}')}  ./config.json\n", encoding="utf-8")
    sizes.write_text("7 ./model-00001-of-00001.safetensors\n2 ./config.json\n", encoding="utf-8")
    files = model_tool.from_listing(listing, sizes)
    assert files["config.json"] == {"sha256": sha(b"{}"), "bytes": 2}
    manifest, problems = model_tool.build(make_base({"model-00001-of-00001.safetensors": {"sha256": sha(b"weights"), "bytes": 7}}, []), files, "listing")
    assert problems == [] and manifest["status"] == "complete"


def test_model_tool_cli_writes_nothing_when_refused(tmp_path):
    folder, base = checkpoint(tmp_path)
    (folder / "config.json").write_bytes(b"changed")
    base_file, out = tmp_path / "base.json", tmp_path / "out.json"
    base_file.write_text(json.dumps(base))
    assert model_tool.main(["--base", str(base_file), "--dir", str(folder), "--out", str(out)]) == 1
    assert not out.exists()


def make_wheel(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def test_package_verdict_is_computed_from_the_files(tmp_path):
    installed = tmp_path / "meralion_3_asr"
    (installed / "backends").mkdir(parents=True)
    (installed / "__init__.py").write_bytes(b"")
    (installed / "backends" / "x.py").write_bytes(b"x=1\n")
    base = {"schema": "package-manifest-v1", "id": "pkg", "status": "pending", "status_note": "n", "provenance_verdict": "pending", "artifact": None, "files": {}, "hints": {}}
    base_file = tmp_path / "base.json"
    base_file.write_text(json.dumps(base))
    wheel = tmp_path / "meralion_3_asr-0.1.0-py3-none-any.whl"
    make_wheel(wheel, {"meralion_3_asr/__init__.py": b"", "meralion_3_asr/backends/x.py": b"x=1\n", "meralion_3_asr-0.1.0.dist-info/METADATA": b"m"})
    out = tmp_path / "out.json"
    assert package_tool.main(["--base", str(base_file), "--package-dir", str(installed), "--artifact", str(wheel), "--out", str(out)]) == 0
    manifest = json.loads(out.read_text())
    assert manifest["status"] == "complete" and manifest["provenance_verdict"] == "identical_to_pypi"
    assert manifest["artifact"]["filename"] == wheel.name and manifest["artifact"]["sha256"] == sha(wheel.read_bytes())
    assert set(manifest["files"]) == {"__init__.py", "backends/x.py"} and "hints" not in manifest

    make_wheel(wheel, {"meralion_3_asr/__init__.py": b"", "meralion_3_asr/backends/x.py": b"x=2\n"})
    assert package_tool.main(["--base", str(base_file), "--package-dir", str(installed), "--artifact", str(wheel), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["provenance_verdict"] == "differs_from_pypi"

    make_wheel(wheel, {"meralion_3_asr/__init__.py": b"", "meralion_3_asr/backends/x.py": b"x=1\n", "meralion_3_asr/extra.py": b"y"})
    package_tool.main(["--base", str(base_file), "--package-dir", str(installed), "--artifact", str(wheel), "--out", str(out)])
    assert json.loads(out.read_text())["provenance_verdict"] == "differs_from_pypi"             # a wheel member the install lacks also counts


def test_not_on_pypi_is_recorded_without_an_artifact(tmp_path):
    installed = tmp_path / "pkg"
    installed.mkdir()
    (installed / "__init__.py").write_bytes(b"")
    base_file, out = tmp_path / "base.json", tmp_path / "out.json"
    base_file.write_text(json.dumps({"id": "pkg", "status": "pending", "files": {}}))
    assert package_tool.main(["--base", str(base_file), "--package-dir", str(installed), "--not-on-pypi", "--out", str(out)]) == 0
    manifest = json.loads(out.read_text())
    assert manifest["provenance_verdict"] == "not_on_pypi" and manifest["artifact"] is None


def test_package_tool_needs_exactly_one_reference_choice(tmp_path):
    with pytest.raises(SystemExit):
        package_tool.main(["--base", "b", "--package-dir", str(tmp_path), "--out", "o"])
