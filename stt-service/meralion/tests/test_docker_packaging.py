"""Stage 4 packaging, static checks only (no image is built, no Docker daemon, GPU, network or AWS is used). They pin the properties the Dockerfile, the allow-list and the vLLM lock must keep."""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
IGNORE = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
FREEZE = ROOT.parent.parent / "evaluation-engine" / "runs" / "worker-v0-env" / "vllm-pip-freeze.txt"            # git-ignored evidence: the tests that need it skip without it


def instructions():
    """The Dockerfile as a list of (keyword, text) with line continuations joined and comments dropped."""
    out, current = [], ""
    for raw in DOCKERFILE.splitlines():
        line = raw.rstrip()
        if not current and (not line.strip() or line.lstrip().startswith("#")):
            continue
        current += (" " if current else "") + line.rstrip("\\").strip()
        if not line.endswith("\\"):
            out.append((current.split(None, 1)[0].upper(), current))
            current = ""
    return out


def lock_pins(path):
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            assert "==" in line and " @ " not in line, line
            name, version = line.split("==", 1)
            pins[name.lower().replace("_", "-")] = version
    return pins


# ---- the Dockerfile -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_base_is_pinned_by_digest_amd64_and_a_cuda_devel_image():
    assert re.search(r"^ARG BASE_IMAGE=nvidia/cuda:12\.8\.0-devel-ubuntu24\.04@sha256:[0-9a-f]{64}$", DOCKERFILE, re.M)
    assert re.search(r"^FROM --platform=linux/amd64 \$\{BASE_IMAGE\}$", DOCKERFILE, re.M) and ":latest" not in DOCKERFILE
    assert "build-essential" in DOCKERFILE and "python3.12-dev" in DOCKERFILE and "libsndfile1" in DOCKERFILE        # FlashInfer and Triton compile at run time; soundfile needs libsndfile


def test_an_image_names_its_backend_and_configuration_there_is_no_default_to_promote():
    assert re.search(r"^ARG BACKEND$", DOCKERFILE, re.M) and re.search(r"^ARG RUNTIME_CONFIG$", DOCKERFILE, re.M)          # no `=value`: the build fails without them
    first_run = next(text for keyword, text in instructions() if keyword == "RUN")
    assert "BACKEND must be" in first_run and "RUNTIME_CONFIG is required" in first_run and "exit 2" in first_run
    config_lines = [text for keyword, text in instructions() if keyword == "ENV" and "MERALION_CONFIG" in text]
    assert config_lines and all("${RUNTIME_CONFIG}" in text for text in config_lines)
    code = " ".join(text for _, text in instructions())
    assert not re.search(r"vllm-rp105|nobudget|fee2|sched0|fp16", code)                                                   # no profile or config file is named in any instruction


def test_the_dependency_lock_is_selected_by_the_backend_and_installed_without_resolving():
    body = DOCKERFILE
    assert "COPY requirements-${BACKEND}.lock" in body and "pip install --no-cache-dir --no-deps ${EXTRA} -r /opt/meralion/requirements.lock" in body
    assert body.index('"${BACKEND}" = "transformers"') < body.index("download.pytorch.org/whl/cu128")                 # the extra index only for the Transformers lock
    assert (ROOT / "requirements-vllm.lock").is_file() and (ROOT / "requirements-transformers.lock").is_file()


def test_the_verified_wheel_is_installed_only_after_its_hash_is_checked_against_the_package_manifest():
    manifest = json.loads((ROOT / "configs" / "third_party" / "meralion-3-asr-0.1.0.json").read_text(encoding="utf-8"))
    name = manifest["artifact"]["filename"]
    assert f"COPY artifacts/{name} " in DOCKERFILE and len(manifest["artifact"]["sha256"]) == 64
    run = next(text for keyword, text in instructions() if keyword == "RUN" and "sha256sum -c" in text)
    assert run.index("sha256sum -c") < run.index("pip install") and "--no-deps" in run and "pip check" in run and "['artifact']['sha256']" in run
    wheel = ROOT / "artifacts" / name
    if wheel.is_file():
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]


def test_the_build_verifies_the_environment_and_the_checkpoint_and_the_checkpoint_is_the_last_big_layer():
    keywords = instructions()
    texts = [text for _, text in keywords]
    verify = next(i for i, text in enumerate(texts) if "verify-environment" in text)
    model_copy = next(i for i, text in enumerate(texts) if text.startswith("COPY model "))
    check = next(i for i, text in enumerate(texts) if "check-model" in text)
    code_copy = next(i for i, text in enumerate(texts) if text.startswith("COPY meralion_worker "))
    assert code_copy < verify < model_copy < check
    assert any("backend" in text and "decoder" in text and "RUNTIME_CONFIG names backend" in text for text in texts)        # the config's backend must equal the build's backend


def test_the_runtime_is_non_root_offline_with_writable_caches_and_no_engine_or_device_variable():
    keywords = instructions()
    assert [text for keyword, text in keywords if keyword == "USER"] == ["USER worker"]
    assert [text for keyword, text in keywords if keyword == "ENTRYPOINT"] == ['ENTRYPOINT ["python", "-m", "meralion_worker"]']
    env = " ".join(text for keyword, text in keywords if keyword == "ENV")
    for needed in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "FLASHINFER_WORKSPACE_BASE=", "VLLM_CACHE_ROOT=", "TRITON_CACHE_DIR=", "TORCHINDUCTOR_CACHE_DIR=", "XDG_CACHE_HOME=", "CUDA_HOME=/usr/local/cuda", "PATH=/opt/venv/bin"):
        assert needed in env, needed
    code = "\n".join(text for keyword, text in keywords)
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in code and "CUDA_VISIBLE_DEVICES" not in code and "VLLM_ATTENTION_BACKEND" not in code       # the profile sets its own; the scheduler owns the device
    assert "apply_patches" not in code and "COPY patches" in code                                                          # the manifest is carried, the MIG patch is never applied in the image
    assert "AWS_" not in code and "SECRET" not in code.upper() and "PASSWORD" not in code.upper()
    assert b"\r" not in (ROOT / "Dockerfile").read_bytes()


def test_every_copy_source_exists_or_is_a_declared_git_ignored_input():
    ignored_inputs = {"model", "artifacts"}                                                                                # git-ignored by design: the checkpoint and the verified wheel
    for keyword, text in instructions():
        if keyword != "COPY":
            continue
        source = text.split()[1]
        if "${" in source:
            source = source.replace("${BACKEND}", "vllm")
        top = source.split("/")[0]
        assert (ROOT / source).exists() or top in ignored_inputs, source


# ---- the allow-list -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_build_context_is_an_allow_list_that_excludes_tests_tools_and_run_outputs():
    assert IGNORE[0] == "*"
    allowed = {rule[1:] for rule in IGNORE if rule.startswith("!")}
    assert {"configs/**", "meralion_worker/**", "model/**", "patches/manifest.json", "patches/*.patch", "artifacts/*.whl", "requirements-*.lock"} <= allowed
    for never in ("tests", "tools", "out", "state", ".venv", ".provenance"):
        assert not any(rule.startswith(never) for rule in allowed), never
    assert "**/__pycache__" in IGNORE and "model/.cache" in IGNORE
    assert (ROOT / "patches" / "manifest.json").is_file() and list((ROOT / "patches").glob("*.patch"))


# ---- the vLLM lock ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_vllm_lock_is_exactly_pinned_has_no_machine_specific_line_and_carries_the_validated_stack():
    path = ROOT / "requirements-vllm.lock"
    pins = lock_pins(path)
    text = path.read_text(encoding="utf-8")
    assert " @ file:" not in text.replace("`meralion-3-asr @ file:///<cluster path>", "")                                   # only the explanatory header mentions the removed line
    assert "meralion-3-asr" not in pins                                                                                   # installed separately from the verified artifact
    expected = {"vllm": "0.16.0", "torch": "2.9.1", "flashinfer-python": "0.6.3", "transformers": "4.57.6", "ninja": "1.13.2", "nvidia-ml-py": "13.615.71", "soundfile": "0.14.0", "numpy": "2.2.6",
                "boto3": "1.44.3", "botocore": "1.44.3", "s3transfer": "0.19.3", "python-dateutil": "2.9.0.post0", "jmespath": "1.1.0", "six": "1.17.0", "urllib3": "2.8.0"}
    for name, version in expected.items():
        assert pins.get(name) == version, (name, pins.get(name))
    raw = [line.split("==")[0].lower().replace("_", "-") for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert len(raw) == len(set(raw))                                                                                      # no duplicate pin


@pytest.mark.skipif(not FREEZE.is_file(), reason="the saved cluster freeze (git-ignored evidence) is not present")
def test_the_vllm_lock_equals_the_verified_freeze_plus_exactly_the_four_storage_packages():
    freeze = {}
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        if line.strip() and " @ " not in line:
            name, version = line.split("==", 1)
            freeze[name.lower().replace("_", "-")] = version
    pins = lock_pins(ROOT / "requirements-vllm.lock")
    assert {k: v for k, v in pins.items() if k in freeze} == freeze                                                       # every frozen pin is carried unchanged
    assert set(pins) - set(freeze) == {"boto3", "botocore", "s3transfer", "python-dateutil"}
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() in (ROOT / "requirements-vllm.lock").read_text(encoding="utf-8")   # the header names the source it was built from


def test_the_runtime_identity_packages_are_pinned_in_both_locks():
    from meralion_worker import runtime_identity
    for lock in ("requirements-vllm.lock", "requirements-transformers.lock"):
        pins = lock_pins(ROOT / lock)
        for package in runtime_identity.PACKAGES:
            if package != "meralion-3-asr":
                assert package in pins or (lock == "requirements-transformers.lock" and package == "vllm"), (lock, package)


# ---- the wheel's declared dependencies versus each lock (Codex finding on the Transformers lock) ------------------------------------------------------------------------------------

def wheel_requirements():
    import zipfile
    manifest = json.loads((ROOT / "configs" / "third_party" / "meralion-3-asr-0.1.0.json").read_text(encoding="utf-8"))
    wheel = ROOT / "artifacts" / manifest["artifact"]["filename"]
    if not wheel.is_file():
        pytest.skip("the verified wheel (git-ignored) is not present")
    with zipfile.ZipFile(wheel) as archive:
        text = archive.read(next(name for name in archive.namelist() if name.endswith("METADATA"))).decode("utf-8")
    required = {}
    for line in text.splitlines():
        if line.startswith("Requires-Dist:") and "extra ==" not in line:
            spec = line.split(":", 1)[1].strip()
            required[re.split(r"[ <>=!~;\[(]", spec, 1)[0].lower().replace("_", "-")] = spec
    return required


def test_the_vllm_lock_pins_every_unconditional_dependency_of_the_verified_wheel():
    required = wheel_requirements()
    assert {"vllm", "fastapi", "uvicorn", "httpx", "python-multipart", "torch", "transformers"} <= set(required)
    pins = lock_pins(ROOT / "requirements-vllm.lock")
    assert [name for name in required if name not in pins] == []                                                       # `pip check` in the image cannot fail for a missing package
    assert pins["vllm"] == "0.16.0"                                                                                   # the one exact requirement of the wheel


def test_the_transformers_image_path_is_recorded_as_unvalidated_because_its_lock_misses_the_wheels_dependencies():
    required = wheel_requirements()
    pins = lock_pins(ROOT / "requirements-transformers.lock")
    missing = sorted(name for name in required if name not in pins)
    assert missing == ["fastapi", "httpx", "python-multipart", "uvicorn", "vllm"]                                      # the finding, pinned: if the lock is ever completed this test must be revisited with the markers below
    assert (ROOT / "requirements-transformers.lock").read_text(encoding="utf-8").startswith("# UNVALIDATED AS AN IMAGE ENVIRONMENT")
    assert "BACKEND=transformers is UNVALIDATED" in DOCKERFILE
