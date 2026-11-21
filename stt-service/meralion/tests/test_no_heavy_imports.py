import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WORKER = PACKAGE_ROOT / "meralion_worker"
HEAVY = ("torch", "meralion_3_asr", "vllm", "transformers", "boto3", "botocore")
FORBIDDEN = ("transcribe_eval", "critical_check", "config_compare", "fast_score", "diag_report", "evaluation_engine", "evaluation_dataset", "evaluation-engine", "evaluation-dataset")


def worker_modules():
    names = {p.relative_to(PACKAGE_ROOT).with_suffix("").as_posix().replace("/", ".").removesuffix(".__init__")
             for p in WORKER.rglob("*.py") if "vendored" not in p.parts or p.name == "__init__.py"}
    return sorted(names)


def test_importing_every_worker_module_loads_no_heavy_library():
    """Run in a fresh interpreter: Stage 1 modules must be importable on a machine with no torch, MERaLiON package, vLLM or even boto3."""
    modules = [m for m in worker_modules() if not m.endswith("__main__")]
    program = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"loaded = [h for h in {HEAVY!r} if h in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('ok', len(sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", program], cwd=PACKAGE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_module_imports_evaluation_or_benchmark_code():
    offenders = []
    for path in sorted(WORKER.rglob("*.py")):
        if "vendored" in path.parts and path.name != "__init__.py":
            continue                                       # frozen copies are checked by hash, not by content
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value] if any(marker in node.value for marker in ("evaluation-engine", "evaluation-dataset", "transcribe_eval")) else []
            for name in names:
                if any(bad in name for bad in FORBIDDEN):
                    offenders.append((path.name, name))
    assert not offenders, offenders


def test_vendored_modules_import_only_the_standard_library():
    """overlap, retry_plan, retry_merge and formatting must stay free of third-party and evaluation imports."""
    stdlib = set(sys.stdlib_module_names) | {"overlap", "retry_plan", "retry_merge", "formatting"}
    for name in ("overlap.py", "retry_plan.py", "retry_merge.py", "formatting.py"):
        tree = ast.parse((WORKER / "vendored" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            assert all(root in stdlib for root in roots), (name, roots)
