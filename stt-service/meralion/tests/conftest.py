import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


import pytest


@pytest.fixture
def ngram_env(monkeypatch):
    """Leaves MERALION_NGRAM_SIZE unset for the test and restores the original state afterwards, whatever the decoder under test sets."""
    monkeypatch.setenv("MERALION_NGRAM_SIZE", "tmp")            # registers the original state for the undo
    monkeypatch.delenv("MERALION_NGRAM_SIZE")
    return monkeypatch


@pytest.fixture(autouse=True)
def scheduling_env(monkeypatch):
    """No test starts with VLLM_ENABLE_V1_MULTIPROCESSING set, and none leaks it: a scheduling-profile decoder sets it in os.environ itself (follow-up E)."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "tmp")      # registers the original state for the undo
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING")
    return monkeypatch
