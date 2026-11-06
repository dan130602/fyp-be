"""Shared session fixtures. Nothing here runs unless a test asks for it."""

import pytest

import multilingual_support as M


@pytest.fixture(scope="session")
def chain250_root(tmp_path_factory):
    """A scratch multilingual release with ACCEPTED G3, G4 and G5 for all 250 calls (fake transports), built once per session.

    Read-only: tests copy it (`M.copy_chain`) before changing anything. About four minutes the first time it is requested.
    """
    tmp = tmp_path_factory.mktemp("chain250")
    chain = M.chain(tmp, list(M.upstream_inputs().call_ids))
    assert set(chain.run_g3().values()) == {"ACCEPTED"}
    assert set(chain.run_g4().values()) == {"ACCEPTED"}
    assert set(chain.run_g5().values()) == {"ACCEPTED"}
    return chain.root, tmp / "audit"


@pytest.fixture(scope="session")
def dialogues250_root(chain250_root, tmp_path_factory):
    """The chain above plus ACCEPTED G6 dialogues for all 250 calls (fake Batch). Read-only: tests copy it."""
    root, _ = chain250_root
    chain = M.copy_chain(root, tmp_path_factory.mktemp("dialogues250") / "dataset")
    assert set(chain.run_g6().values()) == {"ACCEPTED"}
    return chain.root, chain.audit_root


@pytest.fixture(scope="session")
def verified250_root(dialogues250_root, tmp_path_factory):
    """The dialogues above plus ACCEPTED, VALID G7 reports for all 250 calls (fake Gemini). Read-only: tests copy it."""
    root, _ = dialogues250_root
    chain = M.copy_chain(root, tmp_path_factory.mktemp("verified250") / "dataset")
    assert set(chain.run_g7().values()) == {"ACCEPTED"}
    return chain.root, chain.audit_root
