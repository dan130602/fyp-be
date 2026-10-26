"""The versioned Singlish register guide: shared reference material for the authors and verifiers of SINGLISH calls.

The guide (`prompts/shared/register_singlish.<version>.md`) is an *upstream input* of every stage that reads it for a
SINGLISH call: its hash is recorded in that stage's attempts and artifacts and re-read from disk whenever staleness is
checked, so changing the current version marks in-flight attempts `STALE_UPSTREAM` and reports accepted outputs as stale,
through the existing shared machinery, and never silently. Calls of any other language profile never read the guide, so
their hashes, requests and histories are unaffected. Old versions stay on disk (a new version is a new file), so a recorded
hash can always be resolved back to the exact text.
"""

from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_DIR = PROJECT_ROOT / "prompts" / "shared"  # tests point this at a scratch copy to simulate a change
CURRENT_VERSION = "v1"
PROFILE = "SINGLISH"
HASH_KEY = "register_guide_sha256"


def guide_path(version=None, directory=None):
    return Path(directory or GUIDE_DIR) / f"register_singlish.{version or CURRENT_VERSION}.md"


def guide_text(version=None, directory=None):
    return guide_path(version, directory).read_text(encoding="utf-8").replace("\r\n", "\n")


def guide_sha256(version=None, directory=None):
    return text_hash(guide_text(version, directory))


def applies(spec):
    """Whether this call reads the guide (the SINGLISH profile only)."""
    return spec.get("metadata", {}).get("language_profile") == PROFILE


def extra_hashes(spec, directory=None):
    """{"register_guide_sha256": hash of the CURRENT guide file, re-read now} for a SINGLISH call, else {}."""
    return {HASH_KEY: guide_sha256(directory=directory)} if applies(spec) else {}


def fresh_hashes(root, call_id, directory=None):
    """The same, for a call identified by its spec file under `root` (used where only the dataset root is known)."""
    path = Path(root) / "authoring" / "call_specs" / f"{call_id}.json"
    return extra_hashes(parse_json_text(path.read_text(encoding="utf-8")), directory)


def brief_block(spec, directory=None):
    """The guide as it goes into a per-call brief (SINGLISH calls only; None otherwise)."""
    if not applies(spec):
        return None
    return {"version": CURRENT_VERSION, "text": guide_text(directory=directory)}


def version_of(sha256, directory=None):
    """The version whose file hashes to `sha256` (to rebuild a recorded request), or None when no stored version matches."""
    for path in sorted(Path(directory or GUIDE_DIR).glob("register_singlish.*.md")):
        version = path.name[len("register_singlish."):-len(".md")]
        if guide_sha256(version, directory) == sha256:
            return version
    return None
