"""Resume and progress state for one invocation (plan: "Resume and progress").

Everything lives under <state storage>/<invocation id>/ and every file is written exactly once with write_exclusive, so nothing is ever overwritten and a restart can only add what is missing:

    invocation.json          the identity of the run: ordered jobs, effective config and fingerprint, model manifest hash, vendored-sources hash, decoder identity, worker version
    recordings/<job>.json    input sha256, samples, source rate           failed/<job>.json    final input failure
    first_pass/<job>.json    window bounds and every DecodeResult         plan.json            the retry selection, written once, never recomputed
    retries/<retry_id>.json  one DecodeResult                             done/<job>.json      output key and output sha256

An exclusive create stops two processes from writing divergent plans or overwriting outputs. It is NOT a guard against two processes doing the same work at the same time.
"""
import hashlib
import json
import re

from meralion_worker.storage import AlreadyExists

SCHEMA = "meralion-worker-state-v1"
INVOCATION_ID = re.compile(r"[0-9A-Za-z._-]{1,128}")                          # always used with fullmatch()
# output_sha256 is the hash of an output file that contains timing fields, so it differs between two runs that produced identical content
VOLATILE_KEYS = frozenset({"decode_seconds", "batch_seconds", "created_at", "timing", "model_load_seconds", "output_sha256"})
IDENTITY_FIELDS = ("invocation_id", "jobs", "config", "config_fingerprint", "model_manifest_sha256", "vendored_sources_sha256", "decoder", "worker_version")


class ResumeError(Exception):
    """The state or the outputs on disk are incompatible with this run (exit code 3). Nothing is mixed, nothing is overwritten."""


def canonical_bytes(document):
    return (json.dumps(document, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def stable_view(value):
    """The document without its volatile fields (timings, creation times): what two runs must agree on exactly (C3)."""
    if isinstance(value, dict):
        return {k: stable_view(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [stable_view(v) for v in value]
    return value


class InvocationState:
    def __init__(self, storage, invocation_id):
        if not INVOCATION_ID.fullmatch(invocation_id):
            raise ValueError(f"invocation id must fully match {INVOCATION_ID.pattern}")
        self.storage, self.invocation_id = storage, invocation_id

    def key(self, *parts):
        return "/".join((self.invocation_id,) + parts)

    def exists(self, *parts):
        return self.storage.exists(self.key(*parts))

    def read_bytes(self, *parts):
        return self.storage.read(self.key(*parts))

    def read(self, *parts):
        return json.loads(self.read_bytes(*parts).decode("utf-8"))

    def put_once(self, *parts, document):
        """Write the document if the file does not exist yet. Returns (created, sha256 of the file's bytes). An existing file is left untouched and its hash returned."""
        data = canonical_bytes({"schema": SCHEMA, **document})
        try:
            self.storage.write_exclusive(self.key(*parts), data)
            return True, sha256_bytes(data)
        except AlreadyExists:
            return False, sha256_bytes(self.read_bytes(*parts))

    def begin(self, record):
        """Create invocation.json, or check an existing one is identical in every identity field. Returns True when this is a resumed run."""
        created, _ = self.put_once("invocation.json", document=record)
        if created:
            return False
        existing = self.read("invocation.json")
        different = [field for field in IDENTITY_FIELDS if stable_view(existing.get(field)) != stable_view(record.get(field))]
        if different:
            raise ResumeError(f"incompatible resume of invocation {self.invocation_id!r}: {different} differ from the stored invocation.json")
        return True
