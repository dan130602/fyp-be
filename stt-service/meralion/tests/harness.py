"""Test harness for pipeline runs: crash injection and snapshots. No git-ignored data is read."""
import json
from pathlib import Path

import synthetic
from meralion_worker import pipeline
from meralion_worker.decoders.base import Decoder
from meralion_worker.state import stable_view
from meralion_worker.storage import LocalStorage


class SimulatedCrash(Exception):
    """Stands for a process dying at a chosen point (nothing in the pipeline catches it)."""


class CrashingDecoder(Decoder):
    """Delegates to a decoder and dies on call number after + 1 (calls are counted per window, since decode_many loops decode)."""

    def __init__(self, inner, after):
        self.inner, self.after, self.count = inner, after, 0

    def decode(self, piece, *, recording_id, start_sample, end_sample):
        self.count += 1
        if self.count > self.after:
            raise SimulatedCrash(f"decoder died on call {self.count}")
        return self.inner.decode(piece, recording_id=recording_id, start_sample=start_sample, end_sample=end_sample)

    def describe(self):
        return self.inner.describe()


class FlakyStorage(LocalStorage):
    """A LocalStorage that dies on the n-th write whose key matches the predicate (the write itself does NOT happen)."""

    def __init__(self, root, predicate, nth=1):
        super().__init__(root)
        self.predicate, self.nth, self.seen = predicate, nth, 0

    def write_exclusive(self, key, data):
        if self.predicate(key):
            self.seen += 1
            if self.seen == self.nth:
                raise SimulatedCrash(f"storage died writing {key}")
        return super().write_exclusive(key, data)


def run(world, tmp_path, *, decoder=None, names=None, invocation_id="inv-1", effective=None, context=None, output=None, state=None, open_recording=None, jobs=None, base="run"):
    out = output if output is not None else LocalStorage(Path(tmp_path) / base / "out")
    st = state if state is not None else LocalStorage(Path(tmp_path) / base / "state")
    return pipeline.run_invocation(
        invocation_id=invocation_id, effective=effective or synthetic.effective(), jobs=jobs if jobs is not None else world.jobs(names), decoder=decoder or world.decoder(),
        open_recording=open_recording or world.open_recording, output_storage=out, state_storage=st, context=context or synthetic.context())


def snapshot(root):
    """Every JSON file under root, keyed by its relative path, reduced to its stable view (volatile fields dropped)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): stable_view(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(root.rglob("*.json"))}


def raw_files(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def two_jobs():
    """alpha: seam 2 broken (one retry); beta: seam 4 broken and 'half a percent' in an unshared stretch (one retry, one formatter change). 14 windows: run budget 2, both retries fit."""
    return synthetic.World({"alpha": dict(break_at=2), "beta": dict(break_at=4, replace_first="half a percent")})
