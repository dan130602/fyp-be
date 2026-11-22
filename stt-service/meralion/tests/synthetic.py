"""Synthetic fixtures for the pipeline tests. They read no git-ignored data: texts are generated, audio is a length only (ReplayRecording).

A 120 s recording has 7 FG windows (30 s every 15 s), so the invocation-wide run budget is floor(0.15 x 7) = 1. Consecutive windows share ten words, so every seam anchors except the one
the fixture breaks on purpose (window break_at + 1 starts with different words): that seam is "unresolved" and triggers exactly one retry. A retry text made of the last ten words of the left
window and the first ten of the right window anchors to both neighbours and is inserted. The tests assert these seam statuses before relying on them.
"""
import itertools
import json

from meralion_worker import config as cfg
from meralion_worker import strategy_fg
from meralion_worker.audio import InputError, ReplayRecording
from meralion_worker.decoders.base import DecodeResult
from meralion_worker.decoders.replay_decoder import ReplayDecoder, build_table
from meralion_worker.pipeline import Job, RunContext

SAMPLES_120S = 120 * 16000
_SYLLABLES = [c + v for c in "bdfgklmprstvz" for v in "aeiou"]
WORDS = [w for w in ("".join(p) for p in itertools.product(_SYLLABLES, repeat=2)) if w not in {"nono", "kuku"}][:600]


def window_texts(num_windows, break_at=None, replace_first=None):
    """One text per window. Window i holds words[10i : 10i + 20]. break_at=k replaces the first ten words of window k + 1. replace_first replaces words 3..5 of window 0 (an unshared stretch)."""
    texts = []
    for i in range(num_windows):
        chunk = WORDS[10 * i: 10 * i + 20]
        if break_at is not None and i == break_at + 1:
            chunk = WORDS[300 + 10 * i: 300 + 10 * i + 10] + chunk[10:]
        if i == 0 and replace_first:
            chunk = chunk[:3] + replace_first.split() + chunk[3 + len(replace_first.split()):]
        texts.append(" ".join(chunk))
    return texts


def retry_text_for(texts, k):
    """The last ten words of window k and the first ten of window k + 1."""
    return " ".join(texts[k].split()[-10:] + texts[k + 1].split()[:10])


def table_for(job, num_samples=SAMPLES_120S, break_at=2, replace_first=None, with_retry=True):
    """{(job, start, end): text} for the FG windows and (when a seam is broken) the retry window the planner will choose for it."""
    bounds = strategy_fg.windows_for(num_samples)
    texts = window_texts(len(bounds), break_at=break_at, replace_first=replace_first)
    table = {(job, a, b): t for (a, b), t in zip(bounds, texts)}
    if with_retry and break_at is not None:
        pred = strategy_fg.first_merge(job, num_samples, [{"start_sample": a, "end_sample": b, "text": t} for (a, b), t in zip(bounds, texts)])
        from meralion_worker.retry_selection import select_retries
        selection = select_retries({job: pred}, kinds={"unresolved", "low_margin_conflict"}, unresolved_call_cap=0.10, run_cap=1.0)
        for group in selection.selected:
            table[(job, group["window"][0], group["window"][1])] = retry_text_for(texts, break_at)
    return table


def effective(formatter=None, env=None):
    """The packaged DEFAULT config (the selected amt1 pipeline), with the formatter replaced only when one is given."""
    data = json.loads(cfg.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if formatter is not None:
        data["formatter"] = formatter
    return cfg.load_runtime_config(data, env=env or {})


def budgeted_effective(env=None):
    """The amt1 configuration WITH the invocation-wide retry budget (strategy v1), loaded from its own file."""
    return cfg.load_runtime_config(cfg.BUDGETED_CONFIG, env=env or {})


def legacy_effective(env=None):
    """The preserved legacy config (formatter percentages-v2, config_id ...-fee2), loaded from its own file."""
    return cfg.load_runtime_config(cfg.LEGACY_CONFIG, env=env or {})


def context():
    return RunContext(model={"manifest": "test-model", "manifest_sha256": "m" * 64, "revision": "r"}, worker={"version": "0.1.0", "git_commit": "test", "vendored": {}},
                      vendored_sources_sha256="v" * 64, model_manifest_sha256="m" * 64)


class World:
    """A set of synthetic recordings with scripted decoder texts. Each call to decoder() is a fresh ReplayDecoder, like a new process after a crash."""

    def __init__(self, specs):
        """specs: {job: dict(num_samples=..., break_at=..., replace_first=..., with_retry=...)}  (a job missing from the dict does not exist)"""
        self.specs = specs
        self.table = {}
        for job, spec in specs.items():
            self.table.update(table_for(job, **spec))
        self.failing = {}                                           # job -> InputError text
        self.opened = []                                            # every job a recording was opened for, in order

    def jobs(self, names=None):
        return [Job(name, f"/synthetic/{name}.wav") for name in (names or list(self.specs))]

    def decoder(self):
        return ReplayDecoder(build_table(self.table.items()))

    def open_recording(self, job):
        self.opened.append(job.job_name)
        if job.job_name in self.failing:
            raise InputError(self.failing[job.job_name])
        return ReplayRecording(job.job_name, self.specs[job.job_name].get("num_samples", SAMPLES_120S))


def results_of(table):
    return {key: DecodeResult(text=value, tokens=None, stop_reason=None, decode_seconds=0.0) for key, value in table.items()}
