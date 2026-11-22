"""Controlled test of the package's REAL no-repeat-n-gram logits processor (Stage 3 follow-up D, correction C1). Run it in the vLLM environment; it needs torch (CPU is enough) and no GPU.

    python plugin_logits_test.py --out plugin_logits_test.json

Why this exists: counting repeated n-grams in the saved outputs does NOT prove the ban works (a repeated 5-gram does not imply a repeated 6-gram, and re-tokenised text is not the generated ids). This test
drives the real `NoRepeatNGramV1LogitsProcessor` with a KNOWN token history and asserts the logit of the one token that would complete a repeated 6-gram becomes -inf and nothing else changes.

Cases: ban with the exact history [10,11,12,13,14,15,99,10,11,12,13,14] (token 15 must be banned); an uncompleted prefix, a history shorter than n and a `-1` placeholder history must NOT ban 15 (the last one is the
failure mode the package's InputBatch patch exists for); the live-reference property (appending to the output list after update_state is seen by the next apply); slot swap and removal; n = 3 from the
environment; the default n of 6 when the variable is unset; `is_argmax_invariant()` is False. Exit code 0 only if every case holds.

REV 3.2 Amendment 6: this is pure CPU logic and never runs inside a GPU allocation. Importing the package imports vLLM, which resolves its platform at IMPORT time (vllm/platforms/__init__.py `__getattr__`,
reached through vllm/utils/mem_utils.py), and with a MIG or GPU UUID in CUDA_VISIBLE_DEVICES that resolution is exactly what failed on the Blackwell slice. So the tool REFUSES (exit 3, before any import and
without writing a report) when a device UUID is visible, and the report carries an `evidence` block: the worker commit read from CODE_COMMIT, the hash of this file, the hashes of the two package files it
exercises, the versions it ran with and the (empty) device variable, so tools/decision.py can bind it to the evaluated runs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REFUSAL = "refused: run on a host where no device UUID is visible (login node or container with numeric or unset devices)"
# REV 4.1 A1: the cases are a STATIC list. run_cases() must produce exactly these names and the report's `passed` is DERIVED from them (all True); tools/decision.py holds its own copy of the required
# names (REQUIRED_PLUGIN_CASES), so a tool that reported fewer cases could not define its own pass, and a test asserts that the two lists are equal.
CASE_NAMES = (
    "bans_the_token_that_would_complete_a_repeated_6gram",
    "every_other_logit_is_unchanged",
    "is_argmax_invariant_is_false",
    "ngram_size_is_6",
    "no_ban_when_the_prefix_is_not_the_start_of_an_earlier_6gram",
    "no_ban_when_the_history_is_shorter_than_n",
    "placeholder_history_does_not_ban_the_real_token_15",
    "the_output_list_is_a_live_reference",
    "a_slot_swap_moves_the_history_with_it",
    "a_removed_slot_is_forgotten",
    "ngram_size_3_from_the_environment_bans_7",
    "the_default_ngram_size_is_6_when_unset",
)
EXERCISED_FILES = ("_vllm_plugin/__init__.py", "_vllm_plugin/transformers_utils/no_repeat_logits_processor.py")
CODE_ROOT = Path(__file__).resolve().parents[2]                    # tools/origami/ -> the synced code folder, which holds CODE_COMMIT


def uuid_device_visible(env=None):
    """True when CUDA_VISIBLE_DEVICES holds a MIG- or GPU- UUID (any entry)."""
    env = os.environ if env is None else env
    return any(entry.strip().startswith(("MIG-", "GPU-")) for entry in (env.get("CUDA_VISIBLE_DEVICES") or "").split(","))


def worker_commit(code_root=CODE_ROOT, env=None):
    """The clean commit of the synced code (CODE_COMMIT, else MERALION_WORKER_GIT_COMMIT); None for a missing, unknown or dirty one."""
    env = os.environ if env is None else env
    marker = Path(code_root) / "CODE_COMMIT"
    value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else (env.get("MERALION_WORKER_GIT_COMMIT") or "").strip()
    return None if value in ("", "unknown") or value.endswith("-dirty") else value


def collect_evidence(commit, env=None):
    """What the report is bound by. Reads metadata and files only: nothing here imports vllm or torch (the platform class is added by the caller after the cases ran)."""
    from importlib import metadata
    env = os.environ if env is None else env
    spec = importlib.util.find_spec("meralion_3_asr")
    root = Path(list(spec.submodule_search_locations)[0]) if spec and spec.submodule_search_locations else None
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in EXERCISED_FILES if root is not None and (root / name).is_file()}

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {"worker_git_commit": commit, "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "package": {"files_sha256": files},
            "versions": {name: version(name) for name in ("vllm", "torch", "meralion-3-asr")}, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "uuid_device_visible": uuid_device_visible(env), "slurm_job_id": env.get("SLURM_JOB_ID")}


def build(ngram):
    os.environ["MERALION_NGRAM_SIZE"] = str(ngram)
    from meralion_3_asr._vllm_plugin.transformers_utils.no_repeat_logits_processor import NoRepeatNGramV1LogitsProcessor
    import torch
    return NoRepeatNGramV1LogitsProcessor(None, torch.device("cpu"), False)


def update(processor, added=(), removed=(), moved=()):
    processor.update_state(types.SimpleNamespace(removed=list(removed), added=list(added), moved=list(moved)))


def logits(rows=1, vocab=200):
    import torch
    return torch.zeros(rows, vocab)


def neg_inf_positions(row):
    import torch
    return [int(i) for i in torch.nonzero(torch.isneginf(row)).flatten()]


def run_cases():
    import torch
    results = {}
    history = [10, 11, 12, 13, 14, 15, 99, 10, 11, 12, 13, 14]

    processor = build(6)
    update(processor, added=[(0, None, [], list(history))])
    out = processor.apply(logits())
    results["bans_the_token_that_would_complete_a_repeated_6gram"] = neg_inf_positions(out[0]) == [15]
    results["every_other_logit_is_unchanged"] = bool((out[0][torch.arange(200) != 15] == 0).all())
    results["is_argmax_invariant_is_false"] = processor.is_argmax_invariant() is False
    results["ngram_size_is_6"] = processor.ngram_size == 6

    processor = build(6)
    update(processor, added=[(0, None, [], [10, 11, 12, 13, 14, 15, 99, 10, 11, 12, 13, 77])])
    results["no_ban_when_the_prefix_is_not_the_start_of_an_earlier_6gram"] = neg_inf_positions(processor.apply(logits())[0]) == []

    processor = build(6)
    update(processor, added=[(0, None, [], [10, 11, 12, 13, 14])])
    results["no_ban_when_the_history_is_shorter_than_n"] = neg_inf_positions(processor.apply(logits())[0]) == []

    processor = build(6)
    update(processor, added=[(0, None, [], [-1] * len(history))])
    banned = neg_inf_positions(processor.apply(logits())[0])
    results["placeholder_history_does_not_ban_the_real_token_15"] = 15 not in banned          # the silent-disable failure mode: ids that are not the generated ids

    processor = build(6)
    live = [10, 11, 12, 13, 14, 15, 99, 10, 11, 12, 13]
    update(processor, added=[(0, None, [], live)])
    before = neg_inf_positions(processor.apply(logits())[0])
    live.append(14)                                                                              # the scheduler appends to the SAME list object
    after = neg_inf_positions(processor.apply(logits())[0])
    results["the_output_list_is_a_live_reference"] = before == [] and after == [15]

    processor = build(6)
    update(processor, added=[(0, None, [], list(history)), (1, None, [], [1, 2, 3])])
    try:
        from vllm.v1.sample.logits_processor.interface import MoveDirectionality
        update(processor, moved=[(0, 1, MoveDirectionality.SWAP)])
        out = processor.apply(logits(rows=2))
        results["a_slot_swap_moves_the_history_with_it"] = neg_inf_positions(out[0]) == [] and neg_inf_positions(out[1]) == [15]
    except ImportError:
        results["a_slot_swap_moves_the_history_with_it"] = None                                 # no vLLM: not testable here (reported, not hidden)
    update(processor, removed=[1])
    results["a_removed_slot_is_forgotten"] = neg_inf_positions(processor.apply(logits(rows=2))[1]) == []

    processor = build(3)
    update(processor, added=[(0, None, [], [5, 6, 7, 9, 5, 6])])
    results["ngram_size_3_from_the_environment_bans_7"] = processor.ngram_size == 3 and neg_inf_positions(processor.apply(logits())[0]) == [7]

    os.environ.pop("MERALION_NGRAM_SIZE", None)
    from meralion_3_asr._vllm_plugin.transformers_utils.no_repeat_logits_processor import NoRepeatNGramV1LogitsProcessor
    results["the_default_ngram_size_is_6_when_unset"] = NoRepeatNGramV1LogitsProcessor(None, torch.device("cpu"), False).ngram_size == 6
    check_declared(results)
    return results


def check_declared(results):
    """The cases produced must be EXACTLY the declared CASE_NAMES, in order: a dropped or renamed case would otherwise silently shrink what the report claims."""
    if list(results) != list(CASE_NAMES):
        raise RuntimeError(f"the cases produced {list(results)} are not the declared CASE_NAMES {list(CASE_NAMES)}")
    return results


def summarise(results):
    """(passed, failed, untested) from the case results: passed ONLY when every declared case ran and is True (a case that could not be run is neither a pass nor a failure)."""
    failed = [name for name, ok in results.items() if ok is False]
    untested = [name for name, ok in results.items() if ok is None]
    passed = list(results) == list(CASE_NAMES) and all(ok is True for ok in results.values())
    return passed, failed, untested


def main(argv=None):
    if uuid_device_visible():                                      # FIRST statement: before argparse's work, before any torch, vllm or meralion_3_asr import, before a report is written
        print(REFUSAL, file=sys.stderr)
        return 3
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    commit = worker_commit()
    if commit is None:
        print("refused: no clean recorded worker commit (CODE_COMMIT missing, unknown or -dirty): the report could not be bound to the code under test", file=sys.stderr)
        return 3
    import importlib.metadata as metadata
    results = run_cases()
    passed, failed, untested = summarise(results)
    evidence = collect_evidence(commit)
    try:
        import vllm.platforms
        evidence["platform_class"] = type(vllm.platforms.current_platform).__name__
    except Exception:                                              # informational only
        evidence["platform_class"] = None
    report = {"test": "plugin_logits_test", "passed": passed, "failed": failed, "not_testable_here": untested, "cases": results,
              "versions": {name: (metadata.version(name) if name else None) for name in ("torch", "vllm", "meralion-3-asr")}, "evidence": evidence}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "failed": failed, "not_testable_here": untested}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
