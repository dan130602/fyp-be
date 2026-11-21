"""In-process probe of the no-repeat-ngram logits processor inside a running vLLM engine (Stage 3 follow-up D correction C1, follow-up E).

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python vllm_plugin_probe.py --settings-id <default-profile id> ...     (DIAGNOSTIC: the engine must run in this process to be inspected)
    python vllm_plugin_probe.py --settings-id <scheduling-profile id> --jobs jobs.json --call CALL_134 --start 24960000 --end 25440000 --out probe.json   (PROFILE: the decoder sets the variable)

For a default-profile settings id the variable must be set on this command line only: the probe then builds the decoder with `diagnostic_in_process=True` and stamps the report
`execution_kind: diagnostic-in-process` and `diagnostic_only: true` (it is NOT evidence about the default multiprocess execution). For a scheduling-profile id (an `execution` entry in its frozen table)
the command line must NOT set the variable: the decoder sets it itself, the probe is the production execution mode, and the report says `profile-in-process`. See `plan_probe`.

It reports, for the given settings id (stock or candidate): that `NoRepeatNGramV1LogitsProcessor` is among the engine's non-argmax-invariant logits processors, that output-token tracking is on, its n-gram
size, how often `apply` was called while the window decoded, whether the output ids it saw ever contained placeholders (< 0), and whether the last output list it saw equals the generated token ids
(a prefix of `token_ids`: the live list before the final token) of the request.
"""
import argparse
import json
import os
import sys
from pathlib import Path

LOG = {"calls": 0, "slots_seen": 0, "placeholder_calls": 0, "max_neg_inf_per_call": 0, "last_outputs": {}}


def plan_probe(settings_id, launcher):
    """-> (kind, error). Pure (no vLLM): which execution the probe uses for this settings id, or why it refuses."""
    from meralion_worker.decoders.frozen import execution_of
    if execution_of(settings_id) is None:
        if launcher != "0":
            return None, "refuses to run a default-profile probe without VLLM_ENABLE_V1_MULTIPROCESSING=0 on its own command line (a diagnostic mode, never an acceptance mode)"
        return "diagnostic-in-process", None
    if launcher is not None:
        return None, "refuses to run a scheduling-profile probe with VLLM_ENABLE_V1_MULTIPROCESSING set on its command line: the decoder sets it itself"
    return "profile-in-process", None


def install_and_inspect(worker):
    """Runs inside the engine's worker (in this process): inspects the logits processors and wraps `apply` of the n-gram one to count its calls and record the ids it sees."""
    import torch
    batch = worker.model_runner.input_batch
    processors = batch.logitsprocs
    names = [type(p).__name__ for p in processors.non_argmax_invariant]
    target = next((p for p in processors.non_argmax_invariant if type(p).__name__ == "NoRepeatNGramV1LogitsProcessor"), None)
    info = {"non_argmax_invariant": names, "argmax_invariant": [type(p).__name__ for p in processors.argmax_invariant], "plugin_present": target is not None,
            "logitsprocs_need_output_token_ids": bool(batch.logitsprocs_need_output_token_ids), "ngram_size": getattr(target, "ngram_size", None)}
    if target is not None and not hasattr(target, "_probe_original"):
        original = target.apply
        target._probe_original = original

        def wrapped(logits):
            LOG["calls"] += 1
            for slot, (_prompt, output) in list(target._slot_data.items()):
                LOG["slots_seen"] += 1
                snapshot = [int(t) for t in output]
                if any(t < 0 for t in snapshot):
                    LOG["placeholder_calls"] += 1
                LOG["last_outputs"][slot] = snapshot
            result = original(logits)
            LOG["max_neg_inf_per_call"] = max(LOG["max_neg_inf_per_call"], int(torch.isneginf(result).sum(dim=-1).max()))
            return result

        target.apply = wrapped
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings-id", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("MERALION_MODEL_DIR"))
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--call", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    kind, problem = plan_probe(args.settings_id, os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))
    if problem:
        print(f"vllm_plugin_probe: {problem}", file=sys.stderr)
        return 2
    from meralion_worker import audio
    from meralion_worker.decoders.vllm_decoder import VllmDecoder, build_requests
    path = next(job["input"] for job in json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"] if job["job_name"] == args.call)
    piece = audio.load_recording(args.call, path).get(args.start, args.end)
    decoder = VllmDecoder(args.model_dir, settings_id=args.settings_id, diagnostic_in_process=(kind == "diagnostic-in-process"))
    info = decoder.backend.llm.collective_rpc(install_and_inspect)[0]
    outputs = decoder.backend.llm.generate(build_requests(decoder.backend._prompt, [piece]), decoder.sampling_params)
    completion = outputs[0].outputs[0]
    generated = [int(t) for t in completion.token_ids]
    seen = LOG["last_outputs"].get(0, [])
    live = {"apply_calls": LOG["calls"], "placeholder_calls": LOG["placeholder_calls"], "max_neg_inf_in_one_call": LOG["max_neg_inf_per_call"], "generated_tokens": len(generated), "finish_reason": completion.finish_reason,
            "last_seen_output_length": len(seen), "last_seen_equals_generated_prefix": bool(seen) and seen == generated[:len(seen)], "no_placeholder_ids_seen": LOG["placeholder_calls"] == 0}
    checks = {"plugin_loaded": info["plugin_present"], "output_token_tracking_on": info["logitsprocs_need_output_token_ids"], "ngram_size_is_6": info["ngram_size"] == 6,
              "apply_called_for_every_generated_token": LOG["calls"] >= len(generated) - 1 and LOG["calls"] > 0, "sees_real_generated_ids": live["last_seen_equals_generated_prefix"] and live["no_placeholder_ids_seen"]}
    report = {"probe": "vllm_plugin_probe", "diagnostic_only": kind == "diagnostic-in-process", "execution_kind": kind, "execution": dict(decoder.execution),
              "mode": {"VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}, "settings_id": decoder.settings_id,
              "window": {"call": args.call, "start": args.start, "end": args.end}, "engine": info, "live": live, "checks": checks, "passed": all(checks.values()),
              "text_sha256_hint": completion.text[:80], "identity": decoder.identity(), "sampling": decoder.describe()["sampling"], "worker_git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT")}
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"settings_id": decoder.settings_id, "checks": checks, "passed": report["passed"], "apply_calls": LOG["calls"], "generated": len(generated)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
