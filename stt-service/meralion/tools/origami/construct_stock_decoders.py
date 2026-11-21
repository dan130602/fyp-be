"""Construct a STOCK decoder once on the cluster and compare its identity with the pinned values (Stage 3 follow-up D, correction C2). Run once per backend in its own environment.

    python construct_stock_decoders.py --backend transformers|vllm --out result.json

Builds the real decoder (Transformers needs a GPU and the Transformers environment; vLLM starts an engine), checks that its settings id and frozen digest equal the values pinned before the candidate was added,
and records `describe()["settings_id"]`, the digest and the runtime identity. Exit code 0 only if both match.
"""
import argparse
import json
import os
import sys

PINNED = {"transformers": ("stock-transformers-v1", "511903ac4249ad1873ce95b209e907e0b89a94bbd4ab393e37313313ff1fb01d"),
          "vllm": ("stock-vllm-package-v1", "e526468f938c33b955799b80a3eeae437ea1e09ee19e3286971c72d67b0941d2")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=sorted(PINNED), required=True)
    parser.add_argument("--model-dir", default=os.environ.get("MERALION_MODEL_DIR"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.backend == "transformers":
        from meralion_worker.decoders.transformers_decoder import TransformersDecoder
        decoder = TransformersDecoder(args.model_dir)
    else:
        from meralion_worker.decoders.vllm_decoder import VllmDecoder
        decoder = VllmDecoder(args.model_dir)                                           # the default settings id: the stock package settings
    description = decoder.describe()
    expected_id, expected_digest = PINNED[args.backend]
    checks = {"settings_id_is_the_stock_id": description["settings_id"] == expected_id, "frozen_digest_is_the_pinned_stock_digest": description["frozen_digest"] == expected_digest,
              "identity_carries_the_runtime": "runtime" in decoder.identity()}
    report = {"tool": "construct_stock_decoders", "backend": args.backend, "settings_id": description["settings_id"], "frozen_digest": description["frozen_digest"], "checks": checks, "passed": all(checks.values()),
              "identity": decoder.identity(), "worker_git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT")}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=str)
        handle.write("\n")
    print(json.dumps({"backend": args.backend, "checks": checks, "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
