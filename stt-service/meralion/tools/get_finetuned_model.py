"""Download the fine-tuned ClearCall MERaLiON model (decoder LoRA + encoder LoRA + full adapter, "J5XE") and check it against the manifest committed in this repository.

    python tools/get_finetuned_model.py                      # download model/ from Hugging Face into models/clearcall-j5xe and verify all 14 files
    python tools/get_finetuned_model.py --source <folder>    # use an already downloaded copy (the folder that contains model/) instead of downloading

Needs `pip install huggingface_hub` only for the download. The weights are never part of Git; the manifest (SHA-256 of every file) and the runtime configuration are, so a model
that does not match is refused. Prints the command that runs the worker with it.
"""
import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "tanedric/meralion-3-3b-asr-clearcall-j5xe"
CONFIG = "configs/runtime/j5xe-ep2-vllm-rp105-sched0.json"
MANIFEST = "configs/models/meralion-3-3b-asr-sft-j5xe-ep2.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repository (default: %(default)s)")
    parser.add_argument("--revision", default=None, help="branch, tag or commit of the repository (default: main)")
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "models" / "clearcall-j5xe", help="where to put the download (default: models/clearcall-j5xe, git-ignored)")
    parser.add_argument("--source", type=Path, help="an existing local copy (the folder that contains model/): skip the download")
    args = parser.parse_args(argv)

    for needed in (PACKAGE_ROOT / CONFIG, PACKAGE_ROOT / MANIFEST):
        if not needed.is_file():
            sys.exit(f"{needed} is missing: this script must run from a checkout that contains the J5XE manifest and runtime configuration")

    if args.source:
        folder = args.source
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("pip install huggingface_hub   (needed for the download)")
        print(f"downloading {args.repo} (model/, about 6.6 GB) to {args.out} ...", flush=True)
        folder = Path(snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(args.out), allow_patterns=["model/*", "SHA256SUMS.txt", "NOTICE", "README.md", "MERaLiON-3-Public-Licence.pdf"]))
    model_dir = folder / "model"
    if not model_dir.is_dir():
        sys.exit(f"{model_dir} not found")

    print("verifying every file against the manifest ...", flush=True)
    check = subprocess.run([sys.executable, "-m", "meralion_worker", "check-model", "--model-dir", str(model_dir), "--config", str(PACKAGE_ROOT / CONFIG)], cwd=PACKAGE_ROOT)
    if check.returncode == 3:                                    # the worker's exit code for a configuration / model failure
        sys.exit("the model directory does NOT match the manifest: do not use it")
    if check.returncode != 0:
        sys.exit("the check could not run (see the error above). Install the worker requirements first (pip install numpy soundfile boto3, or the full pinned environment: "
                 "requirements-vllm.lock, see README.md), then run this script again with --source <folder that contains model/> to verify the download.")
    print("\nOK. The licence (MERaLiON-3-Public-Licence.pdf) and NOTICE next to model/ apply to these weights.")
    print("Run the worker with it (CUDA GPU and the pinned vLLM environment required):\n")
    print(f'  cd "{PACKAGE_ROOT}"\n  python -m meralion_worker run --invocation-id my-first-run --jobs jobs.json \\\n'
          f'      --model-dir "{model_dir}" --config "{PACKAGE_ROOT / CONFIG}" \\\n      --output-dir out --state-dir state --summary-output summary.json')


if __name__ == "__main__":
    main()
