"""G6 command line: the shared stage CLI bound to `DialogueRunner`.

Identical modes, modifiers, guards and exit codes to G3/G4. G6 supports Batch (bulk generation: `--submit`, `--collect`) and, for pilots,
one synchronous request per call (`--sync --call-id X`); both use the same request builder, validation and acceptance. A synchronous
request whose outcome is unknown (timeout) is recorded as possibly billed and blocks the call until `--sync --acknowledge-duplicate-risk`.
Offline modes never need credentials.

One G6-only option: `--repair-feedback FILE` (with `--dry-run` or `--submit`) adds call-specific repair guidance, derived from the
call's current INVALID G7 report, to that call's request (see `request.load_repair_feedback`) and records its source on the attempt.
"""

import argparse
import sys
from pathlib import Path

from fyp_evaluation.call_configs.cli import main as stage_main

from .request import load_repair_feedback
from .run import DialogueRunner


def main(argv=None, **kwargs):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--repair-feedback", type=Path)
    pre.add_argument("--dataset-root", type=Path)
    known, rest = pre.parse_known_args(argv)
    if known.repair_feedback is not None:
        if known.dataset_root is None:
            print("--repair-feedback needs --dataset-root", file=sys.stderr)
            return 2
        if not any(mode in rest for mode in ("--submit", "--sync", "--dry-run")):
            print("--repair-feedback applies to --submit, --sync or --dry-run", file=sys.stderr)
            return 2
        try:
            feedback = load_repair_feedback(known.repair_feedback, known.dataset_root)
        except (OSError, ValueError, KeyError) as error:
            print(f"repair feedback: {error}", file=sys.stderr)
            return 2
        argv = ["--dataset-root", str(known.dataset_root), *rest]
        kwargs.setdefault("runner_class", lambda root, **kw: DialogueRunner(root, repair_feedback=feedback, **kw))
    kwargs.setdefault("runner_class", DialogueRunner)
    return stage_main(argv, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
