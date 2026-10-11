"""G6 command line: the shared stage CLI bound to `DialogueRunner`.

Identical modes, modifiers, guards and exit codes to G3/G4. G6 is Batch-only: `--sync` exits 2 without sending anything.
Offline modes never need credentials.
"""

from fyp_evaluation.call_configs.cli import main as stage_main

from .run import DialogueRunner


def main(argv=None, **kwargs):
    kwargs.setdefault("runner_class", DialogueRunner)
    return stage_main(argv, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
