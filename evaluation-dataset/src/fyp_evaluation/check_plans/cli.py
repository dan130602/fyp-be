"""G4 command line: the shared stage CLI bound to `CheckPlanRunner`.

Identical modes, modifiers, guards and exit codes to G3 (0 success, 1 invalid/differing outputs or a recorded failed
outcome, 2 input/usage/precondition/credentials/fatal-transport errors). Offline modes never need credentials.
"""

from fyp_evaluation.call_configs.cli import main as stage_main

from .run import CheckPlanRunner


def main(argv=None, **kwargs):
    kwargs.setdefault("runner_class", CheckPlanRunner)
    return stage_main(argv, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
