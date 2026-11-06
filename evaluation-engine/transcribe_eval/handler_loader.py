"""Load the production Lambda handler unchanged, so the evaluation starts jobs exactly the way production does.

The file is loaded under a unique module name because the Lambda's package is simply called `app`, which would collide with other packages.
The handler creates its boto3 client when imported, so the region (and credentials) must already be in the environment.
"""

import importlib.util
import os
import sys

MODULE_NAME = "fyp_transcribe_trigger_handler"


def load_handler(path):
    """The handler module from `path`; the file is executed, never copied or edited."""
    if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
        raise RuntimeError("set AWS_DEFAULT_REGION before loading the handler (Settings.apply_to_environ does this)")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load the Lambda handler from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
