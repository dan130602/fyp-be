"""The G4 response validator: schema, every deterministic authoring rule, rendering, derived constraints and projection.

`errors()` returns [] only when the response can be compiled into a plan that projects back to the frozen skeleton and
satisfies the artifact schema. Acceptance therefore means "G4 structurally valid"; semantic adequacy is G5's concern.
"""

from jsonschema import Draft202012Validator

from .analysis import analyze
from .compile import assemble, get_lexicon, projection_errors

_DUMMY_GENERATION = {"attempt_id": "A001", "model": "validator", "prompt_template_sha256": "0" * 64,
                     "response_schema_sha256": "0" * 64, "accepted_response_sha256": "0" * 64}


class PlanResponseValidator:
    def __init__(self, response_schema, artifact_schema, inputs):
        Draft202012Validator.check_schema(response_schema)
        Draft202012Validator.check_schema(artifact_schema)
        self._response = Draft202012Validator(response_schema)
        self._artifact = Draft202012Validator(artifact_schema)
        self._inputs = inputs

    def errors(self, parsed, ctx):
        schema_errors = [f"schema: {e.json_path}: {e.message}" for e in self._response.iter_errors(parsed)]
        if schema_errors:
            return schema_errors
        authoring, derived, errors = analyze(ctx, get_lexicon(self._inputs), parsed)
        if errors:
            return errors
        plan = assemble(ctx, authoring, derived, _DUMMY_GENERATION)
        errors = projection_errors(ctx, plan)
        errors += [f"artifact schema: {e.json_path}: {e.message}" for e in self._artifact.iter_errors(plan)]
        return errors
