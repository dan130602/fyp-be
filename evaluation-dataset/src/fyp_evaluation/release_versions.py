"""Contract-generation helpers shared by the staged runners (G3-G7).

A release is either schema 1.0 (the historical pilot) or multilingual schema 2.0, decided by its generation plan
(`validation.stage0.is_multilingual`). Schema-2.0 stages use *new* schema, prompt and response-format files named
`<name>.v2.*`, and the historical files are never touched, so the pilot's bytes, request hashes and reconstruction
stay exactly as they were.
"""

from fyp_evaluation.validation.stage0 import is_multilingual

HISTORICAL = "1.0"
MULTILINGUAL = "2.0"


def generation(documents):
    """The artifact/record schema version of a release: "2.0" for a multilingual release, else "1.0"."""
    return MULTILINGUAL if is_multilingual(documents) else HISTORICAL


def versioned(stem, documents):
    """`stem` for the historical release, `stem.v2` for a multilingual one (schema and response-schema file stems)."""
    return f"{stem}.v2" if is_multilingual(documents) else stem


def versioned_prompt(template, documents):
    """`call_config.v1.md` -> `call_config.v2.md` for a multilingual release; the historical name otherwise."""
    return template.replace(".v1.md", ".v2.md") if is_multilingual(documents) else template


def versioned_format(name, documents):
    """`g3_call_config_v1` -> `g3_call_config_v2` for a multilingual release."""
    return name.removesuffix("_v1") + "_v2" if is_multilingual(documents) else name
