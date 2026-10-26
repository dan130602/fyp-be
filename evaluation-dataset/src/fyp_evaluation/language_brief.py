"""The language block every multilingual stage puts in its per-call brief, derived from the frozen call spec only.

The call spec carries `language_requirements` (script rules, each speaker's spoken and required languages, the mixing
style). Prompts and verifiers read the same block, so authors and verifiers apply identical requirements.
"""


def language_brief(spec):
    """{"language_profile", "mixing_style", "script_rules", "speakers": {speaker: {spoken_languages, required_languages}}}."""
    requirements = spec["language_requirements"]
    return {
        "language_profile": spec["metadata"]["language_profile"], "mixing_style": spec["metadata"]["mixing_style"],
        "script_rules": requirements["script_rules"],
        "speakers": {speaker: {"spoken_languages": rule["spoken_languages"], "required_languages": rule["required_languages"]}
                     for speaker, rule in requirements["speakers"].items()},
    }
