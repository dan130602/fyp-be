"""Canonical ordering and hashing of a stage's accepted parsed response (shared by every LLM stage)."""

from fyp_evaluation.call_specs.compile import serialize_json, text_hash


def ordered(value, schema):
    """Re-emit `value` with object keys in the response schema's property order (deterministic bytes)."""
    if isinstance(value, dict):
        if "anyOf" in schema:
            branch = next((b for b in schema["anyOf"] if b.get("type") == "object"), None)
            return ordered(value, branch) if branch else value
        properties = schema.get("properties", {})
        return {key: ordered(value[key], properties.get(key, {})) for key in properties if key in value} | {
            key: value[key] for key in value if key not in properties}
    if isinstance(value, list):
        return [ordered(item, schema.get("items", {})) for item in value]
    return value


def canonical_response(parsed, response_schema):
    return ordered(parsed, response_schema)


def response_hash(parsed, response_schema):
    return text_hash(serialize_json(canonical_response(parsed, response_schema)))
