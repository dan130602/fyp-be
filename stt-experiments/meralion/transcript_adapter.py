"""Preserve numeric symbols before the unchanged evaluator normalizes text."""
import re

def transcript_document(transcript):
    # Keep numeric signs, currency prefixes and percentage suffixes attached so
    # the repository normalizer receives the complete written amount.
    tokens = re.findall(
        r"(?:[A-Za-z]*[$\u00a3\u20ac\u00a5\u20b9])?[+-]?\w+(?:['’.,]\w+)*%?|[^\w\s]",
        transcript,
    )
    return {"results": {
        "transcripts": [{"transcript": transcript}],
        "items": [
            {"type": "pronunciation" if any(char.isalnum() for char in token)
             or token in "$%+-\u00a3\u20ac\u00a5\u20b9" else "punctuation",
             "alternatives": [{"content": token}]}
            for token in tokens
        ],
    }}
