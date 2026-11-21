"""DIAGNOSTIC audit of repeated token n-grams in worker outputs (Stage 3 follow-up D, correction C1). It is evidence for a person to read, NEVER an acceptance gate.

    python tools/ngram_audit.py --docs <transcripts folder> --tokenizer <model dir>/tokenizer.json --out audit.json

What it does and does not show: it re-tokenises each decoded text with the model's tokenizer and counts, per window, whether any token n-gram (n = 3 to 6) occurs twice. That is NOT the generated token
sequence (cleaning, tokenizer round trips), and a repeated 5-gram does not imply a repeated 6-gram (`[1,2,3,4,5,6,1,2,3,4,5,7]` repeats a 5-gram but no 6-gram), so a count of zero repeated 6-grams
neither proves that the no-repeat-ngram processor worked nor is its absence a failure. The functional proof of the processor is `tools/origami/plugin_logits_test.py` plus the live probe. A candidate is
never required to contain repeated 5-grams. Exit code 0 always when the audit could be made (the file is only information); 2 when the tokenizer library or file is missing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance                                                     # noqa: E402

SIZES = (3, 4, 5, 6)


def has_repeat(ids, n):
    seen = set()
    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i:i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def audit(texts, encode):
    """texts: {label: text}. -> {"windows": n, "windows_with_a_repeated_ngram": {n: count}, "flagged_6": [labels]}."""
    counts = {size: 0 for size in SIZES}
    flagged = []
    for label, text in texts.items():
        ids = encode(text)
        for size in SIZES:
            if has_repeat(ids, size):
                counts[size] += 1
                if size == 6:
                    flagged.append(label)
    return {"windows": len(texts), "windows_with_a_repeated_ngram": counts, "windows_with_a_repeated_6gram": flagged}


def decode_texts(docs):
    texts = {}
    for name, document in sorted(docs.items()):
        block = document["meralion"]
        for window in block["trace"]["windows"]:
            texts[f"{name}:window:{window['start_sample']}-{window['end_sample']}"] = window["text"]
        for decode in block["retries"]["decodes"]:
            texts[f"{name}:retry:{decode['window'][0]}-{decode['window'][1]}"] = decode["result"]["text"]
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    except (ImportError, OSError, Exception) as error:                      # noqa: BLE001  any failure to load the tokenizer means the audit cannot be made
        print(f"ngram_audit: cannot load the tokenizer: {error}", file=sys.stderr)
        return 2
    docs, problems = acceptance.scan_documents(args.docs)
    result = audit(decode_texts(docs), lambda text: tokenizer.encode(text, add_special_tokens=False).ids)
    result.update(diagnostic_only=True, caveat="re-tokenised cleaned text, not the generated ids; a repeated 5-gram does not imply a repeated 6-gram; not an acceptance gate", document_problems=problems,
                  evidence=acceptance.evidence_of(docs) if docs else {})
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("windows", "windows_with_a_repeated_ngram")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
