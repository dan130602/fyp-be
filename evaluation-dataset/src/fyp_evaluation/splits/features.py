"""Stratification features, read ONLY from the frozen, committed G2A call specs (never from dialogues or verifier outcomes).

Verifier verdicts are not labels: gold compliance labels are PASS/FAIL only and C01-C12 are the only questions that ever FAIL.
Product, currency, direction and order type are the frozen gold values of T01, T02, T05 and T06.
"""

import csv
import json
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.planning.outputs import canonical_json, sha256

from . import SplitError

SPEC_DIR = "authoring/call_specs"
FINGERPRINTED = ("authoring/call_manifest.csv", "authoring/edge_case_assignments.csv", "authoring/planning_record.json",
                 "authoring/call_spec_record.json")
LANGUAGES = ("ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH")
COMPLIANCE = tuple(f"C{n:02d}" for n in range(1, 13))
SECONDARY = {"T01": "prod", "T02": "cur", "T05": "dir", "T06": "ord"}

# Label groups and their weights in the balance objective (priority order of the plan: duration, edge cases, compliance, style,
# secondary transaction categories).
GROUP_WEIGHT = {"duration": 4.0, "edge": 2.0, "fail": 2.0, "style": 1.5, "secondary": 1.0}


def feature_definitions():
    return {
        "language": "metadata.language_profile (hard quota, not a balance label)",
        "duration": "metadata.duration_profile: label dur:LONG and lang_dur:<language>|LONG (SHORT is the complement)",
        "edge": "edge_cases[]: ec:<edge_case_id>, fam:<family>, ecv:<edge_case_id>|<language_variant> (variant != NONE), edge_free; "
                "each counted once per CALL (distinct calls), never per assignment",
        "fail": "gold C01-C12 expected_verdict: fail:<question> for each FAIL, allpass when none fails",
        "style": "metadata.mixing_style (only EN_ZH_MIXED calls): style:<mixing_style>",
        "secondary": "frozen gold T01 prod:, T02 cur:, T05 dir:, T06 ord:",
    }


def _read(path):
    return parse_json_text(Path(path).read_text(encoding="utf-8"))


def load_features(dataset_root):
    """{call_id: features} from authoring/call_specs/*.json, cross-checked against authoring/call_manifest.csv."""
    root = Path(dataset_root)
    specs = sorted((root / SPEC_DIR).glob("CALL_*.json"))
    if not specs:
        raise SplitError(f"no call specs under {root / SPEC_DIR}")
    with (root / "authoring/call_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = {row["call_id"]: row for row in csv.DictReader(handle)}
    features = {}
    for path in specs:
        spec = _read(path)
        call_id = spec["call_id"]
        meta = spec["metadata"]
        row = manifest.get(call_id)
        if row is None or row["language_profile"] != meta["language_profile"] or row["duration_profile"] != meta["duration_profile"]:
            raise SplitError(f"{call_id}: call spec and call_manifest.csv disagree on language or duration")
        gold = {g["question_id"]: g["expected"] for g in spec["gold"]}
        edge = spec["edge_cases"]
        features[call_id] = {
            "call_id": call_id,
            "language": meta["language_profile"],
            "duration": meta["duration_profile"],
            "mixing_style": meta["mixing_style"],
            "edge_case_ids": sorted({e["edge_case_id"] for e in edge}),
            "families": sorted({e["family"] for e in edge}),
            "variants": sorted({f"{e['edge_case_id']}|{e['language_variant']}" for e in edge if e["language_variant"] != "NONE"}),
            "failing": sorted(q for q in COMPLIANCE if gold[q]["expected_verdict"] == "FAIL"),
            "secondary": {SECONDARY[q]: str(gold[q]["expected_value"]) for q in SECONDARY},
            "assignments": len(edge),
        }
    if sorted(manifest) != sorted(features):
        raise SplitError("call_manifest.csv and call_specs list different calls")
    return features


def labels_of(feature):
    """{label: group} for one call (binary indicators; the language is a hard quota, not a label)."""
    out = {}
    if feature["duration"] == "LONG":
        out["dur:LONG"] = "duration"
        out[f"lang_dur:{feature['language']}|LONG"] = "duration"
    for e in feature["edge_case_ids"]:
        out[f"ec:{e}"] = "edge"
    for f in feature["families"]:
        out[f"fam:{f}"] = "edge"
    for v in feature["variants"]:
        out[f"ecv:{v}"] = "edge"
    if not feature["edge_case_ids"]:
        out["edge_free"] = "edge"
    for q in feature["failing"]:
        out[f"fail:{q}"] = "fail"
    if not feature["failing"]:
        out["allpass"] = "fail"
    if feature["language"] == "EN_ZH_MIXED":
        out[f"style:{feature['mixing_style']}"] = "style"
    for key, value in feature["secondary"].items():
        out[f"{key}:{value}"] = "secondary"
    return out


def fingerprints(dataset_root):
    """SHA-256 of every committed input the split depends on (CRLF-insensitive), plus one aggregate over all call specs."""
    root = Path(dataset_root)
    out = {name: sha256((root / name).read_bytes()) for name in FINGERPRINTED}
    per_call = {p.stem: text_hash(p.read_text(encoding="utf-8")) for p in sorted((root / SPEC_DIR).glob("CALL_*.json"))}
    out["call_specs_sha256"] = sha256(canonical_json(per_call).encode("utf-8"))
    out["call_spec_count"] = len(per_call)
    return out


def dump(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
