# FYP evaluation dataset foundations

Independent evaluation contracts for **AI-Powered Regulatory Compliance
Automation for Call Monitoring**. This project lives in the existing `fyp-be`
repository. It does not import or change the production backend or `fyp-ai`.

**Status: checklist updated as requested; five-call coverage matrix awaiting
user review.** The package and call-answer contracts are now `0.4.0`;
normalization remains `0.3.0` with the approved DOB format unchanged.
No call configurations, gold check plans, dialogues, canonical ground truth or
audio have been generated. Tests cover artificial contract fixtures, the actual
checklist and the fixed pilot CSV. Matrix values are predetermined and must not
be reinterpreted from later dialogue.

## Layout

| Location | Purpose |
| --- | --- |
| `schemas/` | Six core Draft 2020-12 data schemas plus reusable normalization rules. References resolve locally; `.invalid` schema IDs are identifiers, not network endpoints. |
| `authoring/coverage_matrix.csv` | Ten question rows and five call columns of predetermined answers, awaiting approval before downstream authoring. |
| `authoring/call_configs/` | Future per-call scenario configuration and matrix-derived answers. |
| `authoring/check_plans/` | Future predetermined gold checks, created before dialogue. |
| `authoring/dialogues/` | Future generated spoken dialogue, without turn IDs. |
| `configs/normalization.json` | Reusable exact-comparison rules, a compliance rule reference and extraction field-to-rule references. No question-ID mappings or scoring implementation. |
| `configs/render.json` | Unset Polly rendering template, containing no credentials. |
| `src/fyp_evaluation/validation/` | Read-only schema/integrity checks, checklist/normalization and answer compatibility validation, and a pure LangGraph checklist projection. |
| `src/fyp_evaluation/authoring/` | Narrow K01 CSV-cell decoding; deterministic compilation and general CSV/cross-file validation remain future work. |
| `src/fyp_evaluation/rendering/` | Reserved for SSML, Polly, stitching and timings. |
| `src/fyp_evaluation/scoring/` | Reserved for comparing pipeline predictions against gold. |
| `scripts/` | Reserved for thin command-line workflow wrappers. |
| `tests/` | Artificial contract fixtures and regression tests, plus checks of the actual pilot checklist and five-call matrix. No generated calls or dialogue. |
| `evaluation_dataset/checklist.json` | The authoritative internal checklist: `id`, `type`, `text`, plus `requirement_type` for compliance or `field` for extraction. |
| `evaluation_dataset/calls/` | Canonical calls and separate render outputs. Five empty pilot call directories exist. |
| `evaluation_runs/` | Future predictions and evaluation results, kept separate from gold. |

Final artifacts will retain this layout (listed output files do **not** exist yet):

```text
evaluation_dataset/
├── checklist.json
└── calls/
    ├── CALL_001/
    │   ├── ground_truth.json
    │   └── renders/
    │       └── clean_voice_A/
    │           ├── audio.mp3
    │           ├── timings.json
    │           └── segments/
    │               ├── 0.mp3
    │               └── ...
    ├── CALL_002/
    ├── CALL_003/
    ├── CALL_004/
    └── CALL_005/
```

The schema does not hard-code five calls, ten questions, or a maximum recording
duration. Additional calls and renders use the same contracts.

## Required generation order

1. Create and approve the fixed internal checklist. Establish reusable
   field-normalization rules before filling matrix cells.
2. Create and approve `coverage_matrix.csv`, covering every question and call.
3. Expand each call's matrix column into a coherent call configuration. Copy
   the predetermined answers unchanged into `intended_answers`.
4. Create the gold check plan **before** generating dialogue. Its expected
   answers must equal the matrix and configuration answers.
5. Generate employee/client dialogue around the plan, using ChatGPT Work
   separately during the pilot. Include planned presence phrases verbatim.
6. Independently verify the dialogue against every predetermined check,
   including speaker, absence and content constraints. Revise nonconforming
   dialogue; never derive or adjust gold from the completed dialogue.
7. A future Python compiler validates all inputs, assigns turn IDs and resolves
   evidence deterministically, then writes canonical `ground_truth.json`.
8. Generate per-turn SSML from canonical spoken-form text.
9. Render Polly segments, stitch them, and write timings for that specific
   render. Canonical ground truth remains unchanged.
10. Run the actual FYP system and compare its predictions against canonical
    gold. Evaluation outputs belong in `evaluation_runs/`.

Only the validation foundations are implemented. No compiler, generation,
semantic verification, LLM API, audio pipeline, AWS calls, Transcribe,
LangGraph integration, or evaluation metrics are implemented. A structural
validation pass is neither approval nor independent semantic verification.

If a predetermined answer must change, explicitly revise and reapprove the
upstream checklist/matrix/plan and regenerate affected downstream artifacts.
Do not retrofit gold answers to existing dialogue. Approval history and the
independent verification record format remain to be designed.

## JSON contracts

Objects reject unknown fields, except the explicitly keyed rule and field
maps. The call-config, check-plan and ground-truth envelopes use
`schema_version: "0.4.0"` for the K01-only nullable identification contract.
Normalization remains `0.3.0`. The package version is `0.4.0`. The checklist schema shape remains
revision `0.2.0`; dialogue and timing envelopes remain `0.1.0`. Call IDs match
`CALL_` followed by at least three digits. Question/evidence/fact/rule/render
identifiers start with a letter and contain letters, digits, `_` or `-`.
The internal checklist remains an array without an envelope. It is projected
to the legacy LangGraph request shape; it must not be passed there directly.

| Schema | Shape and key rules |
| --- | --- |
| `checklist` | Nonempty array with unique IDs. COMPLIANCE entries contain exactly `id`, `type`, `requirement_type` and `text`. CLIENT_KYC/TRANSACTION entries contain exactly `id`, `type`, `field` and `text`. |
| `call_config` | Version, call ID, `metadata` (`language`, `scenario`), positive ordered target duration bounds in seconds, named `scenario_facts`, and nonempty `intended_answers`. Each answer contains `question_id` and a typed `expected` object. |
| `check_plan` | Version, call ID and nonempty `checks`. Each has `question_id`, typed `expected`, and `evidence_plan`. No resolved turn IDs or render settings. |
| `dialogue` | Version, call ID and nonempty `turns`. Each contains `speaker`, spoken-form `text`, and `critical_tokens` (possibly empty). No turn IDs. Speakers are `EMPLOYEE` or `CLIENT`. |
| `ground_truth` | Version, call ID, metadata, checks preserving their expected answers and evidence plans, and `dialogue` with Python-assigned `turn_id`. Each check adds `gold_evidence_turns` and `gold_evidence_text`. |
| `timings` | Version, call ID, `render_id`, `audio_file: "audio.mp3"`, total `duration_seconds`, and a `turns` array. Each has integer `turn_id`, start/end seconds and optional `words` array. Word entries contain text and start/end seconds. All times are offsets from the start of stitched audio. |
| `normalization` | Version, named `rules`, `compliance_rule` and `field_rules`. Each rule declares its scalar `value_type` and `operation: "exact"`. Rules describe value representation, not question classification. |

COMPLIANCE requires `requirement_type: "OBLIGATION"` or `"PROHIBITION"` and
forbids `field`. CLIENT_KYC and TRANSACTION require `field` and forbid
`requirement_type`. Checklist text must be nonblank. Fields follow the same
identifier syntax as IDs. Gold answers, evidence, source passages, turn IDs,
call-specific fields, `answer_type` and `depends_on` are forbidden.

The only LangGraph adapter is a deterministic, in-memory projection:

```python
langgraph_checklist = [
    {"id": item["id"], "question": item["text"]}
    for item in checklist
]
```

`to_langgraph_checklist(checklist, schema_dir)` validates the internal shape
before applying this projection. It preserves IDs, order and exact text,
does not mutate its input, and never writes another checklist file or calls
LangGraph. No production workflow changes are required.

Expected answers use a tagged union:

```text
Compliance: {kind: "compliance", expected_verdict: "PASS" | "FAIL"}
Extraction: {kind: "extraction", value_type: TYPE, expected_value: VALUE}
TYPE: text | integer | number | boolean | dob_dd_month_yyyy
```

Checklist `type` determines the answer kind: COMPLIANCE means a PASS/FAIL
verdict; CLIENT_KYC and TRANSACTION mean actual extraction values. The
extraction `field` selects a reusable normalization rule, which determines
the scalar `value_type`. Existing `kind`/`value_type` tags in call documents
describe their serialized values; they are derived from these contracts,
not maintained in a second per-question classification file.

Gold extraction checks carry the actual value, not a PASS/FAIL verdict.
K01 alone also permits a native JSON null for an explicitly predetermined
absent identification number; all other expected-value branches stay non-null.
A future scorer must produce PASS or FAIL by
comparing the predicted value to that gold value under the selected rule.
`N/A` and `UNCLEAR` are not valid gold compliance verdicts. Unexpected predictions
with those labels must be recorded separately and must never rewrite gold;
their reporting/scoring policy is still pending.

Canonical DOB is a `DD Month YYYY` **string**, with exactly two day digits,
a full title-case English month name, single spaces and four year digits.
The JSON Schema checks shape; Python additionally checks Gregorian calendar
validity using a fixed English month table independent of operating-system locale.
The old compact, slash-separated and numeric representations are rejected. Text stays case-sensitive and untrimmed under the draft exact rules.
Numbers and booleans remain typed values; numeric tolerance, aliases, whitespace
normalization, and spoken-to-structured conversion are not implemented.

`configs/normalization.json` retains all six exact rules. The old rule property
`answer_type` is replaced by `value_type`; the compliance rule's value type is
`verdict`. `compliance_rule` selects that rule. `field_rules` maps extraction
field names to rule IDs, shared across questions and categories. The initial
canonical DOB field identifier is `dob`, mapped to `exact_dob_dd_month_yyyy`.
The pilot maps `client_identification_number`, `transaction_instrument` and
`transaction_currency` to the existing `exact_text` rule. The obsolete
`client_name` mapping was removed because K01 now extracts identification.
The DOB rule and expected-value tag were replaced with
`exact_dob_dd_month_yyyy` and `dob_dd_month_yyyy`; this includes new format and
calendar checks, not just a name change. Numeric and boolean rules remain
available but are unused by these four extraction fields.
No field-name guessing or implicit text fallback is
performed: an unmapped extraction field fails checklist validation. Add each
approved field once, using the existing integer/number/text/boolean rules as
appropriate. Unused reusable rules and field mappings are allowed.

Python checks that rule references exist, that compliance selects a verdict
rule, that extraction fields cannot select verdict rules, and that `dob`
selects a `DD Month YYYY` rule. `validate_answers` additionally checks exact checklist
ID coverage and compatible expected-answer kinds/value types in each call
config, plan or canonical ground truth. Individual JSON Schema validation
cannot check these cross-file relationships by itself.

Keep three representations separate:

| Stage | DOB representation |
| --- | --- |
| Predetermined matrix and canonical expected value | Calendar-valid `DD Month YYYY` string, e.g. `13 May 2001` |
| Future planned spoken evidence and dialogue | Natural spoken expression of that same date; authored only after the matrix is approved |
| Model prediction | Raw value actually returned by the answering workflow; retained separately from gold |

Future scoring must deterministically normalize equivalent predicted date
representations to `DD Month YYYY` before exact comparison. Raw string equality
alone would incorrectly reject equivalent dates. The current `operation: exact`
rule describes the final comparison, not an implemented prediction parser.
No general-purpose date parser, prediction normalization or AI-based conversion
is implemented here. The new helper validates canonical strings only; it never
coerces old formats or rewrites gold. A future parser must explicitly define
accepted forms and reject ambiguous/unparseable predictions rather than guess.

Turn IDs are contiguous integers `0..n-1` in dialogue order, never `T001` or
numeric strings. Evidence IDs label planned phrases; they are not turn IDs.
Evidence turn arrays are sorted and unique. Critical tokens must occur verbatim
in their turn's spoken text. Canonical objects cannot contain voice, prosody,
timestamps, render IDs or word timings.

## Evidence contract and future resolution

The user has approved exact planned evidence phrases for the initial contracts.

* Presence: `evidence_plan` has `mode: "presence"`, nonempty `planned_evidence`,
  and a `content_constraints` array. Each planned item has a locally unique
  `evidence_id`, `required_speaker` and exact `text`. No turn IDs are allowed.
* Absence: the plan has `mode: "absence"`, `scope: "entire_dialogue"`,
  `required_speaker` (`EMPLOYEE`, `CLIENT` or `ANY`), an `absent_event`
  description, and `content_constraints`. No positive quote is required.
  Either PASS or FAIL may be grounded in absence, depending on the question.
  This mode is also available for extraction checks. K01's absent value is
  native JSON null, with scope `entire_dialogue` and speaker `ANY`; canonical
  evidence is `gold_evidence_turns: []` and `gold_evidence_text: ""`.
  Other extraction fields still require concrete typed values.

For each presence phrase, the future compiler must search **only** eligible
speaker turns for a case-sensitive verbatim substring. Zero matches fails.
More than one eligible turn fails as ambiguous; never choose the first match.
Multiple planned phrases may resolve to the same turn. The check's
`gold_evidence_turns` is the sorted union of resolved turn IDs;
`gold_evidence_text` is the planned phrases joined by a single newline, in
plan order. These are exact excerpts, not newly inferred paraphrases. The
current validator checks these invariants on supplied canonical fixtures;
it does not assign IDs or compile output.

Absence checks have `gold_evidence_turns: []` and `gold_evidence_text: ""`.
An empty array records no positive quote; it does **not** prove absence.
Meaning, contradictions, absences and content constraints still require
independent verification before compilation. A structurally valid dialogue
can still contradict its evidence phrase.

If an intended phrase occurs in multiple turns, stop and report the candidates.
Keep the approved gold answer unchanged. A disambiguation protocol (or a
permitted dialogue revision) needs explicit agreement before compilation can
continue; this scaffold has no automatic disambiguation or semantic matching.

## Coverage CSV contract (future Python validator)

**JSON Schema does not validate this CSV.** The current fixed pilot is parsed
with Python's `csv` module and checked by `tests/test_coverage_matrix.py`.
Those tests are dataset-specific checks, not a reusable CSV validator or an
implementation of the entire cross-file contract below. The future Python
validator must enforce all of these rules:

* UTF-8 CSV, parsed using Python's `csv` module. First header is `question_id`;
  remaining headers are unique call IDs. The template lists the five pilot
  calls. More call columns may be added later. No duplicate or blank headers.
* Exactly one row per checklist ID and one nonempty answer cell per call.
  Reject duplicate/unknown/missing question IDs, ragged rows and blank cells.
  Row order should follow checklist order for review; identity is by ID.
* Read cells as strings first. CSV has no native cell types. Do not let
  spreadsheet type inference convert canonical dates or strip their day zeroes.
  Quoting a CSV cell alone does not prevent spreadsheet coercion.
* Look up each row's ID in the authoritative checklist, then use `type` and
  the extraction `field`'s rule from `configs/normalization.json`. Compliance cells
  contain exactly `PASS` or `FAIL`. Extraction cells contain actual values.
  Text and DOB cells contain raw strings, not JSON-quoted strings. Integer,
  number and boolean cells use finite JSON scalar syntax with the indicated
  type (booleans are lowercase `true`/`false`). Reject type mismatches; do not
  silently trim or coerce values. CSV quoting handles commas or newlines in text.
* DOB cells contain canonical `DD Month YYYY` strings and represent valid calendar dates.
  Extraction values must not be replaced by scoring labels. A literal text
  value that happens to spell a verdict remains text under its field rule.
* K01 alone reserves the exact CSV sentinel `NOT_STATED` for absent identification.
  `authoring.coverage.parse_k01_cell` converts it to Python `None` (JSON null).
  A populated K01 cell must match `[STFGM][0-9]{7}[A-Z]` in full and fail the
  checksum check. The decoder preserves it exactly, without trimming or repair.
  Empty cells, `null`, PASS/FAIL and N/A are rejected for K01. The sentinel must
  never be copied as a literal expected-value string into a call document.
* Require exact agreement among checklist IDs, matrix rows,
  config `intended_answers`, plan checks and canonical checks. Every referenced
  normalization rule must exist and support the derived value representation.
* Call columns must agree with configuration, plan, dialogue and ground-truth
  call IDs. Preserve typed matrix answers through config, plan and compilation;
  compare values **and** derived types, without deriving answers from dialogue.
* Check approved version consistency, complete question coverage per call,
  scenario fact coherence, and the independent verification record before
  permitting compilation. Do not treat schema validity as an approval record.

All selected pilot questions are applicable. No `depends_on`, applicability
gating, N/A gold answers, or complex missing-value logic exists. K01's null
represents absence of an extractable value, not inapplicability or compliance FAIL.

## Rendering boundary

`configs/render.json` intentionally has null voice, region, engine, sample rate,
prosody and silence settings. It is not runnable. Choose settings before any
render; never add credentials. Voice choices, speech rate, pitch, SSML and
render-specific settings belong outside canonical ground truth.

Python assigns canonical turn IDs after dialogue generation and before
rendering. Amazon Polly must never assign, renumber or change those IDs; it
only produces audio and render-specific timing information.

Future renders use `segments/<integer turn_id>.mp3`. Timings are arrays of
objects, not objects keyed by numeric-looking strings. The validator checks
ordered, nonoverlapping intervals within audio duration, and words within their
turn. A future cross-file validator must also verify that timings contain every
canonical turn exactly once, with matching call ID, and that actual media and
segment files agree. Current validation does not inspect media.

Generated audio and run outputs are ignored by Git. Gold, schemas and timing
JSON are not ignored. There are no fake output artifacts.

## Run validation and tests

From `fyp-be/evaluation-dataset`, using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fyp_evaluation.validation schemas
.\.venv\Scripts\python.exe -m pytest -q
```

Once authoring documents exist, validate JSON files, checklist/rule links and
downstream expected-answer compatibility with:

```powershell
fyp-validate json check_plan authoring/check_plans/CALL_001.json
fyp-validate checklist evaluation_dataset/checklist.json configs/normalization.json
fyp-validate answers check_plan authoring/check_plans/CALL_001.json evaluation_dataset/checklist.json configs/normalization.json
```

The checklist command validates the updated pilot checklist and its
four field mappings. This structural pass does not approve its policy content
or establish that a complete dataset exists. Empty checklists remain invalid.
The CLI returns 0 on success, 1 on invalid data, and 2 for input/usage errors.

Without installing, in an environment with the declared dependencies:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m fyp_evaluation.validation schemas
python -m pytest -q
```

The package can be installed/imported in another repository. Schema files are
explicit repository resources, not bundled in the wheel at this stage. Supply
their directory when using it elsewhere:

```python
from fyp_evaluation.validation import (
    validate, validate_checklist, validate_answers, to_langgraph_checklist,
)

errors = validate(document, "ground_truth", schema_dir=absolute_schema_directory)
errors = validate_checklist(checklist, normalization, absolute_schema_directory)
errors = validate_answers(document, "ground_truth", checklist, normalization, absolute_schema_directory)
langgraph_checklist = to_langgraph_checklist(checklist, absolute_schema_directory)
```

For CLI use outside this folder, pass `--schema-dir <absolute path>` before the
subcommand. All validation is read-only. Tests cover the six core shapes,
field-rule references, mixed checklist categories, legacy LangGraph projection,
malformed data, and the most important structural
invariants. They do not certify legal correctness, approval history, cross-file
answer preservation, CSV content, or semantic evidence.

## Migration from the initial scaffold

The earlier checklist schema and documentation required `{id, question}` as
the stored format. Classification and normalization selection lived in
`configs/question_metadata.json`. That duplicates the richer checklist's
classification, so the empty metadata file and its schema were removed.
There were no metadata entries to migrate. The checklist was empty at migration;
it was subsequently populated in the separately authorized checklist task.

The former `validate_metadata` API and `metadata` CLI command were replaced by
`validate_checklist` and the `checklist` command. Their old missing/extra-ID and
rule-compatibility tests were adapted to field-rule checks and exact downstream
answer-ID coverage. Core gold, evidence, DOB, turn-ID and timing tests remain.
New projection tests ensure LangGraph still receives only `id` and `question`.
Legacy internal `{id, question}` entries and old normalization configs now fail
validation explicitly; no silent migration or dual checklist maintenance exists.

At revision `0.2.0`, `call_config`, `check_plan` and `ground_truth` already shared
a compliance/extraction union, so their shapes needed no changes. Revision
`0.3.0` changes the DOB branch of that shared union and versions its consumers.
Mixed-category artificial
fixtures confirm that DOB strings, numeric values and text survive in each
shape. This establishes contract compatibility along matrix -> config ->
plan -> ground truth, not an implemented CSV parser or compiler. Comparing
actual predetermined values across those stages remains future compiler work.

## Pilot checklist interpretation and source review

The sole authoritative question text is in `evaluation_dataset/checklist.json`.
Its ordered IDs are C01-C06, K01-K02 and T01-T02: four compliance obligations,
two compliance prohibitions, two KYC extractions and two transaction extractions.
This section records proposed interpretation and sources, not per-call gold
answers or a predetermined PASS/FAIL distribution.

The six requirements are **provisional pilot policies informed by the sources
below**, not assertions that these exact conversational sentences are universal
statutory requirements. A recorded call alone cannot establish compliance with
the full regulatory framework. In particular, information papers describe
supervisory expectations/practices, while FAA-N16 has application exceptions.
Sponsor confirmation is required before freezing this checklist or its rubrics.

| ID | Behaviour | PASS under the proposed pilot policy | FAIL under the proposed pilot policy | Source |
| --- | --- | --- | --- | --- |
| C01 | Confirm investment objective | The employee explicitly establishes or confirms the client's investment objective in the conversation, with the client supplying or affirming it. | No such exchange, or only an unanswered question/unsupported assumption about the objective. | S1 sections 4.1, 4.2(a); S2 paragraph 11(a). |
| C02 | Confirm willingness to accept losses | The employee explicitly establishes or confirms the client's willingness to bear investment losses, with the client supplying or affirming it. | Loss tolerance is not established; asking about the investment objective alone is insufficient. | S1 sections 4.1, 4.2(c); S2 paragraph 11(b). |
| C03 | Explain capital-loss risk | The employee clearly explains that the client may lose invested capital. | Disclosure is absent, or the employee denies capital-loss risk. A vague statement that an investment is risky is insufficient. | S1 section 5.1, footnote 26(iv); S3 page 5, product-understanding/risks panel. Capital loss is the selected pilot risk, not an exhaustive regulatory risk checklist. |
| C04 | Disclose the transaction fee | The employee states the bank's transaction fee as an amount or a rate. Explicitly stating that no such fee applies also satisfies this disclosure check. | The employee omits the fee disclosure or merely says fees apply without an amount/rate. | S1 section 5.1 and footnote 1 (printed page 1) defining fees. The amount/rate threshold is the pilot's concrete interpretation. |
| C05 | Avoid guaranteed-future-return claims based on past performance | The employee makes no claim that past investment performance guarantees future returns. Absence is sufficient. | The employee makes or endorses such a claim. | S4 section 6.4.2(c), page 21. This specific prohibition is our application of its non-misleading-communication principle, not a verbatim rule in that source. |
| C06 | Avoid coaching inflated loss tolerance | The employee does not encourage the client to misstate their willingness to bear losses. Absence is sufficient. | The employee encourages declaring greater loss tolerance than the client genuinely expresses, to obtain access to an investment. | S1 section 4.2.3, printed page 13; Box 2, printed page 14. |

C01 concerns the investment goal; C02 concerns acceptable downside. C03 concerns
possible principal loss; C05 concerns an invalid claim about the predictive
certainty of historical performance. C02 checks whether the employee obtains
the client's view; C06 checks whether the employee manipulates that view. C04
concerns transaction cost. These are distinct behaviours, not duplicate
questions. One utterance can legitimately affect more than one check, and
semantically inconsistent gold combinations must not be forced into a later
coverage matrix. The fixed pilot matrix below has been reviewed against these
interpretations, without generating evidence or dialogue.

Prohibitions are assessed over the full recording. A client making a misleading
claim is not itself an employee breach unless the employee endorses it.
Repetition to reject or correct a claim is not endorsement. For pilot generation,
avoid ambiguous self-corrections; do not manufacture gold interpretations after
dialogue. C01/C02 count neutral confirmation of a volunteered answer; repeating
an unanswered question is not confirmation. C06 does not prohibit neutral
clarification or the client genuinely changing their preference.

### Sources actually inspected

* **S1 — MAS, Private Banking Sales and Advisory Practices — Observations and
  Supervisory Expectations from Thematic Inspections (Information Paper,
  February 2020).** Local file:
  [Private Banking Sales and Advisory Practices.pdf](<../../fyp-ai/langchain-backend/corpus/Private Banking Sales and Advisory Practices.pdf>).
  Sections 4.1 and 4.2(a),(c): printed page 12 / PDF page 14; section 4.2.3:
  printed page 13 / PDF page 15; Box 2: printed page 14 / PDF page 16;
  section 5.1 and footnote 26: printed page 18 / PDF page 20; footnote 1:
  printed page 1 / PDF page 3. Section 1.4 explains the information paper's
  observational purpose. It is not treated as a statute.
* **S2 — MAS, Notice on Recommendations on Investment Products (FAA-N16), local
  copy last updated 29 December 2025.**
  [Notice on Recommendations on Investment Products.pdf](<../../fyp-ai/langchain-backend/corpus/Notice on Recommendations on Investment Products.pdf>).
  Paragraphs 11(a) and 11(b), printed/PDF page 15, concern collecting financial
  objectives and risk tolerance. Paragraphs 2-4B set application boundaries;
  paragraph 11A sets an exception to part of paragraph 11. The notice does not
  establish that every private-banking call must repeat a full fact-find or that
  all accredited-investor relationships are within every provision's scope.
* **S3 — MAS, Information Paper on Good Practices of Market Conduct Controls in
  the Financial Advisory Industry.**
  [Local PDF](<../../fyp-ai/langchain-backend/corpus/Information Paper on Good Practices of Market Conduct Controls in the Financial Advisory Industry.pdf>).
  Printed/PDF page 5: "Good practices to support representative-customer
  engagement during advisory and sales process", specifically the panel on
  product features and risks. This is broader good-practice support for C03,
  not a direct capital-loss script requirement.
* **S4 — ABS-hosted Private Banking Code of Conduct, version effective
  1 September 2020.** No local ABS copy was found; the official PDF linked
  from S1 was inspected online:
  [Private Banking Code of Conduct](https://www.abs.org.sg/docs/library/pb-code-of-conduct-14-feb-2020.pdf).
  Section 6.4.2(c), printed/PDF page 21, supports C05's non-misleading
  communication principle. The application discussion on page 5 describes
  broad good-practice guidance for private-banking services to accredited
  investors, not a replacement for legislation. This is a version-specific
  reference, not a claim that the 2020 code is the latest edition.

The local execution-of-orders notice was inspected for scope but not used to
claim best execution from a transcript: execution quality cannot be established
from this call alone. The fair-dealing infographic is image-only under text
extraction and was not used for precise citations. The MAS web page for S3
returned a maintenance page; citations above use the inspected local PDF.

### Pilot assumptions requiring sponsor confirmation

* These calls include an investment discussion with a concrete client objective
  and loss preference. C01/C02 require confirmation in the recording even if
  information might already exist in bank records. This is a pilot observation
  rule, not a claim that regulators require a fresh fact-find on every call.
* Products in the pilot expose the client to possible capital loss. This keeps
  C03 applicable. Product-specific additional risks are outside this single
  check and must not be claimed as covered.
* C04 assesses audible disclosure of the bank's transaction fee only. It does
  not verify an external fee schedule, all embedded product costs or actual
  charged amounts. The fee basis must be coherent in later approved inputs.
* Each call contains one unambiguous client transaction instruction with an
  identifiable instrument and transaction currency. T01/T02 target that order,
  including details explicitly accepted by the client, not examples, unrelated
  currencies, instrument domicile or a guessed currency. Multiple orders or
  unresolved currency ambiguities require a future contract decision.
* K01/K02 request client values, not evidence of successful identity verification.
  Identification numbers, DOBs, instruments and currencies are predetermined
  in the coverage matrix before any evidence or dialogue is authored.

### Normalization selection

| Field | Existing rule | Expected representation |
| --- | --- | --- |
| `client_identification_number` | `exact_text` | Exact fictional test-identifier string |
| `dob` | `exact_dob_dd_month_yyyy` | Calendar-valid `DD Month YYYY` string with a two-digit day |
| `transaction_instrument` | `exact_text` | Exact instructed instrument-name string |
| `transaction_currency` | `exact_text` | Exact currency name/identifier explicitly used in the instruction |

The identification field reuses `exact_text`; its obsolete name-field mapping
is removed. The DOB rule now enforces the new canonical format. All four pilot
fields have mappings. No text aliases, case folding, trimming, fuzzy matching
or numeric tolerances were introduced. The matrix uses uppercase currency
codes as exact text; conversion from currency names to codes is not implemented.
Later evidence must communicate the selected identifier/code unambiguously;
semantic aliases are not automatically equivalent under `exact_text`.
DOB's structured target is now `DD Month YYYY`; dialogue will use spoken form
later. The existing `exact_compliance` rule is selected for compliance, while
the integer/number/boolean rules remain available but unused by this checklist.

## Five-call coverage matrix review

`authoring/coverage_matrix.csv` now contains the predetermined answers for
C01-C06, K01-K02 and T01-T02 across CALL_001 through CALL_005. There are 50 gold
answer cells, including one explicit K01 `NOT_STATED` cell, with no blank cells
or N/A/UNCLEAR labels. The CSV contains
answers only: no evidence sentences, speakers, turn IDs or timestamps.

The matrix is the proposed frozen answer plan, awaiting user approval. It was
created after the revised checklist and normalization contracts passed
validation. No downstream call artifacts were used to derive its values.

| Check | PASS count | FAIL count |
| --- | --- | --- |
| C01 | 3 | 2 |
| C02 | 3 | 2 |
| C03 | 2 | 3 |
| C04 | 3 | 2 |
| C05 | 3 | 2 |
| C06 | 3 | 2 |

For obligations, PASS means the required action occurs; FAIL means it does not
satisfy the requirement. For prohibitions, PASS means prohibited conduct is
absent; FAIL means it occurs. All six checks are applicable to all five calls.

### Logical consistency and short-call feasibility

CALL_001 is an all-PASS reference case. The other four calls contain mixed
outcomes. C06 is FAIL only for CALL_003 and CALL_005, where C02 is PASS:
genuine loss tolerance can be established before improper coaching. C01
concerns the investment objective and can independently be omitted in CALL_003.
Confirming willingness to bear losses does not automatically confirm an
investment objective. In later verification, an employee must not silently
replace the client's real preference with the coached response and treat that
as successful confirmation.

C05 is FAIL only for CALL_002 and CALL_003, where C03 is also FAIL. This avoids
having to interpret a capital-loss disclosure contradicted by a return
guarantee in this first pilot. CALL_004 omits capital-loss disclosure while
avoiding the prohibited return claim, so those two behaviours are not treated
as the same check. These are deliberate matrix choices, not new `depends_on`
relationships or universal logical implications between questions.

The behaviours require only a brief objective/loss-tolerance exchange, brief
risk/fee disclosures or omissions, KYC values (K01 is absent in CALL_005) and one transaction
instruction. The mixed outcomes are feasible within the intended 2-3 minutes;
duration has not been measured because no dialogue or audio exists yet.
This is a qualitative feasibility review, not automated semantic proof.

### Synthetic values and exact representation

All identification numbers, dates and fund names were invented for testing and
are not sourced from real clients or a product catalogue. K01 now asks:
"What is the client's NRIC or FIN number?" Its type and field remain CLIENT_KYC
and `client_identification_number`, using the unchanged `exact_text` rule.

The four populated identifiers have the requested nine-character format and
deliberately incorrect checksums. ICA confirms the [prefixes and FIN format](https://www.ica.gov.sg/news-and-publications/newsroom/media-release/new-m-fin-series-to-be-introduced-from-1-january-2022).
The checksum calculation was independently checked against the published
[JavaScript implementation](https://gist.github.com/alesmana/db3d215d649f987ff30e)
and [PHP implementation](https://github.com/IonBazan/NRIC/blob/main/src/NRIC.php).
These are public implementations, not an ICA registry or official validation service.

For S/F/G used here, multiply the seven digits by `2,7,6,5,4,3,2`, add four
for G (zero for S/F), and take modulo 11. Index `JZIHGFEDCBA` for S or
`XWUTRQPNMLK` for F/G. Results checked before insertion:

| Call | Synthetic value | Weighted total with offset | Remainder | Required suffix | Deliberate suffix |
| --- | --- | --- | --- | --- | --- |
| CALL_001 | S9100001X | 27 | 5 | F | X |
| CALL_002 | S8400002X | 48 | 4 | G | X |
| CALL_003 | F7800003A | 76 | 10 | K | A |
| CALL_004 | G8800004A | 84 | 7 | N | A |
| CALL_005 | NOT_STATED | — | — | — | — |

Each supplied suffix is also outside the valid checksum alphabet for its
prefix, independently demonstrating invalidity under these rules. Do not
correct the checksum or substitute another identifier during authoring.

For CALL_001-CALL_004, preserve the matrix identifier exactly in the config,
plan and eventual canonical expected value. Plan a CLIENT utterance with an
unambiguous character-by-character presentation: uppercase letters individually
and all seven digits individually, including every zero. Planned spoken evidence
must occur verbatim in one eligible CLIENT turn and the independently verified
transcript must support extraction of the exact nine-character identifier.
Do not confuse the written expected value with its spoken-form evidence.
No evidence phrase or dialogue is generated at this stage.

For CALL_005, decode K01's `NOT_STATED` to native JSON null before creating the
call config or gold check plan. Never place `NOT_STATED` or the string `null`
in JSON expected values. Plan K01 as absence throughout the entire dialogue,
with required speaker `ANY`: the client must not state an identification number,
and neither speaker may accidentally introduce one. Independently verify the
complete dialogue, including spoken forms. Canonical K01 evidence turns must
be empty and its evidence text must be an empty string. Preserve all of
CALL_005's other predetermined answers. This structural validator does not
verify dialogue semantics; that independent verification remains mandatory.

Each call uses one distinct fictional equity fund: `TEST SGD Equity Fund`,
`TEST USD Equity Fund`, `TEST EUR Equity Fund`, `TEST GBP Equity Fund` or
`TEST AUD Equity Fund`. These labels designate hypothetical unleveraged,
variable-value funds with possible capital loss and no guaranteed return.
Their intended transaction currencies are respectively SGD, USD, EUR, GBP
and AUD. These are explicit synthetic product assumptions, not assertions
about real funds. No financial/product lookup or investment recommendation is
involved. Currency codes should later be stated or unambiguously affirmed in
the instruction; they must not be inferred solely from the fund name.

The five DOBs are distinct, calendar-valid fictional dates written in the new
canonical form. No spoken DOB evidence has been authored. Four calls have a
stated identification number; CALL_005 has none. Every call retains one DOB
and one unambiguous transaction instruction.

### Validation performed and still pending

The existing schema and checklist/rule validators run before matrix authoring.
`tests/test_coverage_matrix.py` then parses the real CSV and verifies the exact
header/order, ten unique question rows, five call columns, nonempty cells,
PASS/FAIL-only compliance, at least two of each verdict per compliance row,
actual-value extraction cells, four checksum-invalid identifiers plus the
K01 absence sentinel, distinct canonical
calendar-valid DOBs, and matching fictional fund/currency pairs. It also checks
the two selected pilot combination constraints above. Run these checks with:

```powershell
python -m pytest -q tests/test_coverage_matrix.py
```

A reusable CSV validator, malformed-input CSV regression suite, approval
provenance, cross-file value preservation, semantic verification and duration
verification remain pending. The fixed pilot tests do not implement those
features. The canonical date tests also cover full English month names, leading
zeroes, Gregorian leap-year rules and rejection of legacy/numeric date forms.

### Contract migration at revision 0.3.0

Only K01 changed in the checklist. K02 and the other eight entries retain their
exact prior content. The old `dob_ddmmyyyy` value tag and
`exact_dob_ddmmyyyy` rule are retired; old DOB formats are not accepted under
the new contract. The normalization, call-config, check-plan and ground-truth
schema versions advance to `0.3.0`. Existing artificial fixtures were updated;
there were no real call artifacts to migrate. Dialogue/timing schemas and the
canonical zero-based turn-ID convention are unchanged. Any future imported
old-format data must be migrated explicitly rather than silently coerced.

### Narrow K01 migration at revision 0.4.0

Only K01's text and five CSV cells changed in the dataset. The other nine
checklist entries and CSV rows are preserved. No real call artifacts existed
to migrate. Normalization configuration, DOB rules and checklist shape are unchanged.

`check_plan.schema.json` adds `identification_expected` and the shared
parent-aware `answer_contract`. Only `question_id: K01` can select the
nullable text branch; other IDs retain the original non-null expected union.
Call config, check plan and canonical ground truth reuse that constraint and
advance to `0.4.0`. Null K01 requires absence evidence across the entire dialogue;
populated K01 requires presence evidence from CLIENT. Existing canonical absence
constraints enforce empty evidence IDs/text. Existing turn-ID rules are unchanged.

Python additionally checks populated K01 checksum invalidity, requires K01's
CLIENT_KYC identification field and exact-text rule, and provides the pure
`parse_k01_cell` decoder. It does not implement a general CSV validator or compiler.
Artificial regression tests cover null round-trips across all three contracts,
rejection of string sentinels and null on other questions, evidence constraints,
checksum-correct controls, and preservation of the other nine matrix rows.

## Decisions before further dataset authoring

Confirmed by the latest user requirements: rich internal checklist categories,
conditional requirement/field properties, no separate question answer-type map,
reusable normalization rules, and a derived `id`/`question` LangGraph request.
Exact planned evidence phrases remain as previously approved.

The following drafted details still need approval before freezing contracts:

* Review and approve the revised matrix before downstream authoring, including
  the existing fictional fund assumptions. The K01 wording, deliberately invalid
  identifiers and NOT_STATED/null convention implement the latest user task;
  sponsor confirmation of regulatory/policy applicability remains separate.
* Newline joining of multiple evidence excerpts and absence scope for checks
  other than K01. K01's whole-dialogue absence requirement is established by
  the current task. Agree how duplicate matches should be
  resolved while preserving the predetermined answers and plan.
* How checklist/matrix approval and independent verification will be recorded
  and tied to specific input versions. These foundations cannot enforce the
  temporal authoring order on their own.

Prediction treatment for unexpected N/A/UNCLEAR and actual render settings can
be decided before their respective later stages. No call configs, check plans,
dialogues, canonical ground truth or renders may be generated until the next
authorization. Future evidence must follow the approved matrix, never revise it.
