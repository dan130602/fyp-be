# FYP evaluation dataset foundations

Prepared dataset export: see [EXPORTING.md](EXPORTING.md) for exporting the
validated JSON snapshot to `fyp-ai`, manifest verification and safe updates.

Independent evaluation contracts for **AI-Powered Regulatory Compliance
Automation for Call Monitoring**. This project lives in the existing `fyp-be`
repository. It does not import or change the production backend or `fyp-ai`.

**Status: the checklist, matrix, five call configs, five check plans, five
dialogues, and five compiled canonical `ground_truth.json` files exist. Polly
audio has been generated locally for all five calls. Turn timing JSON, SSML,
and speaker manifests are versioned; MP3 files are ignored by Git.**
Check-plan and ground-truth versions are `0.6.0`;
approved call configs and their schema remain `0.5.0`, normalization remains `0.3.0`.
Tests cover artificial contract fixtures, the actual
checklist, the fixed pilot CSV, and regression cases mirroring the approved
real artifacts without duplicating their content. Matrix values are
predetermined and must not be reinterpreted from later dialogue.

## Layout

| Location | Purpose |
| --- | --- |
| `schemas/` | Six core Draft 2020-12 data schemas plus reusable normalization rules. References resolve locally; `.invalid` schema IDs are identifiers, not network endpoints. |
| `authoring/coverage_matrix.csv` | Approved, read-only source of all 50 predetermined answers. |
| `authoring/call_configs/` | Five approved configs with matrix-derived answers and authored scenario facts/intents; read-only in this stage. |
| `authoring/check_plans/` | Five approved predetermined gold checks (CALL_001-CALL_005), created before dialogue; read-only in this stage. |
| `authoring/dialogues/` | Five spoken dialogues without turn IDs. |
| `configs/normalization.json` | Reusable exact-comparison rules, a compliance rule reference and extraction field-to-rule references. No question-ID mappings or scoring implementation. |
| `configs/render.json` | Unset Polly rendering template, containing no credentials. |
| `src/fyp_evaluation/validation/` | Schema/integrity and cross-file checks, shared evidence matching, pilot spoken-value verification, independent annotation coverage, CLI and pure LangGraph checklist projection. |
| `src/fyp_evaluation/authoring/` | CSV loading, deterministic expected-answer compilation, mode derivation, safe scaffold writers and canonical ground-truth compilation. |
| `src/fyp_evaluation/rendering/` | Reserved for reusable rendering modules; `scripts/render_polly.py` provides the current Polly workflow. |
| `src/fyp_evaluation/scoring/` | Reserved for comparing pipeline predictions against gold. |
| `scripts/` | Command-line workflow wrappers, including Polly rendering and SSML export. |
| `tests/` | Artificial contract fixtures and regression tests, plus checks of the actual pilot checklist and five-call matrix. No generated calls or dialogue. |
| `evaluation_dataset/checklist.json` | The authoritative internal checklist: `id`, `type`, `question`, plus `requirement_type` for compliance or `field` for extraction. |
| `evaluation_dataset/calls/` | Five canonical calls and separate render outputs. Each call has ground truth, turn timing JSON, and per-turn SSML. Generated MP3 files stay local. |
| `evaluation_runs/` | Future predictions and evaluation results, kept separate from gold. |

All five calls have the following layout. The MP3 files are generated locally
and ignored by Git; timing JSON, SSML, and manifests are versioned:

```text
evaluation_dataset/
├── checklist.json
└── calls/
    ├── CALL_001/
    │   ├── ground_truth.json
    │   └── renders/
    │       └── clean_voice_A/
    │           ├── audio.mp3          (local only)
    │           ├── timings.json
    │           ├── ssml_manifest.json
    │           └── segments/
    │               ├── 0.mp3          (local only)
    │               ├── 0.ssml
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
   Speakers must strictly alternate: a turn is one uninterrupted stretch of
   speech, so an employee who continues (acknowledgement, recommendation, risk
   warning, ...) speaks in one turn, never two consecutive `EMPLOYEE` turns.
   `dialogue-plan` and the compiler reject consecutive same-speaker turns.
6. Independently verify the dialogue against every predetermined check,
   including speaker, absence and content constraints. Revise nonconforming
   dialogue; never derive or adjust gold from the completed dialogue.
7. The Python compiler validates all inputs, assigns turn IDs and resolves
   evidence deterministically, then writes canonical `ground_truth.json`.
8. Generate per-turn SSML from canonical reference text and separate pronunciation settings.
9. Render Polly segments, stitch them, and write timings for that specific
   render. Canonical ground truth remains unchanged.
10. Run the actual FYP system and compare its predictions against canonical
    gold. Evaluation outputs belong in `evaluation_runs/`.

Validation, deterministic config/check-plan scaffolding and canonical compilation are implemented.
No narrative generation,
semantic verification, LLM API, Transcribe, LangGraph integration, or evaluation
metrics are implemented. The five Polly renders were generated locally. A structural
validation pass is neither approval nor independent semantic verification.

If a predetermined answer must change, explicitly revise and reapprove the
upstream checklist/matrix/plan and regenerate affected downstream artifacts.
Do not retrofit gold answers to existing dialogue. Approval history and the
independent verification record format remain to be designed.

## JSON contracts

Objects reject unknown fields, except the explicitly keyed rule and field
maps. **Envelopes now version independently.** Call configs retain
`schema_version: "0.5.0"`; check plans and ground truth use `0.6.0`.
The approved call-config files and their schema have not been re-versioned.
Each intended answer now requires `question_id`, `expected` and a nonblank
`scenario_intent` string. Check plans add required `ordering_constraints`,
and both evidence branches add required `forbidden_patterns`; ground truth
inherits the evidence change by reference. No top-level ordering field is added
to ground truth. The earlier shared-version convention ends at this revision.
Normalization remains `0.3.0`. The package version is `0.5.0`. The checklist schema shape is now
revision `0.3.0` (its `question` property replaces the earlier `text`); dialogue and timing envelopes remain `0.1.0`. Call IDs match
`CALL_` followed by at least three digits. Question/evidence/fact/rule/render
identifiers start with a letter and contain letters, digits, `_` or `-`.
The internal checklist remains an array without an envelope. It is projected
to the legacy LangGraph request shape; it must not be passed there directly.

| Schema | Shape and key rules |
| --- | --- |
| `checklist` | Nonempty array with unique IDs. COMPLIANCE entries contain exactly `id`, `type`, `requirement_type` and `question`. CLIENT_KYC/TRANSACTION entries contain exactly `id`, `type`, `field` and `question`. |
| `call_config` | Version, call ID, `metadata` (`language`, `scenario`), positive ordered target duration bounds in seconds, named `scenario_facts`, and nonempty `intended_answers`. Each answer requires `question_id`, typed `expected` and nonblank `scenario_intent`. |
| `check_plan` | Version, call ID, nonempty `checks`, and required `ordering_constraints` (possibly empty). Each check has `question_id`, typed `expected`, and `evidence_plan`. No resolved turn IDs, critical tokens or render settings. |
| `dialogue` | Version, call ID and nonempty `turns`. Each contains `speaker`, reference-transcript `text`, and `critical_tokens` (possibly empty). No turn IDs. Speakers are `EMPLOYEE` or `CLIENT` and must strictly alternate (no two consecutive turns by the same speaker; merge them). |
| `ground_truth` | Version, call ID, metadata, checks preserving their expected answers and evidence plans, and `dialogue` with Python-assigned `turn_id`. Each check adds `gold_evidence_turns` and `gold_evidence_text`. |
| `timings` | Version, call ID, `render_id`, `audio_file: "audio.mp3"`, total `duration_seconds`, and a `turns` array. Each has integer `turn_id`, start/end seconds and optional `words` array. Word entries contain text and start/end seconds. All times are offsets from the start of stitched audio. |
| `normalization` | Version, named `rules`, `compliance_rule` and `field_rules`. Each rule declares its scalar `value_type` and `operation: "exact"`. Rules describe value representation, not question classification. |

COMPLIANCE requires `requirement_type: "OBLIGATION"` or `"PROHIBITION"` and
forbids `field`. CLIENT_KYC and TRANSACTION require `field` and forbid
`requirement_type`. Checklist `question` text must be nonblank. Fields follow
the same identifier syntax as IDs. Gold answers, evidence, source passages,
turn IDs, call-specific fields, the retired `text` name, `answer_type` and
`depends_on` are forbidden.

The only LangGraph adapter is a deterministic, in-memory selection (revision
0.3.0 stores `question` at the source, so this is no longer a rename):

```python
langgraph_checklist = [
    {"id": item["id"], "question": item["question"]}
    for item in checklist
]
```

`to_langgraph_checklist(checklist, schema_dir)` validates the internal shape
before applying this selection. It preserves IDs, order and exact question
text, drops `type`/`requirement_type`/`field`, does not mutate its input, and
never writes another checklist file or calls LangGraph. No production
workflow changes are required.

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
| Future planned evidence and dialogue text | Reference orthography of the same naturally spoken date, e.g. `13 May 2001`; see the orthography policy below |
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
in their turn's reference text. Canonical objects cannot contain voice, prosody,
timestamps, render IDs or word timings.

## Evidence contract and deterministic resolution

The user has approved exact planned evidence phrases for the initial contracts.

* Presence: `evidence_plan` has `mode: "presence"`, nonempty `planned_evidence`,
  `content_constraints` and `forbidden_patterns` arrays. Each planned item has a plan-wide unique
  `evidence_id`, `required_speaker` and exact `text`. No turn IDs are allowed.
* Absence: the plan has `mode: "absence"`, `scope: "entire_dialogue"`,
  `required_speaker` (`EMPLOYEE`, `CLIENT` or `ANY`), an `absent_event`
  description, `content_constraints` and `forbidden_patterns`. No positive quote is required.
  Either PASS or FAIL may be grounded in absence, depending on the question.
  This mode is also available for extraction checks. K01's absent value is
  native JSON null, with scope `entire_dialogue` and speaker `ANY`; canonical
  evidence is `gold_evidence_turns: []` and `gold_evidence_text: ""`.
  Other extraction fields still require concrete typed values.

### Deterministic evidence mode and ownership (Stage 4)

`derive_evidence_mode(checklist_item, expected)` is pure. It does not choose a
verdict or extraction value; `compile_expected` first obtains that answer from
the approved matrix. The representation-only `answer_contract` cannot establish
which answer or evidence mode is correct. The new cross-file validator does.

| Checklist type | Requirement / value | Locked answer | Derived mode |
| --- | --- | --- | --- |
| COMPLIANCE | OBLIGATION | PASS | presence |
| COMPLIANCE | OBLIGATION | FAIL | absence |
| COMPLIANCE | PROHIBITION | PASS | absence |
| COMPLIANCE | PROHIBITION | FAIL | presence |
| CLIENT_KYC or TRANSACTION | Non-null extraction, including zero or false | Actual value | presence |
| CLIENT_KYC K01 / client_identification_number | Null extraction | Native null | absence |
| Any other extraction | Null | Unsupported | ValueError |

Python owns `expected`, mode, absence scope (`entire_dialogue`), absence speaker
(`ANY` for null K01, otherwise `EMPLOYEE`), evidence-ID allocation and every
scaffold constant. Each presence scaffold starts with exactly one `<QID>_E1`
item: CLIENT for extraction, EMPLOYEE for compliance. There is no presence-level
`required_speaker`; every planned item carries its own role. K01's non-null
evidence always uses CLIENT. Compliance exchanges may legitimately span roles.

AI authors evidence text, absent-event descriptions, content constraints,
forbidden patterns and ordering constraints only in later Stages 4C/4D.
Current placeholders are `TODO_AUTHOR_PLANNED_EVIDENCE` and
`TODO_AUTHOR_ABSENT_EVENT`; both constraint arrays and ordering start empty.
No actual pilot plan has been built or written during this stage.

Use the **minimum number of evidence items** sufficient for the locked answer.
Default to one. Add items only if an answer needs multiple utterances or an
exchange: C01 PASS and C02 PASS may require an employee prompt and a client
response. Never add an item merely to give a FAIL positive evidence; an
obligation FAIL by omission needs no fabricated quote.

When more items are needed, Python allocates `<QID>_E2`, `<QID>_E3`, and so on,
appended in order. Never renumber, reuse or delete an existing ID, and never
alter expected answers or derived modes. The validator checks contiguous IDs
starting at 1 within each check and uniqueness across the plan. Proving that an
ID was not deleted/renumbered across revisions additionally requires comparison
with the prior approved plan; a single-document check cannot prove that history.

`ordering_constraints` contains only `{before: question_id, after: question_id}`
pairs. Both endpoints must exist and be presence checks. Self-reference and
cycles are rejected. These are evidence-order requirements, not applicability
dependencies. Plan validation checks the graph; the Stage 4B compiler resolves
turns and enforces `max(before_turns) < min(after_turns)`.

`forbidden_patterns` is a limited safeguard. An empty array is always legitimate;
never add a pattern without a precise, safe target. JSON Schema checks arrays
and nonblank strings; Python `re.compile` checks regex syntax. Python patterns
are not claimed to be ECMAScript regexes. Plan validation compiles patterns;
Stage 4B dialogue-plan validation also searches the relevant turns. A contiguous identifier regex does not
prove that no identifier was spoken with spaces, number words or other forms.
Absence verdicts always require independent whole-dialogue semantic review.

### Reference orthography for future canonical dialogue

Canonical `text` is the reference transcript for later ASR evaluation, not an
SSML script or a phonetic spelling guide. Choose the following written forms
in advance for English pilot speech. This is our convention for the normalized
forms we expect from Transcribe, **not a claim that Transcribe will reproduce
them**. AWS documents context- and language-dependent number formatting in
[Transcribing numbers and punctuation](https://docs.aws.amazon.com/transcribe/latest/dg/how-numbers.html).
The choices below are project conventions, not promises from that documentation.

| Content | Canonical reference-text convention |
| --- | --- |
| Amounts | Arabic digits with comma grouping for thousands and a decimal point when needed; omit unnecessary fractional zeroes for whole amounts. Keep an explicitly spoken currency code separate; do not add a currency that was only implied. |
| Percentages | Numeric rate followed immediately by `%`; preserve meaningful decimal digits rather than spelling out number words. |
| Currency codes | The explicitly spoken three-letter code in uppercase, contiguous letters without periods or spaces. Do not infer a code from the instrument name. |
| Synthetic identifiers | The locked `expected_value` itself always stays the exact uppercase, contiguous nine-character string, preserving every digit and zero — this compact form is never altered. Planned spoken evidence and canonical dialogue text MAY instead spell the identifier's characters out separately (e.g. "S, 9, 1, 0, 0, 0, 0, 1, X"), which the Stage 4B verifier accepts via a separator-tolerant match against the same locked characters. Whichever form dialogue text uses, it stays literal, case-sensitive prose — not SSML or a phonetic spelling guide, and never a description of pronunciation. |
| DOBs | Day number without ordinal suffix or leading zero, full title-case English month and four-digit year, separated by single spaces. The separately locked expected value remains `DD Month YYYY`, including its required two-digit day. |

Use ordinary sentence punctuation and capitalization. Planned verbatim evidence
must use the same reference orthography as eventual dialogue text. Pronunciation
hints, phoneme spellings, SSML and voice/channel settings remain outside canonical
text. A later smoke test will assess actual ASR output and these assumptions
without altering approved gold answers or silently rewriting the reference to
match predictions. The Stage 4B verifier accepts the limited authoring forms
documented below without rewriting reference text. No smoke test is implemented.

EMPLOYEE and CLIENT are semantic roles. The compiler assigns zero-based integer
`turn_id` values later. Neither roles nor turn IDs are audio channels or diarization
labels. Never assume a diarization label identifies the authored employee/client
without verification; any channel mapping belongs in rendering configuration.
AWS distinguishes diarization from speaker identification in its
[Transcribe service card](https://docs.aws.amazon.com/ai/responsible-ai/transcribe-speech-recognition/overview.html).

For each presence phrase, the shared matcher searches **only** eligible
speaker turns for a case-sensitive verbatim substring. Zero matches fails.
More than one eligible turn fails as ambiguous; never choose the first match.
Multiple planned phrases may resolve to the same turn. The check's
`gold_evidence_turns` is the sorted union of resolved turn IDs;
`gold_evidence_text` is the planned phrases joined by a single newline, in
plan order. These are exact excerpts, not newly inferred paraphrases. The
canonical validator checks these invariants through the same matcher; the
compiler additionally assigns IDs and requires nondecreasing resolved plan order.

Absence checks have `gold_evidence_turns: []` and `gold_evidence_text: ""`.
An empty array records no positive quote; it does **not** prove absence.
Meaning, contradictions, absences and content constraints still require
independent verification before compilation. A structurally valid dialogue
can still contradict its evidence phrase.

If an intended phrase occurs in multiple turns, stop and report the candidates.
Keep the approved gold answer unchanged. A disambiguation protocol (or a
permitted dialogue revision) needs explicit agreement before compilation can
continue; this scaffold has no automatic disambiguation or semantic matching.

## Coverage CSV contract

**JSON Schema does not validate this CSV.** `authoring.coverage.load_matrix`
uses Python's `csv` module to check structure and approved IDs/columns.
`compile_expected` checks cell types, while `validation.call_config.validate_call_config`
checks matrix-to-config agreement. `validation.check_plan.validate_check_plan`
also checks plans against their matrix and approved config. Pilot-specific tests
check the frozen answer distribution. Canonical compilation is implemented;
authoring chronology and approval provenance remain separate within the contract below:

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
prosody and silence settings. The template is not runnable. The rendering script
instead takes explicit command-line options with documented defaults; never add
credentials. Voice choices, speech rate, pitch, SSML and render-specific settings
belong outside canonical ground truth.

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

### Generate audio with Amazon Polly

To convert text into SSML locally, run these commands from `evaluation-dataset`:

```powershell
python scripts/convert_ssml.py --all
python scripts/convert_ssml.py --call CALL_001
python scripts/convert_ssml.py --input example.txt
python scripts/convert_ssml.py --text "Hello and welcome." --output example.ssml
```

Call conversion preserves turn IDs and speakers and writes
`renders/<render_id>/segments/<turn_id>.ssml` plus `ssml_manifest.json`.
Use `--render-id` to select another render folder. Text files produce a sibling
`.ssml` file by default; `--text` prints SSML when no output file is specified.
The converter wraps XML-escaped text in `<speak>` and makes no AWS requests.
It refuses to replace an existing text output or differing call SSML unless
`--overwrite` is supplied. Plain text conversion uses one selected Polly voice;
use call conversion for a dialogue with separate employee/client turns.

`scripts/render_polly.py` reads the canonical `ground_truth.json` dialogue for
each call. It assigns the employee and client separate voices, synthesizes each
turn, inserts 400 ms of silence between turns, and writes a stitched `audio.mp3`,
per-turn MP3 segments and turn-level `timings.json` under
`evaluation_dataset/calls/CALL_XXX/renders/clean_voice_A/`. It does not change
canonical ground truth or the prepared transcripts. Word timings are not
generated; listen to each render and check the spoken identifiers, amounts and
dates before using the audio for ASR evaluation.

The committed `clean_voice_A/timings.json` files describe the original
`clean_voice_A/audio.mp3` files. Keep those exact MP3 files in separate storage
for evaluation. A new Polly synthesis can have different durations and must use
its own timing file and render ID.

To save the exact SSML sent for each turn without making another Polly request,
run `python scripts/export_ssml.py --all`. Each render then has
`segments/<turn_id>.ssml` plus `ssml_manifest.json`, which maps turn IDs to
EMPLOYEE or CLIENT. The SSML wraps XML-escaped canonical text in `<speak>`;
the voice is selected by the Polly API request, not embedded in the SSML.

Install the rendering dependencies and configure an AWS profile with permission
to call `polly:DescribeVoices` and `polly:SynthesizeSpeech`. No AWS credentials
belong in this repository. Then run from this directory:

```powershell
python -m pip install -e ".[render]"
python scripts/render_polly.py --all
python scripts/render_polly.py --all --render-id clean_voice_B --execute
```

The command previews by default; `--execute` makes billable AWS calls. Defaults
are `ap-southeast-1`, neural engine, Jasmine for EMPLOYEE and Matthew for CLIENT.
Use `--region`, `--engine`, `--employee-voice`, `--client-voice`,
`--silence-ms` or `--render-id` to select another render. Existing audio is
protected unless `--overwrite` is passed; `--skip-existing` leaves completed
renders alone when rendering the batch. Do not overwrite the committed
`clean_voice_A` timings when creating a new audio version. Polly returns 16 kHz PCM, which the
script stitches before encoding the MP3 files with FFmpeg supplied by
`imageio-ffmpeg`.

## Transcript and speaker-role conversion

The five `timings.json` files contain render-specific turn timings.
`fyp_evaluation.transcripts` deterministically converts each call's
canonical `ground_truth.json` into two separate prepared files the offline
NLP evaluation framework in `fyp-ai` consumes, without an LLM, Polly, AWS
Transcribe, Bedrock or any other network call:

```text
evaluation_dataset/calls/CALL_XXX/
├── ground_truth.json      (unchanged)
├── transcript.json        (generated)
└── speaker_roles.json     (generated)
```

`transcript.json` is a normalized, AWS Transcribe-shaped results payload —
`transcripts: [{transcript: <full text>}]` plus one `audio_segments` entry
per canonical dialogue turn (`id`, `transcript`, `start_time`, `end_time`,
`speaker_label`), in original turn order. It has no outer `results`
wrapper and no `speaker_roles` key. `speaker_roles.json` separately holds
`{"spk_0": "RM", "spk_1": "Customer"}` — this key
ordering/mapping is fixed (EMPLOYEE is always `spk_0`/RM, CLIENT is always
`spk_1`/Customer) and independent of which speaker's turn comes first.
Only labels actually present in the dialogue are emitted. `fyp-ai`'s
ingestion node (`langgraph-backend/compliance_pipeline/nodes/ingestion.py`)
reconstructs its working transcript by joining diarized segments with two
newlines (`_TURN_SEPARATOR = "\n\n"`); `transcripts[0].transcript` uses the
same separator so the two representations agree.

The prepared transcript files still carry the documented temporary placeholder
`"start_time": "0.0", "end_time": "0.0"`. These are not measurements. The
same converter is meant to regenerate `transcript.json` with real
per-turn timing from `ground_truth.json` + `timings.json` together, rather
than patching the placeholder file in place; that enrichment path is not
implemented yet (`build_transcript_artifacts(..., timings=...)` raises
`NotImplementedError` for any non-`None` value).

Reusable API (`src/fyp_evaluation/transcripts/converter.py`, pure, no I/O):

```python
from fyp_evaluation.transcripts import build_transcript_artifacts

transcript, speaker_roles = build_transcript_artifacts(ground_truth)
```

Cross-file agreement between a ground truth and its generated pair —
segment count/order/ids, verbatim text, speaker-label mapping, the
full-transcript reconstruction, and speaker-role correspondence — is
checked by `fyp_evaluation.validation.transcript.validate_transcript_pair`
(imported from that module directly, not re-exported from
`fyp_evaluation.validation`, to avoid a package-init import cycle with
`fyp_evaluation.transcripts`).

```powershell
python scripts/build_transcripts.py --call CALL_001
python scripts/build_transcripts.py --all
python scripts/build_transcripts.py --all --check       # read-only; writes nothing
python scripts/build_transcripts.py --call CALL_001 --overwrite
```

Every selected call is built and cross-validated before anything is
written, so one malformed call cannot leave the dataset with a partially
updated or mismatched `transcript.json`/`speaker_roles.json` pair for
another call; existing outputs are never replaced without `--overwrite`.
Exit codes follow the `fyp-validate` convention: 0 success, 1 when
`--check` loads existing files but finds them cross-file inconsistent, 2
for input/usage errors and for a build that cannot produce a valid pair.

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

Stage 4 adds these command signatures using the same trailing inputs and optional
`--call-ids` convention. They are documented for later authorized authoring;
**do not run the scaffold command against real pilot data during this stage**:

```text
fyp-validate check-plan <path> <call_config> <matrix> <checklist> <normalization> [--complete]
fyp-validate scaffold-check-plans <matrix> <output_dir> <checklist> <normalization> [--overwrite] [--only CALL_ID ...]
```

`--only` selects output files only. The loader still requires the complete
approved matrix column list (five pilot calls by default); the writer validates
all columns and their configs before writing any selected output. The CLI reads
approved configs from `<matrix parent>/call_configs/<call_id>.json`, irrespective
of output location. Both commands work through `python -m fyp_evaluation.validation`
as well. Scaffolds pass without `--complete` and fail with it while placeholders
remain. `check-plan --complete` also invokes the shared Stage 4B spoken-value
verifier for non-null pilot text/DOB extractions, before any dialogue is needed.
Regex syntax checks also run in the `json` and `answers` CLI commands for
check plans and ground truth, after structural validation.

Reusable Stage 4 APIs:

* `authoring.check_plans.derive_evidence_mode(checklist_item, expected)` returns
  the derived mode or raises `ValueError` for unsupported combinations.
* `authoring.check_plans.build_check_plan(call_id, checklist, normalization, matrix)`
  compiles expected objects and constructs constants/placeholders, without I/O.
* `authoring.check_plans.write_check_plans(output_dir, checklist, normalization,
  matrix, call_configs, schema_dir, *, only=None, overwrite=False)` takes a mapping
  of all approved call IDs to config objects. It validates the complete batch,
  preflights output paths, uses exclusive creation by default and writes LF newlines.
  Existing regular files require explicit overwrite. As with the config writer,
  symlinks are always refused and writes are per file, not a batch transaction.
* `validation.check_plan.validate_check_plan(plan, call_config, checklist,
  normalization, matrix, schema_dir, *, complete=False)` returns errors. It checks
  internal config/plan ID equality first (reporting both IDs on mismatch), then
  structure, matrix identity and typed value equality, derived modes,
  speaker rules, evidence-ID sequences, regex syntax and an acyclic presence-only
  ordering graph. Complete mode reuses the call-config placeholder detector on
  evidence text, absent events and both authored constraint arrays. It also requires
  at least one planned item to communicate each locked non-null text/DOB extraction
  through the shared spoken-value verifier. Numeric/boolean scalar contracts keep
  their existing checks; their spoken forms are outside this five-call pilot.
* `validation.check_plan.validate_forbidden_patterns(document)` checks only regex
  syntax in an already schema-valid check plan or ground-truth object. Direct
  users of the low-level `validate`/`validate_answers` APIs must additionally call
  it for ground truth; JSON Schema alone cannot enforce Python regex syntax.

The shared placeholder detector was extracted unchanged from the call-config
validator with user approval. Existing test fixtures/assertions were migrated
for independent versions and required arrays. Stage 4B adds the tools below;
Entity WER scoring remains unimplemented.

### Stage 4B: deterministic compilation and separate annotation review

These command signatures are for later authorized authoring. This stage runs
them only on artificial test files, never on the approved pilot artifacts:

```text
fyp-validate dialogue-plan <dialogue> <check_plan>
fyp-validate critical-tokens <dialogue> <matrix> <checklist> <normalization>
fyp-validate compile-ground-truth <call_config> <check_plan> <dialogue> <output_path> <matrix> <checklist> <normalization> [--overwrite]
```

The matrix commands retain `--call-ids` (five pilot columns by default).
`--schema-dir` precedes the subcommand. Compilation validates everything before
opening the output. Existing regular files require `--overwrite`; symlinks and
output paths that replace an input are refused. Output is UTF-8 JSON with LF
newlines. Validation reports return exit 1; compilation refusals raise
`ValueError` and return exit 2 under the existing CLI exception convention.
No output is created on a validation refusal.

Reusable APIs and return values:

* `validation.evidence.resolve_evidence(evidence_plan, turns)` returns
  `(resolved_ids, errors)`. Successful IDs are in planned-item order, including
  repetitions when items share a turn. Structured errors contain `code`
  (`not_found` or `ambiguous`), `evidence_id`, `matching_turn_ids` and `message`.
  Candidates in errors are diagnostics, never a selected match. Repetition
  inside one eligible turn is valid; matches in multiple eligible turns fail.
* `validation.evidence.validate_dialogue_against_plan(dialogue, check_plan)`
  returns error strings. It checks internal call-ID equality first, then exact
  matches and nondecreasing evidence order. Schema validation is a separate
  caller responsibility. This helper never reads or requires `critical_tokens`.
  Absence patterns use the plan-level speaker (ANY means all); presence patterns
  search all turns because presence has only item-level speakers.
* `validation.spoken.verify_spoken_value(expected, planned_text)` returns
  `None` for a permitted extraction form or an error string. It checks gold
  authoring only and never alters the canonical answer or parses predictions.
  Instrument text must occur verbatim, case-sensitively. Pilot currency codes
  SGD/USD/EUR/GBP/AUD and identifier-shaped text permit only whitespace, commas,
  periods and hyphens between canonical characters, retaining letter case.
  Identifier digits may also use the closed zero–nine word table.
  DOB forms use a padded day, bare day, cardinal word or the closed 1–31 ordinal
  table, optionally followed by `of`, a full month and optional comma, then a
  four-digit year or two-part spoken year for the pilot's 19xx dates. Generated
  word compounds permit spaces or hyphens, e.g. `nineteen eighty-two`.
  Thus locked `04 August 1991` accepts `the fourth of August, nineteen ninety-one`
  and rejects `the fifth of August, nineteen ninety-one`. The expected object
  still requires exact matrix-derived value and Python type; accepted spoken
  forms never replace it. Exact phrase matching between plan and dialogue remains
  separate from checking whether a phrase communicates the locked value.
  DOB matching is case-insensitive and does not match fragments of larger days
  or years. Null values need no evidence. Other scalar spoken forms, aliases,
  phonetic letter names and general date parsing are outside this pilot scope.
* `validation.tokens.validate_critical_token_coverage(dialogue, checklist,
  normalization, matrix, *, schema_dir=...)` returns `(errors, advisory)`.
  It reuses schema/integrity validation for nonblank, unique, own-turn literal
  annotations. From the validated dialogue's internal ID it selects the matrix
  column and requires at least one token anywhere in the dialogue to satisfy the
  spoken verifier for each non-null extraction. A larger containing span counts;
  null K01 is exempt. There is no token-count cap and no check-plan argument.
  Annotations may sit outside all planned evidence, and evidence turns may have
  empty token arrays. Separate advisory records (`turn_id`, `text`, `message`)
  identify uncovered financial-looking digit, percentage, currency and amount
  strings. They never fail validation or prove annotation completeness. Repeated
  strings have no occurrence offsets; this is a review aid, not span tokenisation.
* `authoring.compile.compile_ground_truth(call_config, check_plan, dialogue,
  checklist, normalization, matrix, *, schema_dir=...)` returns a validated
  canonical object or raises `ValueError`; it performs no I/O and mutates no
  inputs. All three internal call IDs must agree and select an existing matrix
  column before other validation. Config and plan must pass complete mode.
  It independently rechecks each copied expected object against `compile_expected`
  by Python type and value. Each non-null extraction must have at least one
  planned evidence item denoting its full locked value. Independent annotation
  coverage is also a compilation gate; advisories are nonblocking.

The compiler assigns contiguous integer IDs from dialogue order, copies metadata,
expected objects and evidence plans, uses sorted unique evidence IDs and
newline-joined planned texts, and preserves text and token strings/order exactly.
Absence checks retain `[]` and `""`. It enforces evidence-order edges, rejects
ambiguity and pattern hits, and validates the finished canonical object. It never
uses annotations to choose compliance outcomes, repairs answers, or infers
semantic truth from the dialogue. Unencoded contradictions, free-text content
constraints, whole-dialogue semantic absence and approval/provenance still need
independent review in the later authoring stages.

For installed use, explicitly supply the schema resource directory to the token
validator and compiler. Their defaults locate `schemas/` in this source checkout;
schemas are not bundled in the wheel. Tests cover the compiler only with clearly
artificial inputs. No real check plan, dialogue, canonical gold, SSML or audio is
created in Stage 4B. Schema files and all approved data remain unchanged.

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
subcommand. Validation commands are read-only; `scaffold-call-configs`,
`scaffold-check-plans` and `compile-ground-truth` explicitly write outputs.
Tests cover the six core shapes,
field-rule references, mixed checklist categories, legacy LangGraph projection,
malformed data, and the most important structural
invariants, malformed CSVs and matrix-to-config answer preservation.
They do not certify legal correctness, approval history or semantic evidence.

## Stage 3A: deterministic call-config scaffolding

The approved CSV is the sole source of all 50 expected answers. The shared
`answer_contract` specifies their JSON representation, never which answer is
correct. Python copies/decodes each cell; AI must not author, retype, infer or
change expected answers. Only later narrative fields may be authored.

Each scaffold has this structure (shape notation, not another dataset artifact):

```text
schema_version: "0.5.0"
call_id: the selected approved CSV column
metadata:
  language: "en-SG"
  scenario: "private_banking_investment_advice_and_order"
target_duration_seconds: {min: 120, max: 180}
scenario_facts: []
intended_answers: ten entries in checklist order
  - question_id: checklist ID
    expected: compiled from the corresponding matrix cell
    scenario_intent: "TODO_AUTHOR_SCENARIO_INTENT"
```

The existing metadata, fact and duration structures and strict extra-property
restrictions remain. Intent describes what should occur or be withheld; exact
evidence belongs in a later check plan. No evidence, turn IDs, dialogue or timings
are introduced here. Only the required intent field changes the call-config
shape; the other two shared envelopes change version only.

Reusable APIs:

* `authoring.coverage.load_matrix(path, *, checklist=None, call_ids=PILOT_CALL_IDS)`
  returns `{call_id: {question_id: raw_string_cell}}`. By default it reads the
  authoritative checklist at `path.parent.parent/evaluation_dataset/checklist.json`
  and requires CALL_001-CALL_005 in order. Pass the approved checklist and an
  explicit ordered call-ID list for other layouts or larger datasets. Rejects
  duplicate/unknown/missing IDs, duplicate/wrong/missing call headers, ragged rows,
  blank cells, NULs and malformed CSV. Quoted commas/newlines and text whitespace
  are preserved. Row identity is by ID; scaffold output follows checklist order.
* `authoring.coverage.compile_expected(checklist_item, normalization, cell)`
  preserves PASS/FAIL, text and calendar-valid DOB strings. It parses finite
  numeric/boolean JSON scalars using the field rule without coercing types.
  Only the existing K01 parser maps `NOT_STATED` to `None`/JSON null. Validate
  the checklist and normalization before using this low-level function.
* `authoring.call_configs.build_call_config(call_id, checklist, normalization, matrix)`
  returns a deterministic unfilled object without mutating any input.
* `validation.call_config.validate_call_config(data, checklist, normalization, matrix, schema_dir, *, complete=False)`
  reuses schema and answer validation, then checks call-column membership, exact
  checklist order, matrix value/type equality and that no fact name duplicates
  an extraction field. Existing duration-order and unique-fact-name checks apply.
  Returns an error list without changing inputs. Load the matrix with the
  approved column list first.
* `authoring.call_configs.write_call_configs(output_dir, checklist, normalization, matrix, schema_dir, *, overwrite=False)`
  validates the whole batch and existing paths before writing. Exclusive file
  creation refuses overwriting by default. Explicit `overwrite=True` or CLI
  `--overwrite` is required to replace existing files. Symlink targets are refused.
  Writes are per file, not transactional: an I/O failure can leave earlier
  validated files in place; retrying will protect them from overwrite.

`complete=False` permits empty facts and the designated intent placeholder.
`complete=True` requires populated facts and authored intents. It rejects
reserved TODO/TBD/FIXME/PLACEHOLDER markers (case-insensitive), plus bare
TBC, PENDING, N/A, NONE, NULL, `...`, `-` and TO BE AUTHORED. Blank intents
always fail the schema. Completeness is a mechanical check, not semantic review.
All five fresh scaffolds pass scaffold mode and fail complete mode with exactly
11 unfinished-content errors each: one empty fact list and ten placeholder intents.

From this project directory, after installation (or setting `PYTHONPATH=src`):

```powershell
python -m fyp_evaluation.validation matrix authoring/coverage_matrix.csv evaluation_dataset/checklist.json configs/normalization.json
python -m fyp_evaluation.validation scaffold-call-configs authoring/coverage_matrix.csv authoring/call_configs evaluation_dataset/checklist.json configs/normalization.json
python -m fyp_evaluation.validation call-config authoring/call_configs/CALL_001.json authoring/coverage_matrix.csv evaluation_dataset/checklist.json configs/normalization.json
python -m fyp_evaluation.validation call-config authoring/call_configs/CALL_001.json authoring/coverage_matrix.csv evaluation_dataset/checklist.json configs/normalization.json --complete
```

The generation command now refuses because the five files already exist; this
is intentional. The final command failed for fresh Stage 3A scaffolds; it now
passes for the approved, completed Stage 3B configs. The same subcommands are exposed through `fyp-validate`.
Use `--call-ids CALL_... CALL_...` to explicitly select another approved column
set; extra/missing/reordered columns otherwise fail. Normalization remains at
`0.3.0`; approved checklist and CSV bytes are unchanged by this stage.

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

### Checklist field rename at revision 0.3.0 (`question` replaces `text`)

The richer `{id, type, text, requirement_type|field}` shape from the previous
paragraph kept its free-text property named `text`, distinct from the
LangGraph-only `question` name produced by `to_langgraph_checklist`. Manual
testing against the actual LangGraph deployment (Studio) surfaced this as a
real source of confusion: pasting the raw checklist directly into a Studio
input field produced `KeyError('question')`, because Studio does not run any
adapter for you. Rather than only document that workflow step, the checklist
source itself was changed so it already matches the graph's field name.

`evaluation_dataset/checklist.json`, `schemas/checklist.schema.json` and
`to_langgraph_checklist` were updated together: every checklist item's prompt
is now stored under `question` instead of `text`; `type`, `requirement_type`
and `field` are unchanged. `to_langgraph_checklist` no longer renames a field,
it only selects `id`/`question` and drops the classification fields. `text`
is now itself an out-of-scope, `additionalProperties: false`-rejected legacy
name — the inverse of the previous paragraph's migration, which rejected the
old bare `{id, question}` shape for missing classification. Both remain true
simultaneously: a bare `{id, question}` item is still rejected today, but now
because `type`/`requirement_type`/`field` are missing, not because of the
field's name.

`fyp-ai/evaluation/fyp_evaluation_loader/dataset.py`'s independent
`_project_checklist` (which deliberately reimplements this same selection
without importing this package) and `fyp-ai/evaluation/README.md` were
updated in lockstep so a published snapshot's `checklist.json` and its loader
agree. `fyp-ai/evaluation/tests/test_dataset.py` exercises this against the
committed `datasets/pilot/` snapshot; regenerating that snapshot (via
`EXPORTING.md`) after this rename is future publication work, not part of
this change.

At revision `0.2.0`, `call_config`, `check_plan` and `ground_truth` already shared
a compliance/extraction union, so their shapes needed no changes. Revision
`0.3.0` changes the DOB branch of that shared union and versions its consumers.
Mixed-category artificial
fixtures confirm that DOB strings, numeric values and text survive in each
shape. This establishes contract compatibility along matrix -> config ->
plan -> ground truth. Stage 3A now implements CSV-to-config compilation and
validation; comparison against plans is now implemented in Stage 4. Stage 4B
implements canonical compilation. Comparing pipeline predictions remains future work.

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

The matrix is the approved frozen answer plan for Stage 3A. It was
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

A reusable CSV loader, malformed-input regression tests and matrix-to-config
preservation checks are now implemented in Stage 3A. Approval provenance,
agreement with later artifacts, semantic verification and measured duration
remain pending. The canonical date tests also cover full English month names, leading
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
`parse_k01_cell` decoder. At revision 0.4.0, a general CSV validator/compiler
was not implemented; Stage 3A now adds CSV-to-config support using this decoder.
Artificial regression tests cover null round-trips across all three contracts,
rejection of string sentinels and null on other questions, evidence constraints,
checksum-correct controls, and preservation of the other nine matrix rows.

## Decisions before further dataset authoring

Confirmed by the latest user requirements: rich internal checklist categories,
conditional requirement/field properties, no separate question answer-type map,
reusable normalization rules, and a derived `id`/`question` LangGraph request.
Exact planned evidence phrases remain as previously approved.

The following drafted details still need approval before freezing contracts:

* The checklist, matrix and completed call configs are approved. Review these
  Stage 4 contract/tooling changes before authorizing any real plan scaffolding
  or evidence authoring.
  Sponsor confirmation of regulatory/policy applicability remains separate.
* The future revision/reapproval protocol for resolving duplicate matches while
  preserving predetermined answers and plans. Newline joining is now fixed by
  the Stage 4B compiler contract; ambiguous matches are always refused.
  Mode derivation, whole-dialogue absence scope and absence speakers are now
  established by the Stage 4 contracts rather than awaiting a new convention.
* How checklist/matrix approval and independent verification will be recorded
  and tied to specific input versions. These foundations cannot enforce the
  temporal authoring order on their own.

Prediction treatment for unexpected N/A/UNCLEAR and actual render settings can
be decided before their respective later stages. Do not fill scenario facts or
intents or generate check plans, dialogues, canonical ground truth or renders
without the next authorization. Future evidence must follow the approved matrix,
never revise it.
