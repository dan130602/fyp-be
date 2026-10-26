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
| `datasets/legacy_pilot_v1/authoring/coverage_matrix.csv` | Approved, read-only source of all 50 predetermined answers. |
| `datasets/legacy_pilot_v1/authoring/call_configs/` | Five approved configs with matrix-derived answers and authored scenario facts/intents; read-only in this stage. |
| `datasets/legacy_pilot_v1/authoring/check_plans/` | Five approved predetermined gold checks (CALL_001-CALL_005), created before dialogue; read-only in this stage. |
| `datasets/legacy_pilot_v1/authoring/dialogues/` | Five spoken dialogues without turn IDs. |
| `configs/normalization.json` | Reusable exact-comparison rules, a compliance rule reference and extraction field-to-rule references. No question-ID mappings or scoring implementation. |
| `configs/render.json` | Unset Polly rendering template, containing no credentials. |
| `src/fyp_evaluation/validation/` | Schema/integrity and cross-file checks, shared evidence matching, pilot spoken-value verification, independent annotation coverage, CLI and pure LangGraph checklist projection. |
| `src/fyp_evaluation/authoring/` | CSV loading, deterministic expected-answer compilation, mode derivation, safe scaffold writers and canonical ground-truth compilation. |
| `src/fyp_evaluation/rendering/` | Reserved for reusable rendering modules; `scripts/render_polly.py` provides the current Polly workflow. |
| `src/fyp_evaluation/scoring/` | Reserved for comparing pipeline predictions against gold. |
| `scripts/` | Command-line workflow wrappers, including Polly rendering and SSML export. |
| `tests/` | Artificial contract fixtures and regression tests, plus checks of the actual pilot checklist and five-call matrix. No generated calls or dialogue. |
| `datasets/legacy_pilot_v1/` | Frozen five-call pilot release. Its manifest identifies the dataset/checklist and enumerates calls; its checklist, authoring inputs, canonical calls and renders move together. |
| `datasets/advisory_execution_pilot_v1/` | New advisory/execution release. Its five G0 (Stage 0) contracts and the G1 planning outputs (`authoring/call_manifest.csv`, `coverage_matrix.csv`, `edge_case_assignments.csv`, `planning_record.json`) are present; calls are empty until later stages. |
| `schemas/stage0/` | Schemas for the release-level Stage 0 contracts (check semantics, release normalization, edge-case catalog, generation plan). Kept out of the top-level `schemas/*.json` set used by the legacy exporter. |
| `evaluation_runs/` | Future predictions and evaluation results, kept separate from gold. |

All five calls have the following layout. The MP3 files are generated locally
and ignored by Git; timing JSON, SSML, and manifests are versioned:

```text
datasets/
├── legacy_pilot_v1/
│   ├── dataset_manifest.json
│   ├── checklist.json
│   ├── authoring/
│   └── calls/
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
│       └── CALL_005/
└── advisory_execution_pilot_v1/
    ├── dataset_manifest.json
    ├── checklist.json
    ├── authoring/
    └── calls/
```

The schema does not hard-code five calls, ten questions, or a maximum recording
duration. Additional calls and renders use the same contracts.

## advisory_execution_pilot_v1: G0 Stage 0 contracts

| File | Owns |
| --- | --- |
| `checklist.json` | The 20 questions (C01-C12, K01-K02, T01-T06), validated by `schemas/checklist.schema.json`. |
| `check_semantics.json` | PASS/FAIL conditions (C05 FAIL is the exact complement of PASS), a `presence_pattern` per compliance check (EMPLOYEE_ONLY, EXCHANGE or ANY_SPEAKER), two relative `ordering_rules` (C03 after the C02 disclosure; C08 after the final transaction state), extraction final-state resolution, and the single hard dependency C03 PASS => C02 PASS. Each extraction question names a `normalization_profile`. |
| `normalization.json` | Release-local canonical forms, aliases, spoken/reference-text forms and comparison steps. Declarative only (`DECLARATIVE_ONLY`); nothing implements these steps yet. Edge-case-only spoken forms are required on the targeted value when their edge case is assigned (`applies_to`: every zero for "oh", every run of the assigned length for double/triple, the whole value for shorthand) and forbidden otherwise; the letter O is excluded from K01 because it is ambiguous with a spoken zero. The legacy `configs/normalization.json` is not used by this release. |
| `edge_case_catalog.json` | Edge-case targets, required parameters, verdict requirements, gold prerequisites, antecedent speakers and the readback mutual exclusion. |
| `authoring/generation_plan.json` | Exact-count G1 distribution: 50 calls (45 SHORT at 180-300 s, 5 LONG at 1801-3600 s), the 5/5/5/35 composition grid, exact per-call counts of compliance FAILs (0/1/2/3/4 FAILs in 10/10/15/10/5 calls, at least 6 FAILs per check) and of edge cases (0/1/2/3/4 in 10/10/15/10/5 calls, 90 in total), value pools and product currencies. |

All 20 questions apply to every call. Call intents (`["ADVISORY", "EXECUTION"]`)
are multi-label and descriptive only: `ADVISORY|EXECUTION` in CSV, a JSON array
in JSON. There is no applicability routing, N/A or NOT_APPLICABLE.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -c "from fyp_evaluation.validation.stage0 import validate_stage0_release as v; print(v('datasets/advisory_execution_pilot_v1') or 'valid')"
python -m pytest -q tests/test_stage0_contracts.py
```

Pipeline names for this release: G0 contracts, G1 deterministic planning, G2A
call_spec (immutable call specifications), G2B check_plan_skeleton (deterministic check-plan
skeletons), G3 call_config, G4 check_plan wording, G5 plan verification, G6 dialogue, G7 dialogue
verification, G8 audio (SSML, Polly, timings), G9 ground truth and evaluation. Implemented: G0, G1 (`fyp_evaluation.planning`),
G2A (`fyp_evaluation.call_specs`) and G2B (`fyp_evaluation.check_plan_skeletons`) without an LLM, and G3
(`fyp_evaluation.call_configs`, model-assisted) and G4 (`fyp_evaluation.check_plans`, model-assisted, structural acceptance only),
G5 (`fyp_evaluation.plan_verifications`, gold-aware semantic verification by Gemini, Python-adjudicated) and G6
(`fyp_evaluation.dialogues`, model-assisted dialogue generation, structural acceptance only), G7
(`fyp_evaluation.dialogue_verifications`, gold-aware dialogue verification), G8 (`fyp_evaluation.rendering`, SSML conversion and Amazon
Polly rendering with per-turn timings; the SSML and the offline gate are implemented, the billable Polly run is separate) and a
G9 (`fyp_evaluation.ground_truths`, deterministic canonical ground-truth compilation; the FINAL compile with audio-derived timings is
implemented and has been run). The stage numbers follow the order of work: the final canonical ground truth needs the real turn timings, so it is compiled last.

G6 dialogue -> G7 verification -> G8 render (SSML, Polly audio, `timings.json`) -> **final G9 compile** (stitches the timings in) ->
export and evaluation. The G9 files of the advisory/execution release are the FINAL compile of render `clean_voice_A`; a provisional compile
(placeholder `0.0` times) is still available by omitting `--render-id`, but is refused over final outputs.

Model decisions per stage: G3, G4 and G6 use `gpt-6.1-sol` (OpenAI). G5 uses `gemini-3.8-flash` with thinking
level `medium` on the paid Flex tier; G7 uses `gemini-3.8-flash` with thinking level `high` on Flex.

```powershell
python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_pilot_v1             # write (refuses existing files)
python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_pilot_v1 --check     # byte-for-byte reproduction
python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_pilot_v1 --validate  # validate files on disk
```

G2A (`fyp_evaluation.call_specs`, no LLM) compiles the frozen G1 outputs into one immutable
`authoring/call_specs/CALL_XXX.json` per call plus `authoring/call_spec_record.json`. It is a pure
join: nothing is sampled, normalized or invented, and it refuses to run unless G0 and G1 validate
on disk. Its strongest invariant is that the 50 specs project back to the three G1 CSVs
byte-for-byte (via G1's own serializer), with hashes matching `planning_record.json`. Schemas live
in `schemas/stage2/`, outside the legacy top-level schema set.

```powershell
python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_pilot_v1 --overwrite  # replace the full set; refuses without the flag
python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_pilot_v1 --validate   # validate files on disk against G1
python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_pilot_v1 --check      # byte-for-byte reproduction; reports missing/changed/extra
```

G2B (`fyp_evaluation.check_plan_skeletons`, no LLM) turns each immutable call spec plus the frozen
G0 semantics, edge-case catalog and normalization profiles into
`authoring/check_plan_skeletons/CALL_XXX.json` and `authoring/check_plan_skeleton_record.json`. A
skeleton is a purely structural evidence blueprint with three parts: `planned_events` (symbolic
records with closed kinds and statuses, `allowed_speakers`, relative references and provider-independent
spoken-form requirements), `checks` (all 20 questions: evidence mode, condition references, FINAL events,
gold-restatement policy) and `constraints` (only ORDER, SAME_TURN, NOT_SAME_TURN and
MIN_INTERVENING_TURNS, each with provenance). It contains no wording, turn numbers or SSML. G2B reads
business truth only from the call specs and G0, and refuses to run unless G0, G1 and G2A validate on disk.
Each planned event is realised later as one contiguous span within a single turn. Every catalog edge case
has exactly one compiler in `edge_cases.py`; a contradictory combination (for example a long-distance gap
between questions that share one FINAL event) fails loudly instead of being silently relaxed.

G2B conventions the code fixes (structural, not business truth; nothing here changes gold):

- Speakers: extraction evidence (VALUE_STATEMENT, VALUE_CORRECTION, ORDER_STATEMENT, REFERENCE_SELECTION) is CLIENT;
  READBACK is EMPLOYEE; ORDER_INVALIDATION, GUARANTEE_REQUEST and WEAK_ACKNOWLEDGEMENT are CLIENT; GUARANTEE_REFUSAL is
  EMPLOYEE. Compliance patterns: EMPLOYEE_ONLY events are EMPLOYEE; EXCHANGE is an EMPLOYEE ACT then a CLIENT RESPONSE
  ordered ACT before RESPONSE; ANY_SPEAKER allows `[CLIENT, EMPLOYEE]`. Speakers are sorted arrays.
- Granularity: each planned event is one contiguous span in one turn; a plain extraction field is one FINAL
  VALUE_STATEMENT with no ordering against other fields. Distractors carry no ordering relation to FINAL events.
- Failures by omission (for example C03 or C08 FAIL) create no compliance events; ABSENCE checks carry no evidence events.
- Ordering rules: the C03 rule orders the C03 ACT after C02 events; the C08 rule orders the C08 request ACT after
  the distinct FINAL transaction events.
- Constraints: exact duplicates (same type, endpoints, parameters and provenance) are merged; event pairs are stored
  sorted by event ID; the same endpoints with different provenance stay separate.
- Spoken-form IDs come from `normalization.json`; readbacks use `min_field_count` 1 (error) or the catalog value (one-field).

```powershell
python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_pilot_v1 --overwrite  # replace the full set; refuses without the flag
python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_pilot_v1 --validate   # validate files on disk (independent re-derivation + canonical bytes)
python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_pilot_v1 --check      # byte-for-byte reproduction; reports missing/changed/extra
```

G3 (`fyp_evaluation.call_configs`, model-assisted: OpenAI `gpt-6.1-sol` through the Responses API with strict
Structured Outputs, sync or Batch) writes the creative scenario for each call into
`authoring/call_configs/CALL_XXX.json`. Python owns every immutable fact: it refuses to run unless G0-G2B validate,
derives product references (`P1`, `P2`, ... in first-seen order, roles unioned), the required slots (fee disclosure,
limit prices, holding context, red flags, edge-case motivations) and the whole request; the model returns only
scenario content (`schemas/stage3/call_config_response.schema.json`), which Python validates against the call's
gold-conditioned rules before it is accepted. The artifact contains no gold and no edge parameters (G4 onward load the
call spec); it records the call spec and skeleton hashes. The skeleton body is validated but never sent to the model.
Convention (G3/G4, not G0): the final T01 product is the recommended product; for a SELL the recommendation may be to
sell, reduce or exit that holding; C05 stays a general capital-loss disclosure; non-final products are only frozen
candidates, comparisons or superseded orders.

```powershell
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run [--call-id CALL_001] [--print-request]  # offline: gate, 50-call feasibility, request preview; writes nothing
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --sync --call-id CALL_001       # ONE paid live request
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --submit [--call-id ...]         # paid Batch of all eligible calls (or a subset)
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --collect                        # poll + ingest; safe to repeat
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --status | --validate [--require-complete] | --check   # offline
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --attach-batch SUB_0001 batch_...   # recover a SUBMISSION_UNKNOWN submission
python scripts/generate_call_configs.py --dataset-root datasets/advisory_execution_pilot_v1 --mark-not-submitted SUB_0001 --acknowledge-duplicate-risk
```

Modifiers: `--retry-rejected` (new attempts after a REJECTED one, 3 per call), `--reopen --call-id X --reason "..."`
(deliberately supersede an accepted config; the previous accepted version stays in the provenance history), `--audit-dir`.
There is no `--overwrite`: accepted calls are never selected unless reopened. Credentials: copy `.env.example` to `.env`
in this directory (git-ignored) and set `OPENAI_API_KEY`, or set it in the operating-system environment (which wins);
offline commands and the whole test suite need no key. Install the SDK with `pip install -e ".[llm]"`.
Operational state (live attempts, canonical requests, raw provider envelopes, batch submissions, the run lock) lives in
the git-ignored `.llm_audit/`; the committed trail is `authoring/call_config_provenance/CALL_XXX.json` (attempt summaries,
accepted parsed response, accepted history) plus `authoring/call_config_record.json`. `--check` rebuilds every accepted
artifact from the stored parsed response and the frozen inputs, never from a provider envelope.
Each Batch submission keeps, under `.llm_audit/<dataset_id>/G3/submissions/`, its record (`SUB_NNNN.json`, with the
terminal batch snapshot), the exact uploaded `SUB_NNNN.input.jsonl` and the exact downloaded `SUB_NNNN.output.jsonl` /
`SUB_NNNN.error.jsonl`. A call that provenance records as accepted but whose artifact is missing is reported as
`INCONSISTENT` and blocked from paid attempts until investigated. A sync pilot that must not count towards production is
run against an isolated copy of the release with its own `--dataset-root` and `--audit-dir`.
A Batch creation whose outcome is unknown is recorded as `SUBMISSION_UNKNOWN`: it is never resubmitted automatically
(its calls are blocked) and is resolved only by `--attach-batch` or `--mark-not-submitted --acknowledge-duplicate-risk`.

G4 (`fyp_evaluation.check_plans`, model-assisted, same `gpt-6.1-sol` and shared `StageRunner` lifecycle as G3) writes
`authoring/check_plans/CALL_XXX.json`: the frozen G2B skeleton copied verbatim by Python plus, per planned event, an
`authoring` block (resolved speaker, the model's text template, the Python-rendered `planned_text` and exact
`value_mentions` with character offsets) and `derived_constraints`. The model only returns, per skeleton event, a text
template with `[[PLACEHOLDER]]` slots, a speaker choice where more than one is allowed, spoken-form choices where the
skeleton allows a choice, and (for read-backs) the extra fields it reads back; it never writes a value, event ID or any
frozen field. Python renders every value (NRIC/FIN, date of birth, amounts, quantities, currencies, direction, order type,
product, fee, limit price) and enforces the contract: exact event coverage, placeholder rules, forbidden digits/magnitude
words/percentages/currency names/product names in the template wording, required spoken forms, read-back rules with derived
ORDER constraints and cycle rejection, and the invariant that removing the G4 additions reproduces the canonical skeleton
bytes (`serialize_json(project_check_plan_to_skeleton(plan)) == skeleton bytes`). G4 acceptance is STRUCTURAL: whether the
wording really establishes a compliance behaviour is G5's semantic verification, not G4's. G4 refuses to run unless G0-G2B
validate and G3 is complete and valid; the plan records the G3 config hash, so a reopened G3 config makes it stale.

```powershell
python scripts/generate_check_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run [--call-id CALL_010] [--print-request]
python scripts/generate_check_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --sync --call-id CALL_010   # ONE paid live request
python scripts/generate_check_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --submit [--call-id ...]       # paid Batch
python scripts/generate_check_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --collect | --status | --validate [--require-complete] | --check
```

The CLI, modifiers, retry budgets, `SUBMISSION_UNKNOWN` recovery and audit layout are identical to G3 (audit under
`.llm_audit/<dataset_id>/G4/`; provenance and record under `authoring/check_plan_provenance/` and
`authoring/check_plan_record.json`).

### Upstream versions, stale responses and dependency status (G3/G4)

Accepted G3 and G4 outputs are versioned through the provenance `accepted_history` (a `--reopen --reason` keeps the
superseded version). Because a G4 plan depends on a G3 config, three rules keep the pair consistent without ever
regenerating accepted data:

- **Historical reproduction.** `--validate`/`--check` rebuild every G4 version (current or historical) against the G3
  version it was built from: the G3 config hash is read from that version's committed attempt summary
  (`upstream_sha256.call_config`) and the matching G3 version is reconstructed from G3's own accepted history and verified
  against its recorded `artifact_sha256`. If any link is missing or corrupted the command fails explicitly; it never falls
  back to the current inputs. Nothing under `.llm_audit/` is needed.
- **Acceptance safety.** Before a paid response is accepted, the upstream files are re-hashed from disk under a short shared
  write lock (`<dataset root>/.stage_write.lock`, keyed by the resolved dataset root, independent of `--audit-dir`, taken after
  the stage's own audit lock and never held across a provider call). A response whose recorded upstream hashes differ from the
  files now closes as `STALE_UPSTREAM`: terminal, kept for audit, budget-neutral, never resubmitted automatically (use
  `--submit`/`--sync`, or `--reopen --reason` if an older version is accepted). If the lock stays busy the response stays
  `RESPONDED` (`DEFERRED_WRITE_LOCK`) and the next `--collect` finishes it; if the upstream cannot be read or fails validation
  it stays `RESPONDED` as `BLOCKED_UPSTREAM_INVALID` with the exact error, and acceptance resumes once the upstream is repaired.
  A leftover lock file after a crash is never removed silently: confirm that no G3/G4/G5 process is active, then delete it.
- **Dependency status.** `dependency_status()` (shown by `--status` as `dependency=` and the labels `ACCEPTED_STALE` /
  `ACCEPTED_UPSTREAM_INVALID`) compares each accepted artifact's recorded upstream hashes with hashes re-read from disk, for G3
  and G4 alike. After a G3 change the old G4 plan stays historically valid but is `stale`: a later stage must not use it until
  it is regenerated with `--reopen`. `--check` then also reports the G4 record as changed until that happens, by design.

The same three rules apply to G5, whose upstream is the call spec, skeleton, G3 config, G4 plan and the five G0 files.

### G5: plan verification (gold-aware semantic verification)

G5 (`fyp_evaluation.plan_verifications`, `scripts/verify_authoring_plans.py`) checks, per call, whether the accepted G3
scenario and G4 evidence plan **together** faithfully implement the frozen G0-G2B gold. It never derives, revises or
predicts gold and it does not look at dialogue (that is G7). The verifier is `gemini-3.8-flash` (thinking `medium`, paid
**Flex** tier, `store=false`, no tools, strict JSON output, one stateless Interactions request per call); there is never a
fallback to the Standard tier or another model.

- **Brief (Python-built, deterministic).** The stable system instruction (`prompts/stage5/plan_verification.v1.md` plus the
  frozen checklist and `check_semantics.json`) and a per-call brief: the expected outcome/value and resolved pass/fail
  condition of all 20 questions, the G3 `products`/`scenario`, the assigned edge cases with their catalog entries, EVERY
  planned G4 event (speaker, planned text, values, relations) and every frozen or derived constraint. The gold is the target
  being checked, never evidence: a G3 fact is not a performed act, and array order is not dialogue order (chronology comes
  only from constraints, explicit relations and statuses). UNCLEAR is reserved for a named unresolved dependency.
- **Report and adjudication.** Gemini returns, per question, evidence (G4 `event_id` or a G3 JSON Pointer, with a verbatim
  quote or null), a rationale and MATCH/MISMATCH/UNCLEAR, plus issues (code, affected questions/assignments, evidence,
  advisory `repair_scope` G4_ONLY/G3_PLUS_G4/REVIEW). Python rejects any report that breaks a rule (20 questions in
  checklist order, two-way issue linkage, severity consistency, resolvable references, quotes grounded after a narrow
  normalisation of curly quotes, ellipses, dashes and whitespace - the exact source substring is stored, and an ambiguous
  match is rejected - and a quoted own evidence event for every PRESENCE MATCH). Python then derives the verdict: any
  MISMATCH -> `INVALID`, otherwise any UNCLEAR -> `UNCLEAR`, otherwise `VALID`, and the recommended repair.
- **Provider schema limits (found with the real API).** Gemini rejects the response schema with an uninformative HTTP 400 when
  `minItems`/`maxItems` are also set on arrays nested inside the bounded `checks` array. The provider-facing schema therefore
  bounds only `checks` (exactly 20); the evidence (at most 6) and issue (at most 20) limits are enforced by Python.
- **Completed versus VALID.** A structurally valid report is ACCEPTED whatever its verdict: INVALID and UNCLEAR are completed
  verifications, stored like VALID ones and never retried automatically. Only a current (`dependency=current`) VALID report
  that is also intact makes a call eligible for G6 (`g6_eligibility()`, shown by `--status`). Intact means the same per-call
  validation `--validate` runs passes: schema-valid accepted provenance that agrees with the artifact, matching artifact and
  response hashes, a rebuild against the exact G3/G4 versions, and no stale upstream. An orphan or edited report is never
  eligible (it fails closed and its `verdict` shows as `None`). `--status` also prints observational usage (provider sends,
  capacity retries, tokens, estimated cost, completed/outstanding calls); it is an estimate, not a provider balance, and
  never limits a run. Repair scopes are advice for a future controller;
  G5 never regenerates G3 or G4. Re-verification of an accepted report is explicit (`--reopen --reason`) and keeps history;
  a historical report is rebuilt against the exact G3/G4 versions it verified.
- **Retries and recovery.** The application is the only retry owner (the SDK's automatic retries are switched off and a
  test proves one HTTP request per send). HTTP 429/503 capacity rejections are resent with identical bytes inside the same
  attempt (up to 4 sends, ~30/90/270 s backoff); persistent capacity stops the run. A timeout, connection loss or other
  5xx is AMBIGUOUS (possibly billed): the attempt closes as an ambiguous `TRANSPORT_FAILED`, the run stops, and the call is
  blocked (`AWAITING_ACKNOWLEDGEMENT`) until `--run --call-id X --acknowledge-possible-duplicate`. A provider echo of a tier
  other than `flex` (or another model) is fatal and stops the run, also when found while recovering a stored response; a
  missing tier echo is accepted and counted as `tier_not_echoed`. Malformed or truncated reports are REJECTED (3 per call,
  retried only with `--retry-rejected`). `--run` always recovers first, under the G5 lock and before eligibility: a stored
  response is adjudicated without resending, and an attempt interrupted mid-request becomes ambiguous.

```powershell
python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run [--call-id CALL_021] [--print-request]
python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --run --call-id CALL_021   # paid Flex request(s)
python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --run [--limit N]          # all eligible calls
python scripts/verify_authoring_plans.py --dataset-root datasets/advisory_execution_pilot_v1 --status | --validate [--require-complete] | --check
python scripts/build_g5_controls.py --dataset-root datasets/advisory_execution_pilot_v1 --out ../g5_controls             # offline control copies
```

Credentials: `GEMINI_API_KEY` in the operating-system environment or in `.env` (the OS wins); offline modes and tests need
none. Outputs: `authoring/plan_verifications/CALL_XXX.json` (the report with Python-derived verdict), the committed
`authoring/plan_verification_provenance/CALL_XXX.json` (attempt summaries with requested and observed tier, usage and cost,
accepted response, history) and `authoring/plan_verification_record.json`; operational state under
`.llm_audit/<dataset_id>/G5/`. G5 never writes G3/G4 files. The control builder writes three isolated copies (unchanged,
an omission leak, a contradictory final instruction) through G4's own acceptance path for smoke-testing the verifier.

## G6 dialogue generation

G6 turns each call's accepted G3 scenario and G4 evidence plan into one EMPLOYEE/CLIENT dialogue
(`fyp_evaluation.dialogues`, `scripts/generate_dialogues.py`). It uses `gpt-6.1-sol` (medium reasoning, Responses Structured
Outputs). Bulk generation uses **Batch** (`--submit`, `--collect`); pilots may use one synchronous request per call (`--sync --call-id X`).
Both send the byte-identical model-facing request and go through the same validation and acceptance; only the transport envelope,
attempt record (`transport` is `batch` or `sync`) and price (sync is billed at the standard rate) differ. Synchronous G6 requests turn the
SDK's automatic retries off and use a 30-minute timeout: an unknown outcome (timeout, lost connection) is recorded as a possibly billed,
ambiguous attempt and the call is blocked until `--sync --acknowledge-duplicate-risk --call-id X`; nothing is ever re-sent automatically.
G6 acceptance means the dialogue is structurally valid and ready for G7; it is never semantic verification.

- **Evidence is inserted by Python.** The model writes whole-token markers such as `[[EV_013]]`; Python substitutes each event's
  exact G4 `planned_text`, assigns zero-based contiguous `turn_id` values and derives every span (`text[start:end]` of the turn
  equals the planned text). The model outputs only `turns[{speaker, text}]`. Nothing is copied or paraphrased by the model, so
  false-start dashes, ellipses and spelled identifiers are preserved by construction.
- **Structural rules** (REJECTED on failure): schema; every event exactly once, in a turn by its G4 speaker; strict EMPLOYEE/CLIENT
  alternation; every frozen and derived ORDER, SAME_TURN, NOT_SAME_TURN and MIN_INTERVENING_TURNS constraint; no planned text repeated
  outside its marker (a text contained in another event, as in CALL_017, is allowed); no unplanned K01/K02 value; length inside the band.
  Other value surfaces outside events are recorded as `review_notes` for G7, not rejected.
- **Brief.** Only harmless background (names, setting, holding context, final product name and category, discussion topics), the
  events, the placement constraints and the absence conditions. Speakable compliance content from G3 (risks, fees, recommendation
  basis, red flags, objectives, loss attitude) is never shown. That restriction is a G6 authoring policy, not a frozen G0 rule.
- **Length policy `g6-length-v1`.** Estimated from words at 160 wpm, measured on the repository's own Polly renders (150-169 wpm).
  SHORT accepts 510-750 words, LONG 5,103-9,000, so each band stays inside the frozen duration profile for any rate from 150 to 170
  wpm. The policy is recorded in every attempt's `params` and rebuilt from there, so a later recalibration never changes an accepted
  dialogue. `estimated_seconds` is an estimate, never a guaranteed audio duration.
- **Call intents and the recommendation.** The brief carries `call.call_intents` unchanged from the call spec and uses its array order as
  the intended segment order (every call is ADVISORY then EXECUTION): the employee's recommendation comes before the final
  confirmation/commitment, while preliminary client product mentions may come first. G1 requires the array to be the call's unique
  intents in *vocabulary order*, so for this pilot the order is canonical, not independently selected per call; reading it as segment
  order is a downstream authoring policy that may be revised if a future release varies the order. The frozen G0-G2B files are untouched.
  The brief also carries `planned_recommendation_event_ids`, derived from the C07 plan structure (no keywords, no model call). The 43 C07
  PASS calls have a planned recommendation event; the 7 C07 FAIL calls (an absence check with no event) have an empty list.
- **The unplanned recommendation (C07 FAIL).** When ADVISORY is an intent and the plan has no planned recommendation event, the brief
  also carries `advisory_recommendation`, derived by Python from frozen inputs only (`dialogues/advisory.py`; no call IDs, keywords or extra
  model call): `product_name` (the G3 FINAL product), `direction` (BUY/SELL from gold T05, only when EXECUTION is also an intent),
  `place_after_event_ids` (events carrying an edge-case assignment that targets the product T01 or direction T05: reference selection,
  corrections, replacement; this is what puts CALL_031's recommendation after the client's product-reference selection, while the quantity
  references of CALL_001/036 settle nothing about the product), `place_before_event_ids` (the final confirmation exchange, C08) and
  `keep_apart_from_event_ids` (the C01 objective and C06 loss-tolerance evidence). The model writes ONE clear employee recommendation of that
  action on that product, with no concrete rationale (no feature, benefit, performance, suitability or link to the client's goals,
  situation or risk appetite), no order detail and no copy of any planned text, placed after the first list and before the second, not in
  or immediately after the third, then the client's next turn opens with a mild acknowledgement ("Okay, noted."), never an instruction,
  commitment, confirmation or "I understand". Earlier placement and ending the employee turn without a planned question are preferences.
  The prompt's rule that supporting speech never states a product choice has exactly this one carve-out. G6 acceptance stays
  structural: naming the product outside the events is a `review_notes` entry, never a rejection. `preflight_errors` also reports
  contradictory placement requirements offline (a settled-first event ordered after the final confirmation, or a shared turn that leaves
  no employee slot clear of objective/loss evidence) before any request. `scripts/lint_advisory.py` is an advisory-only lint of the
  accepted dialogues (missing product name, rationale-like wording such as "because"/"since", order detail, placement, adjacency, a client
  reaction that is more than a mild acknowledgement); its signals are hints for a human and never fail anything. Meaning and
  naturalness are judged by G7. Changing the shared prompt changes the request bytes of all 50 calls (the other 43 have no
  `advisory_recommendation`), so only their artifacts are required to stay untouched, never their requests.
- **G5 gate.** A call is eligible only while its G5 report is current, intact and VALID (`g6_eligibility()`), per call. At acceptance,
  under the dataset write lock: unreadable inputs hold the response (resumable `UPSTREAM_INVALID`); changed fingerprints, a
  confirmed-deleted G5 report included, close it as terminal `STALE_UPSTREAM` (raw response kept, never resent); changed G6
  settings do the same; a failing G5 approval with unchanged fingerprints is a resumable hold. Historical dialogues are rebuilt
  against the exact G3/G4 versions and the verified G5 approval they recorded. Freeze G3-G5 authoring while G6 runs.
- **Outputs.** `authoring/dialogues/CALL_XXX.json` (turns, `event_alignment`, length metrics), committed
  `authoring/dialogue_provenance/CALL_XXX.json` and `authoring/dialogue_record.json`; operational state under
  `.llm_audit/<dataset_id>/G6/`. Spoken-turn consumers read only `turns[].turn_id/speaker/text`. The legacy ground-truth compiler
  does not accept this artifact; the G9 compiler (below) translates it.

```powershell
python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run [--call-id CALL_027] [--print-request]
python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --submit [--call-id ...]    # paid Batch
python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --collect
python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --status | --validate [--require-complete] | --check
```

Credentials: `OPENAI_API_KEY` in the operating-system environment or `.env`; offline modes and tests need none.

## G7 dialogue verification (gold-aware semantic verification of the real dialogue)

G7 (`fyp_evaluation.dialogue_verifications`, `scripts/verify_dialogues.py`) checks, per call, whether the accepted G6 dialogue,
read as a whole, implements the frozen gold and genuinely realises every assigned edge case. It never generates, repairs or changes
a dialogue or gold. The verifier is the G5 stack with **high** thinking: `gemini-3.8-flash`, thinking `high`, paid **Flex** tier,
`store=false`, no tools, strict JSON output, one stateless synchronous Interactions request per call (LONG calls included, no chunking),
`max_output_tokens` 32768, timeout 900 s. There is no Standard-tier or other-model fallback and no Batch path. The Gemini send,
capacity-retry, ambiguity and recovery loop is the one G5 proved, shared through `llm/gemini_runner.py` (`GeminiFlexMixin`).

- **G7A preflight (Python, before any paid request).** A call is eligible only if `DialogueRunner.g7_eligibility()` passes: the G6
  dialogue file exists and is readable (a read that fails with anything but `FileNotFoundError` is *unreadable*, never "deleted"),
  committed G6 provenance records an accepted attempt whose artifact hash equals these exact bytes (`_validate_call` alone is not
  proof, it is silent when both files are absent), the shared per-call validation passes (rebuild under the exact G3/G4/G5 versions,
  markers, spans, speakers, constraints, recorded length policy), the dialogue is current, and its G5 approval is still current,
  intact and VALID. An ineligible call receives no request and `--dry-run`/`--run` report why.
- **Brief.** The stable system instruction (`prompts/stage7/dialogue_verification.v1.md`, the frozen checklist, `check_semantics.json`
  and the release-local `normalization.json` profiles) and, per call, G5's brief (expected gold for all 20 questions, resolved pass/fail
  conditions, G3 scenario, assigned edge cases with catalog semantics, every planned event with relations and constraints) plus the
  **entire actual dialogue** (every integer `turn_id`, speaker and text) and the turn each event landed in. No hashes, provenance,
  provider envelopes, offsets or G5 verdicts are sent. Gold is the target being checked, never evidence; a planned compliance FAIL is
  not an error and MATCH means the dialogue implements the expected gold, omissions and deliberate violations included.
- **FAIL semantics and advisory realization.** The verifier follows each compliance row's own `target.must` / `evidence_mode` and the
  checklist's `requirement_type`: an OBLIGATION FAIL (C01-C08) is an omission or a deliberately inadequate act per the frozen
  `fail_condition`, a PROHIBITION FAIL (C09-C12) requires the AUTHORED VIOLATION to be present (never "every FAIL is an absence").
  It also verifies actual advisory and execution realization: `call_intents` order is the intended segment order (ADVISORY then
  EXECUTION; see the G6 section for why this is a downstream authoring policy); the employee's recommendation must precede the final confirmation/commitment (preliminary client product mentions may come first); a
  C07 FAIL keeps a clear recommendation of the frozen product and action without a concrete rationale. For those calls the G7 brief
  carries `advisory_expectation`, the same Python-derived data as the G6 brief plus the turn of every named event, and G7 verifies by
  meaning: a recommendation that names neither product nor action, or none at all, is a C07 `MISSING_SUPPORT` MISMATCH; a concrete
  rationale is an `OMISSION_LEAKAGE` MISMATCH (adjacency alone is not one, and "as"/"since" are not decisive); naming the product before
  the client's reference selection or correction has settled it is an `EDGE_CASE` MISMATCH; a client reaction that is an instruction,
  confirmation or understanding statement is a MISMATCH on C08/C03. Tests pin the quadrant mapping for all 50 briefs and the prompt wording.
- **Report.** Per checklist question (20, checklist order) and per frozen edge-case assignment (frozen order, an empty array when none):
  evidence (`turn_id` + verbatim quote), a rationale, MATCH/MISMATCH/UNCLEAR; `issues` (codes COMPLIANCE_MEANING, OMISSION_LEAKAGE,
  EXTRACTION_STATE, SCENARIO_CONTRADICTION, EDGE_CASE, CROSS_CHECK_INTERFERENCE, MISSING_SUPPORT; advisory `repair_scope`
  G6_ONLY/UPSTREAM_REVIEW/REVIEW); and `quality_findings` (naturalness, NOTE/SERIOUS). The model never states a verdict or gold.
- **Python validation (REJECTED on any failure).** Schema; exact 20 checks and exact assignments in order; two-way issue linkage and
  severity agreement; known turns; every quote grounded in the cited turn after G5's narrow normalisation (the exact source substring is
  stored together with **every** matching occurrence span in that turn; a quote matching different exact substrings is rejected);
  size limits. Citation rules are mechanical: a PRESENCE check marked MATCH cites its own evidence event, an edge case marked MATCH cites an
  event carrying it, **every MISMATCH issue except `MISSING_SUPPORT` cites the actual speech it rests on** (an explicit contrary statement
  such as a later BUY against a SELL gold is `EXTRACTION_STATE`, never `MISSING_SUPPORT`), every quality finding cites its turn; UNCLEAR
  issues, `MISSING_SUPPORT` and omissions may cite nothing. Python derives `event_ids` of a citation from the G6 alignment.
- **Verdict, quality and G8/G9.** Python derives the verdict over check rows, edge-case rows and issues: any MISMATCH -> `INVALID`,
  otherwise any UNCLEAR -> `UNCLEAR`, otherwise `VALID`; an edge-case MISMATCH or a whole-dialogue issue therefore decides the verdict even
  when all 20 checks MATCH. `quality_review` is `REVIEW_REQUIRED` when any quality finding is SERIOUS, else `CLEAR`. It is **advisory**:
  it never changes the verdict and never gates G8 or G9, but it is always reported (`--status`, the record): VALID does not mean quality-cleared.
  `g8_eligibility()` (eligible for the G8 render, and so for G9) = current, intact (per-call validation passes) and VALID. INVALID and UNCLEAR are completed verifications: never
  retried automatically; an accepted report is skipped on resume and re-verified only with `--reopen --reason` (history is kept; this is also
  how a repaired G6 dialogue is re-verified, since its stale report reads "already accepted").
- **Acceptance ordering** (under the dataset write lock): unreadable inputs hold the response (resumable `UPSTREAM_INVALID`); changed
  fingerprints (G0-G6 and G5 report, a confirmed-deleted dialogue included) close it as terminal `STALE_UPSTREAM` (raw response kept, never
  resent, no structural budget used); a changed prompt, response schema or settings, re-hashed from the files now, does the same; a failing
  preflight with unchanged fingerprints is a resumable hold. Historical reports are rebuilt against the exact G3/G4/G5 versions and the
  exact verified G6 dialogue they recorded (never the current one); missing or corrupted history fails explicitly.
- **Provider limits.** As for G5: nested array bounds make Gemini answer HTTP 400, so only `checks` is bounded (20); the other limits (6
  citations, 20 issues, 10 quality findings, 700/1200/400 characters) are enforced by Python. A truncated or malformed report is REJECTED
  (3 per call, `--retry-rejected`). 429/503 are resent inside one attempt (up to 4 sends, same bytes); a timeout is ambiguous and blocks the
  call until `--acknowledge-possible-duplicate`.

```powershell
python scripts/verify_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --dry-run [--call-id CALL_016] [--print-request]
python scripts/verify_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --run --call-id CALL_016        # paid Flex request(s)
python scripts/verify_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --status | --validate [--require-complete] | --check
python scripts/build_g7_controls.py --dataset-root datasets/advisory_execution_pilot_v1 --out .llm_audit/g7_controls     # offline control copies
```

Credentials: `GEMINI_API_KEY` in the operating-system environment or `.env`; offline modes and tests need none. `--dry-run` is offline,
credential-free and writes nothing; it prints the request hash, characters, UTF-8 bytes and **approximate** tokens per call and a rough
dated Flex cost estimate (not a promise). `--status` separates lifecycle, verdict, quality review, G8 eligibility, provider sends, usage and
estimated cost. Outputs: `authoring/dialogue_verifications/CALL_XXX.json`, committed
`authoring/dialogue_verification_provenance/CALL_XXX.json`, `authoring/dialogue_verification_record.json`; operational state under
`.llm_audit/<dataset_id>/G7/`. The control builder writes three isolated copies of CALL_016 (unchanged; an employee fee disclosure that
leaks the C04 omission; a client BUY instruction after the final confirmation) by reopening G6 through its own acceptance path.

## G8: SSML conversion and Amazon Polly rendering (verified dialogues only)

G8 (`fyp_evaluation.rendering`, `scripts/render_dialogues.py`) turns each **G7-verified G6 dialogue** into per-turn SSML, Polly audio and
`timings.json`. It reads `authoring/dialogues/CALL_XXX.json` directly (the same zero-based `turn_id`s G9 reuses), never
`ground_truth.json`, so it needs no G9 output, and it never writes gold. A call is processed only when its G7 report is current, intact
and VALID (`DialogueVerificationRunner.g8_eligibility`); every other call is listed with its reason and never sent.

```powershell
python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --preview                 # eligibility, characters, approximate cost; writes nothing
python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --ssml                    # offline: segments/<turn_id>.ssml + ssml_manifest.json
python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --check                   # offline: renders vs the CURRENT verified dialogues
python scripts/render_dialogues.py --dataset-root datasets/advisory_execution_pilot_v1 --execute --call-id CALL_001   # BILLABLE Polly requests
```

- **SSML.** Each turn becomes `<speak>...</speak>` with XML-escaped text. Upstream already fixes how values are WRITTEN and spoken (release
  `normalization.json` `spoken_forms` and the edge-case catalog): identifiers character by character with digits as words ("S six five ...
  nine A", "oh" and "double"/"triple" only under their edge cases), dates as "13 March 1999" or an ordinal, currencies by natural name, and
  financial shorthand (`500k`, `10m`, `50k`, `75k`) only under EC_FINANCIAL_SHORTHAND, where the audio must keep the shorthand. The text is
  already in spoken form, so SSML only adds `ssml-v3` pronunciation hints where a neural voice would misread a token, and every hint is a
  `<sub alias>` from one **closed table** (no `say-as`, `phoneme`, `break`, `prosody` or any other tag):
  - the prefix and suffix letters of a spoken identifier use letter names ("A" -> "ay", "S" -> "ess"); a lone "A" would otherwise be read as
    the article. "oh", "double" and "triple" in between are left exactly as written;
  - NRIC, FIN, ETF, US and the release's currency codes are spelled by letter names ("NRIC" -> "en ar eye see"; a test keeps the list in step
    with the release's allowed currencies);
  - shorthand keeps its suffix: `500k` -> "five hundred kay", `10m` -> "ten em", `50k` -> "fifty kay", `75k` -> "seventy-five kay". The suffix is
    never expanded to "thousand" or "million". (A magnitude expression written in words, such as "1.2 million" under
    EC_LARGE_NUMBER_EXPRESSION, is plain text and is left alone.) The letter names and these shorthand readings are a project choice, not
    something upstream spells out, and have to be confirmed by listening.

  Canonical dialogue text is never changed. `assert_faithful` checks BOTH that the visible text of every SSML file is exactly the canonical turn
  text AND that every hint is the one the table prescribes; an arbitrary alias (even on the right text) or any unsupported tag or attribute is
  rejected. Dates, ordinals and everything else are left to the voice. `ssml_manifest.json` records the dialogue hash, the pronunciation version
  and a hash per turn; `--check` validates the SSML even before any audio exists (missing, stale, hand-edited or old-version SSML fails).
- **Polly.** `--execute` first checks the SSML files on disk against a fresh build and sends **those files**, one request per turn
  (neural engine, `ap-southeast-1`, Jasmine for EMPLOYEE and Matthew for CLIENT by default; PCM 16 kHz), stitches the PCM with 400 ms of
  silence between turns, encodes `audio.mp3` and `segments/<turn_id>.mp3` with FFmpeg (`imageio-ffmpeg`), and derives the per-turn
  `timings.json` (the existing `timings.schema.json` shape). Polly never assigns or changes turn IDs.
- **Provenance.** `render_manifest.json` per render records the dialogue hash, the G7 report hash, the SSML manifest hash, the timings
  hash, the audio hash, the settings, the billed-character estimate and the time. Audio is git-ignored; SSML, timings and manifests are
  versioned. `--check` fails a render whose dialogue changed since (stale), whose timings do not cover every turn exactly once, or whose
  audio does not match its recorded hash.
- **Spending.** `--preview` prints the characters and an approximate list-price cost (an estimate, not a quote; the free tier is not
  modelled). `--execute` refuses while any selected call is ineligible, never replaces an existing render without `--overwrite`
  (`--skip-existing` leaves complete current renders alone), stops at the first failure and lists the calls that finished. Each call is
  written only after all its turns were synthesized, so a failure never leaves a half render. Credentials come from the standard AWS chain
  (`--profile` optional) and are never read, printed or stored. The rendering dependencies are the `render` extra
  (`python -m pip install -e ".[render]"`). A G6 reopen makes the G7 report stale, which makes the call ineligible until it is
  re-verified and re-rendered.

## G9: canonical ground truth (compiled last, from the G8 timings)

G9 (`fyp_evaluation.ground_truths`, `scripts/compile_ground_truth.py`) is deterministic: no LLM, no network. It combines the frozen gold
(G2A call spec), the accepted G4 check plan and the accepted G6 dialogue into the three files each call needs, written to
`datasets/<release>/calls/CALL_XXX/`:

- `ground_truth.json`: the 20 expected answers (copied from the frozen gold, never derived from the dialogue) plus ACTUAL evidence
  from the dialogue: `gold_evidence_turns`, exact spans (`gold_evidence`: event, turn, speaker, character offsets, text) and
  `gold_evidence_text`; `final_turn_id` and a `value_history` (every turn that states a value, with its status such as
  SUPERSEDED / FINAL, readback errors included) for extraction questions; `challenge_tags` from the assigned edge cases; the full
  dialogue turns; source hashes; and a `status` block. Absence questions have empty evidence by design. The check plan holds
  PLANNED evidence; this file holds what is actually in the accepted dialogue, and the compiler refuses unless each span equals the
  planned text and is spoken by the planned speaker.
- `transcript.json` and `speaker_roles.json`: the AWS-Transcribe-shaped pair (EMPLOYEE = spk_0 = RM, CLIENT = spk_1 = Customer),
  from the same converter as the legacy pilot; segment times are the placeholder `"0.0"` in the provisional compile and the G8
  audio-derived times in the final one.

`authoring/ground_truth_record.json` hashes every compiled file and `dataset_manifest.json` lists the call IDs. `--write` refuses unless
G6 is complete, valid and current, and refuses to replace differing files without `--overwrite`; `--check` rebuilds in memory and
compares byte for byte. Outputs are therefore always regenerable and never hand-edited.

**Two modes.** The compile is PROVISIONAL (`status.g7_dialogue_verification` PENDING, `status.timings` PLACEHOLDER_ZERO, every transcript
time the placeholder `"0.0"`) unless it is given a render, in which case it is FINAL:

```powershell
python scripts/compile_ground_truth.py --dataset-root datasets/advisory_execution_pilot_v1 --write --overwrite --render-id clean_voice_A   # FINAL
python scripts/compile_ground_truth.py --dataset-root datasets/advisory_execution_pilot_v1 --check                                          # follows the record
```

The order of work, and where this release stands (the FINAL compile of render `clean_voice_A` has been written over the provisional files):

1. **G7** dialogue verification (gold-aware, report-only; production run 50/50 VALID). Anything it flags is fixed by regenerating the
   affected G6 dialogue (or upstream authoring), never by editing gold or the compiled files.
2. **G8 render** (above): SSML, Polly audio and `timings.json`.
3. **FINAL G9 compile** (`--render-id`). It refuses unless every call has a G7 report that is current, intact and VALID **and** a complete,
   current G8 render: `render_manifest.json` bound to the current accepted dialogue and G7 report, SSML equal to a fresh build, `timings.json`
   covering every dialogue turn exactly once and in order with non-overlapping intervals, and (when present) audio matching its recorded
   hash. Reopening G6 therefore invalidates the render as well as G7. The audio-derived per-turn times (fixed three-decimal strings, the
   400 ms inter-turn silence included) enter **only** `transcript.json`. `ground_truth.json` gets `status` VERIFIED / AUDIO_DERIVED and a
   `timing` block (`render_id`, `timings_sha256`, `render_manifest_sha256`, `audio_sha256`, `g7_report_sha256`, `duration_seconds`);
   `ground_truth_record.json` records the render ID. The 20 expected answers, evidence turns, spans, value histories, challenge tags and
   the dialogue are byte-identical to the provisional compile (a test pins this), and `speaker_roles.json` is unchanged. A PROVISIONAL
   `--write` over FINAL outputs is refused; `--check` and the exporter follow the render the record names, so they need no flag.
4. Re-export to fyp-ai as a new version of the same snapshot folder (NOT done yet; held until requested).

The exporter (`scripts/export_dataset.py`, see EXPORTING.md) understands G9 releases (it uses the release's own `normalization.json`,
re-compiles G9 in memory, following the render a FINAL record names, and refuses any call file that differs) and records `dataset_status`
and `timing_status` (`placeholder_zero` or `audio_derived`) in the snapshot manifest.

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
7. Generate per-turn SSML from the verified dialogue's reference text (turn IDs are the dialogue's zero-based positions) and
   separate pronunciation settings.
8. Render Polly segments, stitch them, and write timings for that specific
   render. No ground truth is read or changed in this step.
9. The Python compiler validates all inputs (including that the timings cover every turn of the current verified dialogue), assigns
   turn IDs, resolves evidence deterministically and stitches in the timings, then writes the **final** canonical `ground_truth.json`,
   `transcript.json` and `speaker_roles.json`. Gold answers are identical to the provisional compile; only the timing fields and
   `status` differ.
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

For the advisory/execution releases (G6 onwards) rendering happens **before** the final ground-truth compile: the render reads the
G7-verified G6 dialogue, and the final G9 compile then reads the render's `timings.json`. The render therefore never depends on a
compiled `ground_truth.json`, and ground truth is never rewritten to fit a render. The `render_polly.py` description below
(reading `ground_truth.json`) is the legacy-pilot workflow.

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
python scripts/convert_ssml.py --dataset-root datasets/legacy_pilot_v1 --all
python scripts/convert_ssml.py --dataset-root datasets/legacy_pilot_v1 --call CALL_001
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
`datasets/legacy_pilot_v1/calls/CALL_XXX/renders/clean_voice_A/`. It does not change
canonical ground truth or the prepared transcripts. Word timings are not
generated; listen to each render and check the spoken identifiers, amounts and
dates before using the audio for ASR evaluation.

The committed `clean_voice_A/timings.json` files describe the original
`clean_voice_A/audio.mp3` files. Keep those exact MP3 files in separate storage
for evaluation. A new Polly synthesis can have different durations and must use
its own timing file and render ID.

To save the exact SSML sent for each turn without making another Polly request,
run `python scripts/export_ssml.py --dataset-root datasets/legacy_pilot_v1 --all`.
Each render then has
`segments/<turn_id>.ssml` plus `ssml_manifest.json`, which maps turn IDs to
EMPLOYEE or CLIENT. The SSML wraps XML-escaped canonical text in `<speak>`;
the voice is selected by the Polly API request, not embedded in the SSML.

Install the rendering dependencies and configure an AWS profile with permission
to call `polly:DescribeVoices` and `polly:SynthesizeSpeech`. No AWS credentials
belong in this repository. Then run from this directory:

```powershell
python -m pip install -e ".[render]"
python scripts/render_polly.py --dataset-root datasets/legacy_pilot_v1 --all
python scripts/render_polly.py --dataset-root datasets/legacy_pilot_v1 --all --render-id clean_voice_B --execute
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
datasets/legacy_pilot_v1/calls/CALL_XXX/
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

The legacy pilot's prepared transcript files still carry the documented temporary placeholder
`"start_time": "0.0", "end_time": "0.0"`. These are not measurements, and `build_transcript_artifacts(..., timings=...)` for that legacy path
still raises `NotImplementedError`. For the advisory/execution release the shared converter
(`transcript_artifacts_from_dialogue(..., timings=...)`) regenerates `transcript.json` with the real audio-derived per-turn times in the final
G9 compile, after the G8 render, rather than patching a placeholder file in place.

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
python scripts/build_transcripts.py --dataset-root datasets/legacy_pilot_v1 --call CALL_001
python scripts/build_transcripts.py --dataset-root datasets/legacy_pilot_v1 --all
python scripts/build_transcripts.py --dataset-root datasets/legacy_pilot_v1 --all --check
python scripts/build_transcripts.py --dataset-root datasets/legacy_pilot_v1 --call CALL_001 --overwrite
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
fyp-validate json check_plan datasets/legacy_pilot_v1/authoring/check_plans/CALL_001.json
fyp-validate checklist datasets/legacy_pilot_v1/checklist.json configs/normalization.json
fyp-validate answers check_plan datasets/legacy_pilot_v1/authoring/check_plans/CALL_001.json datasets/legacy_pilot_v1/checklist.json configs/normalization.json
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
  checklist adjacent to that dataset's `authoring/` directory
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
python -m fyp_evaluation.validation matrix datasets/legacy_pilot_v1/authoring/coverage_matrix.csv datasets/legacy_pilot_v1/checklist.json configs/normalization.json
python -m fyp_evaluation.validation scaffold-call-configs datasets/legacy_pilot_v1/authoring/coverage_matrix.csv datasets/legacy_pilot_v1/authoring/call_configs datasets/legacy_pilot_v1/checklist.json configs/normalization.json
python -m fyp_evaluation.validation call-config datasets/legacy_pilot_v1/authoring/call_configs/CALL_001.json datasets/legacy_pilot_v1/authoring/coverage_matrix.csv datasets/legacy_pilot_v1/checklist.json configs/normalization.json
python -m fyp_evaluation.validation call-config datasets/legacy_pilot_v1/authoring/call_configs/CALL_001.json datasets/legacy_pilot_v1/authoring/coverage_matrix.csv datasets/legacy_pilot_v1/checklist.json configs/normalization.json --complete
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

`datasets/legacy_pilot_v1/checklist.json`, `schemas/checklist.schema.json` and
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

The legacy pilot's authoritative question text is in
`datasets/legacy_pilot_v1/checklist.json`.
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

`datasets/legacy_pilot_v1/authoring/coverage_matrix.csv` contains the predetermined answers for
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
