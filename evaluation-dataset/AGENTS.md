# Evaluation dataset agent instructions

Scope: this self-contained `fyp-be/evaluation-dataset` project only. Do not
modify backend, infrastructure, LangGraph or unrelated files for dataset work.

Read `README.md`, applicable schemas and `configs/normalization.json` before edits.
Each dataset release has one authoritative checklist containing `id`, `type`, `question`, plus
`requirement_type` (OBLIGATION/PROHIBITION) only for COMPLIANCE or `field` only
for CLIENT_KYC/TRANSACTION. `question` was renamed from `text` at checklist
schema revision 0.3.0 so the source already matches the LangGraph id/question
shape; `text` is now the retired/out-of-scope name. Reject incompatible fields and extra properties.
Derive verdict versus extraction from `type`; do not maintain `answer_type`
or a question metadata file. The legacy pilot's field-to-rule mappings belong in
`configs/normalization.json`. Unmapped extraction fields fail validation.

advisory_execution_pilot_v1 (G0 Stage 0): its five contracts are `checklist.json`,
`check_semantics.json`, `normalization.json`, `edge_case_catalog.json` and
`authoring/generation_plan.json`, validated by `validation.stage0` against
`schemas/stage0/`. Its release-local `normalization.json` owns canonical,
alias, spoken-form and comparison profiles; `check_semantics.json` owns only
semantic/final-state rules and names a profile per extraction question. Never
route this release through `configs/normalization.json`. The normalization file
is declarative: do not implement its steps until authorized. All 20 questions
apply to every call; call intents are multi-label and descriptive only; the
only hard compliance dependency is C03 PASS => C02 PASS. G1 rules: at most one
edge case per question per call (EC_LONG_DISTANCE_DEPENDENCY exempt); superseded,
distractor and initial-order values stay inside the configured domains and realism
constraints; no fee amount/rate distractor when C04 is FAIL; every other choice is
seeded, balanced and stably tie-broken. No applicability,
N/A or NOT_APPLICABLE. Pipeline names: G0 contracts, G1 deterministic planning,
G2 call_spec, G2B event_skeleton, G3 call_config, G4A check_plan_skeleton,
G4B check_plan, G5 plan verification, G6 dialogue, G7 dialogue verification,
G8 ground truth, G9 audio and evaluation. Do not implement G1 until approved.
Validate rule references and exact downstream checklist-ID/type coverage.
Derive LangGraph input as `{id: item.id, question: item.question}` preserving
IDs and order. Do not duplicate a checklist within a release or change production LangGraph.

Preserve the workflow: approved checklist -> approved coverage matrix -> call
config -> predetermined check plan -> dialogue -> independent verification ->
deterministic Python compilation -> SSML -> audio/timings -> evaluation.
Never infer, revise or reinterpret gold answers from generated dialogue.
Correct dialogue that violates the plan. Changes to gold require explicit
upstream revision and reapproval, followed by regeneration.

Legacy pilot Stage 3A: the checklist and matrix are approved and read-only. All config
expected answers must be compiled by Python from matrix cells, never authored
or retyped by AI. The answer_contract governs representation only. Reuse
authoring.coverage and authoring.call_configs; refuse existing output paths
unless overwrite is explicitly requested. Call configs retain version 0.5.0.
Each config answer requires scenario_intent. Initial scaffolds retain empty
scenario_facts and TODO_AUTHOR_SCENARIO_INTENT for every question. Do not fill
these narrative fields until separately authorized. No fact name may duplicate
a checklist extraction field. Validate configs against the matrix in scaffold
mode (complete=False); require complete=True before later check-plan authoring.
Completeness checks do not replace semantic review. Preserve all matrix values.

Stage 4 contracts: envelopes now version independently. Check plans and ground
truth are 0.6.0; approved call configs and their schema remain 0.5.0 and read-only.
Use artificial fixtures only in this tooling stage; create no real artifact.
Compile expected from matrix cells and derive evidence mode; never choose mode.
OBLIGATION PASS/FAIL -> presence/absence; PROHIBITION PASS/FAIL -> absence/presence.
Non-null extraction -> presence; only null K01 -> absence. Python owns mode,
absence scope/speaker, scaffold constants and evidence-ID allocation.
Presence has per-item speakers only. Default one item, EMPLOYEE for compliance
and CLIENT for extraction; C01/C02 exchanges can legitimately need both roles.
Use the minimum sufficient evidence. Append QID_E2, QID_E3 after QID_E1; never
renumber, reuse or delete IDs. Never fabricate a quote for an omission FAIL.
Later AI authoring is limited to planned text, absent_event, content_constraints,
forbidden_patterns and ordering_constraints. Empty pattern arrays are legitimate.
Compile Python regexes for syntax only at this stage; regexes do not prove semantic
absence. Ordering edges name existing presence checks only and must be acyclic.
Follow README reference orthography for canonical text; keep SSML/pronunciation
hints separate. The convention is chosen before generation, not a guarantee of
ASR output. Later smoke tests must not change approved gold. EMPLOYEE/CLIENT are
semantic roles, never assumed audio channels or diarization identities. Critical
tokens belong only to Stage 5 dialogue turns, not check plans or planned evidence.

An LLM must never directly author or overwrite final canonical ground truth.
Only a future deterministic Python compiler may produce it from validated,
approved and independently verified inputs. Structural validation alone does
not establish approval, correct semantics, chronology or provenance.

Dialogue turns must strictly alternate EMPLOYEE/CLIENT: never two consecutive
turns by the same speaker. Merge continued speech into one turn. Turn IDs are
positions, so this is also what makes evidence turn IDs and the fyp-ai evaluator's
turn-level scoring meaningful. `validate_speaker_alternation` enforces it in
`dialogue-plan` and `compile-ground-truth`; fix the dialogue, never the check.
Use zero-based integer turn IDs. Intermediate dialogue and predetermined plans
must not contain resolved turn IDs; Python assigns IDs after dialogue creation.
Exact planned phrases and required speakers govern positive evidence. Missing
or ambiguous matches fail; never arbitrarily choose a matching turn. Absence
checks can have empty evidence arrays/quotes and need independent verification.

Rendering must never modify canonical ground truth. Keep voices, prosody,
SSML, timestamps and word timings in rendering inputs/outputs. Use render
timing arrays with integer turn IDs. Do not fabricate audio or final gold files.
Polly only renders audio and timings; it never assigns or changes canonical IDs.

Gold compliance is PASS/FAIL. Gold extraction is a typed actual value, including
calendar-valid `DD Month YYYY` string DOBs (two-digit day, full English month,
four-digit year). K01 extracts a fictional identification number, not a name.
K01 asks for the client's NRIC or FIN number and keeps field
client_identification_number with exact-text normalization. Preserve the four
matrix identifiers exactly; they match [STFGM][0-9]{7}[A-Z] but deliberately
fail checksum validation. Never fix their checksums. Plan unambiguous CLIENT
spoken evidence and independently verify exact identifier extraction.
K01 alone permits absent extraction: CSV NOT_STATED decodes deterministically
to Python None / native JSON null, never the strings NOT_STATED or null.
CALL_005 has no identification number: plan absence for the entire dialogue
with required speaker ANY, verify neither speaker introduces a number in any
spoken form, and retain empty canonical evidence turns/text. Preserve every
other matrix answer. Null is not N/A, inapplicability or compliance FAIL.
A later scorer reports PASS/FAIL for extraction comparison and deterministically
normalizes equivalent predicted date forms without changing canonical gold.
Do not accept N/A or UNCLEAR predictions as new gold labels. No applicability
gating or depends_on logic in the legacy pilot.

Do not generate real checklist questions, matrix rows, scenarios, check plans,
dialogues or dataset artifacts without user approval. Test fixtures must be
clearly artificial and stay under `tests/`. Do not add LLM API, AWS, Polly,
Transcribe, LangGraph, semantic verification or scoring implementations during
this initial scaffold task.

Keep schema changes versioned and synchronized with validation, tests and docs.
Run `python -m pytest -q` and the schema-definition validator after changing
contracts. Report unresolved contract decisions instead of silently guessing.
