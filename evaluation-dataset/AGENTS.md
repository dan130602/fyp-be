# Evaluation dataset agent instructions

Scope: this self-contained `fyp-be/evaluation-dataset` project only. Do not
modify backend, infrastructure, LangGraph or unrelated files for dataset work.

Read `README.md`, applicable schemas and `configs/normalization.json` before edits.
The authoritative checklist contains `id`, `type`, `text`, plus
`requirement_type` (OBLIGATION/PROHIBITION) only for COMPLIANCE or `field` only
for CLIENT_KYC/TRANSACTION. Reject incompatible fields and extra properties.
Derive verdict versus extraction from `type`; do not maintain `answer_type`
or a question metadata file. Reusable field-to-rule mappings belong in
`configs/normalization.json`. Unmapped extraction fields fail validation.
Validate rule references and exact downstream checklist-ID/type coverage.
Derive LangGraph input as `{id: item.id, question: item.text}` preserving IDs
and order. Do not store a second checklist or change production LangGraph.

Preserve the workflow: approved checklist -> approved coverage matrix -> call
config -> predetermined check plan -> dialogue -> independent verification ->
deterministic Python compilation -> SSML -> audio/timings -> evaluation.
Never infer, revise or reinterpret gold answers from generated dialogue.
Correct dialogue that violates the plan. Changes to gold require explicit
upstream revision and reapproval, followed by regeneration.

An LLM must never directly author or overwrite final canonical ground truth.
Only a future deterministic Python compiler may produce it from validated,
approved and independently verified inputs. Structural validation alone does
not establish approval, correct semantics, chronology or provenance.

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
gating or depends_on logic in the pilot.

Do not generate real checklist questions, matrix rows, scenarios, check plans,
dialogues or dataset artifacts without user approval. Test fixtures must be
clearly artificial and stay under `tests/`. Do not add LLM API, AWS, Polly,
Transcribe, LangGraph, semantic verification or scoring implementations during
this initial scaffold task.

Keep schema changes versioned and synchronized with validation, tests and docs.
Run `python -m pytest -q` and the schema-definition validator after changing
contracts. Report unresolved contract decisions instead of silently guessing.
