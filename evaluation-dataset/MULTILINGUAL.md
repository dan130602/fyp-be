# Multilingual release contract (`advisory_execution_multilingual_v1`)

A multilingual release keeps the gold-first G0 -> G9 pipeline and adds language profiles. Canonical gold stays
language-independent: the coverage matrix stores `T04 = 700`, never `七百`. A profile only constrains how values and
dialogue are spoken. The historical `advisory_execution_pilot_v1` release is unchanged and still reproduces byte-for-byte.

Implemented so far: G0, G1, G2A and G2B (Phase 1), resized to 250 calls (75 ENGLISH, 75 MANDARIN, 50 EN_ZH_MIXED, 50 SINGLISH), the
deterministic G4 layer for schema 2.0 (Phase 2; see "G4 (schema 2.0)" below) and, offline, the multilingual G3-G7 generations and the G9
source schema (Phase 3; see "G3-G5 (schema 2.0)" and "G6, G7 and G9 (schema 2.0)"). No paid call has been made: every multilingual
G3-G7 test runs on fake transports, and a live run is refused unless it is covered by a committed allowlist batch. TTS, STT, scoring, the
deferred 70/30 development / held-out split (175 / 75 calls) and CI are later work.

## Release shape

| | Monolingual (pilot) | Multilingual |
|---|---|---|
| `authoring/generation_plan.json` | schema 1.0 (global counts) | schema 2.0 (`language_profile_plans`, exact counts per profile); schema `schemas/stage0/generation_plan.v2.schema.json` |
| `edge_case_catalog.json`, `normalization.json` | 1.0 | 2.0 (same schema files, 2.0-only fields enforced by `validation.stage0`) |
| `language_profiles.json` | absent | required (`schemas/stage0/language_profiles.schema.json`) |
| `call_manifest.csv` | `language` | `language_profile`, `mixing_style` (`NONE` outside EN_ZH_MIXED) |
| `edge_case_assignments.csv` | as before | adds `language_variant` (`NONE` or a catalog variant ID) |
| `planning_record.json` | 1.0, global summary | 2.0, summary per language profile; hashes all six G0 files |
| call spec | 1.0 | 2.0: `metadata.language_profile`/`mixing_style`, a frozen `language_requirements` snapshot, `language_variant` per edge case, `source.language_profiles_sha256` |
| skeleton | 1.0 | 2.0: spoken forms filtered to the call's languages; `required_language` only on events a variant pins |

`validation.stage0` refuses a release that mixes contract generations, and requires `language_profiles.json` exactly when
the generation plan is 2.0.

## Language profiles

`ENGLISH` (standard English), `MANDARIN` (pure Mandarin: Latin letters only inside the rendered K01 identifier, no
Arabic digits, no codes or acronyms, approved Chinese product names), `EN_ZH_MIXED` (the client personally uses
substantive English and Mandarin with within-turn and distributed switching; the employee may use either; styles
`EN_LED`/`ZH_LED`/`BALANCED` describe intended client behaviour, not token quotas) and `SINGLISH` (client-only
colloquial Singapore English, no Chinese characters, no deliberate Mandarin/Hokkien; the employee speaks professional
English). Profiles constrain spoken dialogue only; canonical values, metadata, schema fields and planning prose keep
their English representation. NRIC/FIN labels (`terminology.zh`) never change the identifier value.

## Normalization 2.0

Every spoken form names its language; each language has exactly one DEFAULT form per profile. `generation_surfaces`
is the allowlist Python will render from; `aliases` are the accepted decoding aliases (bare "yuan"/"yen" are accepted
but never generated; E-U-R, J-P-Y and C-N-Y are never generated). Standalone 元/块/刀 and "dollar(s)" are rejected
currency expressions (`match: STANDALONE`; 美元, 新元, 欧元 and 日元 are unaffected). Product names keep their canonical
English identity; `language_names` holds the approved Chinese aliases (MAP_LANGUAGE_NAMES). `language_rules.zh` states
the deterministic Mandarin cardinal, year, date, identifier, decimal and percentage rules; it is declarative until G4
implements them.

## Edge cases and variants

All 21 edge-case IDs are kept. Each catalog entry lists its eligible `language_profiles` and controlled
`language_variants` (`READY_V1`, `NEEDS_REVIEW`, `EXCLUDED_V1`). A variant changes how an existing mechanism is
realised, never what is tested or which value is gold, and never counts as an extra assignment.

* `CROSS_LANGUAGE_EN_TO_ZH` / `CROSS_LANGUAGE_ZH_TO_EN` (EN_ZH_MIXED): the mechanism's earlier state and later state
  are spoken in different languages; G2B pins exactly those events (`check_plan_skeletons/languages.py`).
* `EN_SURFACE` (EN_ZH_MIXED, the four English spoken-number cases): mentions carrying the assigned English form are
  pinned to English.
* `ZH_MAGNITUDE_CONTRAST` (MANDARIN, `EC_MULTIPLE_MONETARY_VALUES`): exactly one distractor is the final amount x10 or
  /10 from the same currency domain (a divided distractor never takes a balance role). Variant-free Mandarin monetary
  distractors never contain such a contrast, so every contrast is labelled.
* `ZH_BAIWAN_DECIMAL` is `EXCLUDED_V1`: ordinary Mandarin 一百二十万 is normal normalisation, not an edge case.
* The four English spoken-number cases are not eligible for MANDARIN in this release.

`python scripts/report_multilingual_plan.py --dataset-root datasets/advisory_execution_multilingual_v1 --compatibility`
prints the edge-case x profile matrix generated from the catalog; `--composition` prints the planned composition from
the validated planning record.

## Sizing and quota resolution

The release has 250 calls (the earlier 500-call plan was superseded; the release name is unchanged). The approved 500-call Option D
quotas are kept verbatim as the design weights in `authoring/quota_design.json`; `fyp_evaluation.planning.quotas` scales each one
exactly (x 1/2) and rounds jointly, so profile totals and global targets stay consistent, then stores the integer result in
`generation_plan.json` (with the design hash in `quota_resolution`). Never edit those quotas by hand: change the design and rerun

```
python scripts/resolve_multilingual_quotas.py --dataset-root datasets/advisory_execution_multilingual_v1 --check   # stored == recomputed
python scripts/resolve_multilingual_quotas.py --dataset-root datasets/advisory_execution_multilingual_v1 --write   # replace the quotas
```

and then regenerate G1, G2A and G2B with their own tools. Totals: 560 compliance FAIL labels and 560 edge-case assignments;
25 all-PASS, 25 edge-free and 10 pure-control calls; 225 SHORT / 25 LONG; per-question FAIL targets C02 40, C03 65, C08 50 and 45 for
each other check; integer T03 exposure caps 26/26/17/17 (35% rounded down). The resolver refuses a request that has no exact integer
solution (for example a profile size that makes a scaled total fractional) instead of relaxing a rule.

`python scripts/report_multilingual_plan.py --dataset-root datasets/advisory_execution_multilingual_v1 --coverage` prints the
per-profile mechanism and variant counts as `assignments / distinct calls` (SINGLETON / VERY SMALL groups flagged by distinct calls)
and a future 70/30 development / held-out assessment (175 / 75 calls) over whole calls: which groups a proportional split would leave
with one call in a subset or unable to appear in both. Distinct calls come from the G1 outputs; with only the quotas they are shown
as `unknown` and no split assessment is produced. The group estimates are independent, so they are indicators, not proof that one
split satisfies every group at once. The report assigns
nothing; the deterministic stratified split, CI and the registry of inspected or pilot-exposed calls (to be assigned to development)
are later work, and the planning artifacts already carry the metadata a split needs (language profile, mixing style, duration,
verdicts, edge-case IDs, families, variants and transaction attributes).

## G4 (schema 2.0)

`check_plans` keeps the schema-1.0 behaviour for the pilot byte for byte and switches to language-aware behaviour when the release is
multilingual (`validation.stage0.is_multilingual`). The model still writes only templates and chooses among explicitly permitted forms;
Python validates each choice, renders every surface from frozen inputs and computes spans as Python character offsets.

* **One source of choices.** `PlanContext.choices(event, placeholder)` returns `FormChoice(form, language, surface)` for question
  placeholders and for `FEE` / `LIMIT_PRICE` / `CURRENCY`; request previews, response validation and rendering all read it.
  Choices are the skeleton's forms narrowed to the event's allowed languages (`required_language`, else the allowed speakers'
  `spoken_languages`) and to the forms that can express the particular value (EUR and JPY have no `SPELLED_CODE`). A field with no valid form is
  not offered for readback; a REQUIRED form that cannot be spoken in the event is a contract error. A missing choice is a validation error.
* **Surfaces.** `surfaces.SurfaceCatalog` reads `spoken_forms`, `generation_surfaces`, `aliases`, `language_names` and implements
  `language_rules.zh` through `zh_numbers` (algorithmic cardinals with 万/亿, 零 gaps and the 两 rule; decimals; years with 〇; dates;
  identifiers with Han digits). When a form lists several approved surfaces, one is picked deterministically from the call, question, value and form.
* **Decoders** parse the surface alone (never the expected value) and reject shortened or ambiguous forms. Accepted equivalents:
  两 or 二 before 百/千/万/亿, 〇 or 零 in a year, spaces inside a Mandarin identifier. Fee and limit-price surfaces are checked on amount,
  currency and basis with `Decimal`; more decimal places than the declared rule is an error, never a rounding.
* **Guards** (`guards.py`, plus the CJK-safe patterns in `template.py`) are mechanical and narrow: frozen script rules, event language pins,
  numeric expressions in Chinese wording outside the slots (a bounded rule set, not a number parser), standalone 元/块/刀, approved Chinese product
  names outside `[[T01]]`, and bare unqualified affirmatives in planned weak acknowledgements. Mandarin compliance wording is NOT checked in G4; G5/G7 judge it.
* **Provenance.** Plans and records are schema 2.0 (`check_plan.v2` / `check_plan_record.v2`): mentions carry `language` and `source` / `upstream`
  carry `language_profiles_sha256`.
* **Selected-call readiness.** With `--call-id`, G3 must be intact release-wide but only the selected calls need an accepted, current G3 config.

The Mandarin golden examples and word lists in the tests and `guards.py` are provisional until a native speaker has reviewed them.

## G3-G5 (schema 2.0)

Each stage selects its generation from the release (`release_versions`): a multilingual release uses `<name>.v2` schemas and prompts, the
historical release keeps the originals byte for byte (request hashes are pinned in `tests/test_request_hashes.py`).

* **Provenance.** Every v2 source and upstream block records `language_profiles_sha256`. G3 configs carry language-specific
  `spoken_names` (Latin for English speech, Han for Mandarin speech; the planning name stays English) and the scenario prose stays English.
* **Register guide.** `prompts/shared/register_singlish.<version>.md` is read only for SINGLISH calls and is an *upstream input* of
  G4 and G5 (G6 and G7 in Phase 3B): its hash is recorded in attempts and artifacts and re-read from disk whenever staleness is checked, so a
  change makes in-flight attempts `STALE_UPSTREAM` and accepted SINGLISH outputs `stale`, and never touches other profiles. A change
  is a new version file; older versions stay so a recorded hash resolves to its exact text (`register_guide.version_of`). A guide
  change never triggers regeneration by itself; it only reports what is affected.
* **G4 prompt v2.** Within-turn switching is a property of the client's dialogue turn: it never forces a bilingual event and never overrides
  a pin. Singlish register is realised only in existing client events; the employee stays professional English.
* **G5 prompt v2.** A language section, the `LANGUAGE_REQUIREMENT_NOT_MET` issue code (a plan that breaks a stated language requirement,
  including employee wording that is not professional English in a SINGLISH call, judged in context and blocking), and Chinese quotes
  grounded verbatim. Controls for the semantic checks live in `plan_verifications/controls.py` (`MULTILINGUAL_CONTROLS`).
* **Paid-run guard.** A live (non-fake transport) run of any stage on a multilingual release needs an allowlisted batch
  (`authoring/live_allowlist.json`: stages and calls), a git commit and no uncommitted change under `src`, `prompts`, `schemas`, `scripts`.
  The attempt records an `exposure` block (`run_kind`, `code_commit`, `code_dirty_paths`, `allowlist_batch`) in its PREPARED state, before
  anything is sent. `scripts/report_exposure.py` derives `authoring/exposure_log.json` from the audit store and provenance (live attempts only).

## G6, G7 and G9 (schema 2.0)

* **G6 (dialogues).** Batch or synchronous (`--sync`, one call at a time, no SDK retries); both send the same request. `dialogue.v2.md` / `dialogue.v2`, `g6_dialogue_v2`. The brief carries the language block, each speaker's spoken names,
  the length unit and (SINGLISH) the register guide; the advisory recommendation gets `product_names` per allowed language (the Chinese
  name is the approved controlled alias). A marker may sit beside Han characters or CJK punctuation but never beside Latin letters, digits
  or another marker. Deterministic checks on the model's own words (`dialogues/language_checks.py`): the frozen script rules, and no
  numeral or value word glued to a planned number (a planned `七百` can never become `七百五十`). Everything else about filler (a confirmation,
  a fee, market commentary) is judged by G7, never by G6. Length is counted per language profile: ENGLISH and SINGLISH keep the historical
  word policy; MANDARIN counts Han characters and EN_ZH_MIXED counts Han characters plus English words. **These units, rates and bands are
  PROVISIONAL** (see `LENGTH_POLICY_V2`) until the pilot's audio calibrates them. Each attempt records its call's own policy; the policy
  version names the unit (`g6-length-v2-han-chars`, `g6-length-v2-han-chars-plus-words`).
* **G7 (verification).** `dialogue_verification.v2.md`. Python derives `language_rows` per call from the frozen profile (EN_ZH_MIXED: client
  substantive in both languages, a within-turn switch, bilingual speech distributed; SINGLISH: client register realised, every employee turn
  professional English; plus one row per edge case with a language variant). Rows are blocking like checklist rows (`LANGUAGE_REQUIREMENT_NOT_MET`);
  a MATCH must cite real speech of the right kind (mechanical rules in `dialogue_verifications/language_rows.py`, which prove the citation,
  not the requirement). `LANGUAGE_NATURALNESS` is an advisory quality finding and never changes the verdict. Controls: `MULTILINGUAL_CONTROLS`.
* **G9.** Only the source schema grows (`ground_truth.v2`: `language_profiles_sha256`, and the SINGLISH dialogue's `register_guide_sha256`);
  the answer-support rule (`answer_support.py`, extracted unchanged from the G9 compiler) is shared, and the pilot's G9 output reproduces
  byte for byte.
* **Register guide.** A SINGLISH call records the guide version it was built under in its accepted generation metadata, so an accepted
  artifact still rebuilds and validates after the guide changes (only its dependency status becomes stale).
* **Watch item: G7 advisory rule (d).** G7 v2 checks that the client's turn after a G6-authored recommendation is not an instruction,
  commitment, confirmation or statement of understanding. It does not distinguish G6's own words from a planned G4 event in that turn.
  G6 prompt v3 lets that turn start directly with a planned event, so a legitimate planned instruction could be flagged. Never assume
  such a finding is a false positive. Examine the cited speech first, and classify it as a G7 false positive only after confirming both:
  (1) the flagged instruction or commitment is the exact text of a planned G4 event that the plan authorised there; (2) G6's own words
  in that turn add no further commitment, confirmation, agreement or understanding. Otherwise it is a genuine G6 issue (stop and report).
  Only a confirmed false positive justifies versioning rule (d) to cover G6-authored speech only, as a change separate from any G6
  prompt change.

## Claude-assisted review (G5 and G7, multilingual releases)

A reviewer (Claude Code, recorded as `claude-code-harness:<model>`, never as the provider) can supply the G5 or G7 verification instead of
Gemini, with no provider request and no credential:

```
python scripts/verify_authoring_plans.py   --dataset-root DS --prepare-review --call-id CALL_X --output packet.json   # G5; scripts/verify_dialogues.py for G7
python scripts/verify_authoring_plans.py   --dataset-root DS --ingest-review review.json --call-id CALL_X [--reopen --reason ...] [--retry-rejected]
```

* `--prepare-review` writes the stage's own brief and its `request_sha256` (read-only). The reviewer reads the brief, the frozen targets and
  the stage's instructions, and writes the stage's own response schema in a `claude-review/1` envelope bound to that hash.
* `--ingest-review` runs the normal lifecycle: recovery, freshness, eligibility, validation (coverage, ordering, issue links, every quotation
  located in the cited turn or event, mechanical language-row rules), Python verdict derivation and acceptance. A review written against any
  other brief (changed inputs, guide or instructions between preparation and acceptance) is refused; a re-import is refused unless
  `--reopen` (history is kept); a rejected review needs `--retry-rejected`; the semantic budget applies as for a provider response.
* The response is stored in the response schema's key order so the accepted artifact rebuilds byte for byte. The attempt has `transport:
  manual`, `params.method: manual_review`, cost 0 and exposure `run_kind: manual` (never counted as provider exposure). An ambiguous Gemini
  attempt still blocks Gemini for that call but does not block a review.
* VALID permits the next stage (G6 after G5, G8 after G7); INVALID and UNCLEAR block it. G7 language rows are blocking; naturalness findings
  are advisory. The review is judgement, Python checks only that it is complete, consistent and grounded in real text.

G6 currently selects `dialogue.v5.md` on multilingual releases. Earlier versioned prompts are retained; the first four pilot dialogues record the v2 hash.

### Codex review adapter and qualification

For Codex, add `--reviewer-kind codex-harness` to **prepare** (both G5 and G7).
The default remains the historical Claude packet. Codex packets use
`codex-review/1`; the reviewer writes that format with `reviewer.kind:
codex-harness`, an honestly identified reviewing model, the prepared
`request_sha256`, `review_policy_sha256` from `review_policy.sha256`, and the
stage's unchanged `response` object. Ingestion reads the identity from the file;
do not pass `--reviewer-kind` on ingest or label Codex as Claude or Gemini.

The packet contains the exact existing provider request **plus** the immutable
`prompts/shared/codex_review.v1.md` supplement. The request hash binds the brief
and original stage instructions; the separate policy hash binds the additional
review scope. Both are checked again before a new attempt is created. The policy
version and hash persist in generation parameters; its full text is also kept
with the raw manual response. Historical reconstruction uses recorded metadata,
not a replacement current policy, and existing reports are not automatically
reopened. Add a new policy version rather than editing a used version.

For G7, this supplement explicitly checks authoring conformance even when gold
is unchanged. A G6-authored refusal to execute, protected escalation, review
commitment, disclosure or instruction can violate the authoring rules without
changing a checklist answer. Report the actual act as a whole-dialogue
`CROSS_CHECK_INTERFERENCE` MISMATCH when no checklist row changes; do not falsely
mark an unaffected checklist row MISMATCH. Such issues already participate in
Python's blocking verdict. Authorized G4 acts and the permitted rationale-free
G6 recommendation are exceptions, assessed contextually. Naturalness stays
advisory. Historical Claude/Gemini reports have not assessed this added scope
merely because their verdict says VALID.

Adapter tests use artificial responses to verify lifecycle and compatibility;
they are not semantic qualification. Qualification separately requires actual
reviews of existing examples covering all four profiles, paired clean and
defective controls (fee invention, strengthened weak confirmation, changed final
instruction, omitted negation, unplanned escalation), grounded findings for the
intended reasons, and explicit reporting of misses and false alarms. Keep this
exercise in isolated text-only copies with separate audit stores. Preserve
production reports, gold and all existing dialogue bytes. A small successful
exercise is evidence for use, not a statistical reliability guarantee.

The manual transport makes no provider request and records zero API spend;
the reviewing assistant still consumes its own usage allowance. Model-family
blind spots remain possible. Independent audits should include apparently VALID
cases, not only disputes. A paid rollout still needs its own approved scope,
budget and committed allowlist. Use OpenAI Batch for G3/G4/G6 when so authorized,
with G5 approval before G6 submission and G7 after collection. Check the semantic
verdict as well as CLI exit status: an accepted verification artifact can be
INVALID/UNCLEAR. Stop new submissions on a blocker; in-flight Batch work may
still complete and cost money. Never retry an unchanged semantic failure.

## G1 rules (2.0)

Each profile is planned exactly: duration, intent, composition, FAIL histogram, per-question FAIL counts, edge-case
histogram, per-case counts, per-variant counts, mixing styles, currency, direction and order type. The kernel
(`planning/multilingual.py`) builds a per-call layout first, places edge cases, assigns variants without pinning one
event to two languages, fills compliance FAILs exactly with a max-flow and only then builds values. Additional rules,
all re-checked independently by `planning/validate.py`:

* C02 PASS with C03 FAIL occurs exactly when `EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED` is assigned (a generation
  rule for this release; `check_semantics.json` is unchanged and G5/G7 still judge the wording).
* At most one C09-C12 prohibition FAIL per call; SHORT calls carry at most 6 FAILs plus edge cases.
* Calls targeting T03 stay within the per-profile exposure cap.
* An order replacement supplies every transaction FINAL, so no correction or reference case may target a transaction
  field in the same call, and no long-distance dependency may share the call.
* No call carries two assignments with the same case and target set (one behavioural requirement would be counted twice).
  Long-distance dependencies on different question pairs in one call stay legitimate; the allocator draws an unused pair.
* K01/K02 are unique release-wide; superseded and distractor values stay inside the configured domains.
* G0 rejects a planned (profile, case, currency) combination with no coherent eligible value (the EUR/CNY
  large-number gap).

## Commands

```
python scripts/resolve_multilingual_quotas.py --dataset-root datasets/advisory_execution_multilingual_v1 --check
python scripts/plan_dataset.py --dataset-root datasets/advisory_execution_multilingual_v1 --check
python scripts/compile_call_specs.py --dataset-root datasets/advisory_execution_multilingual_v1 --check
python scripts/build_check_plan_skeletons.py --dataset-root datasets/advisory_execution_multilingual_v1 --check
python -m pytest -q tests/test_multilingual_planning.py tests/test_quota_resolution.py tests/test_multilingual_foundations.py tests/test_multilingual_g5_chain.py tests/test_multilingual_g6.py tests/test_multilingual_g7.py tests/test_multilingual_g9.py
python scripts/report_exposure.py --dataset-root datasets/advisory_execution_multilingual_v1 --check   # once a live batch exists
```


## Advisory G6 length (user decision, 2026-10-27)

New multilingual G6 requests use `dialogue.v4.md` and a recorded `*-advisory-v1` length-policy version. Counts, target bands,
reference acceptance bands and estimated seconds are retained; being under or over the band does not reject a response.
New artifacts record `length.enforcement = "advisory"` and `length.within_range`. These are text measurements, not measured
speech duration. All other G6 gates, G5 eligibility and G7 semantic verification remain mandatory. Quality findings remain
visible for review; this change does not resolve repetitive dialogue. No audio generation is implied.

Earlier prompts and policy versions remain unchanged, and accepted historical artifacts rebuild under their recorded policy.
To reuse an existing completed response rejected ONLY for length, without another provider request:

```powershell
python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_multilingual_v1 --revalidate-length-only --call-id CALL_113 --attempt-id A001 --reason "User approved advisory length"
```

This checks the original request hash, completed response, original length-only rejection, current upstream hashes, G5
approval and every non-length validation rule. It creates a separate zero-cost `revalidation` attempt that links to the
original response and retains its actual generation-prompt hash. The original rejection, response and original cost remain
unchanged. Stale, ambiguous, incomplete or otherwise invalid responses cannot be reused. Recovered dialogues still need G7.


## G6 prompt v5 and in-place regeneration (user decision, 2026-10-27)

New multilingual G6 requests use `dialogue.v5.md` (v4 plus whole-call naturalness: topic development instead of repetition, a
recognisably banker employee role, chronology of the speakers' own wording, an opening that follows `setting.call_reason`, natural turn
length, and non-forced code-switching). Events, markers, placement constraints, the eleven supporting-speech bans, the recommendation rules
and the advisory length text are unchanged; v1 to v4 stay byte-for-byte. Intentional non-response, hesitation and avoidance are preserved
and no compliance act may be added to make a reaction sound appropriate.

A completed call is regenerated in place with `--sync --reopen --reason ...` (batch `PILOT_2_G6V5`). The previous accepted attempt is marked
SUPERSEDED and kept in `accepted_history` with its full response; its request, raw response and cost stay in the audit store, and the
exposure ledger counts every live attempt. The G7 review of the old dialogue becomes STALE (G8-ineligible) until a new review of the new
dialogue is ingested with `--reopen`; the old review stays in G7 history. The gitignored audit store must be backed up before regeneration.

To make an earlier accepted dialogue active again (a plain Git restore is NOT supported: it contradicts provenance and the audit store):

```powershell
python scripts/generate_dialogues.py --dataset-root datasets/advisory_execution_multilingual_v1 --reinstate-accepted --call-id CALL_103 --attempt-id A002 --reason "v5 candidate judged worse"
```

This is offline and zero cost. It needs the source in `accepted_history`, its stored request and raw response matching their recorded
hashes and parsing to the recorded accepted response, current upstream and G5 eligibility, and passing current validation. It creates a new
`reinstatement` attempt (never reusing an ID) that supersedes the currently accepted attempt through the normal reopen path; the dialogue
is then re-reviewed with G7 `--reopen`.

Upstream presuppositions that v5 cannot fix (frozen G4 text or derived placement) are reported separately, not changed here:
CALL_028 (a fee-currency distractor says "the currency I selected" before the currency is stated, forced by the derived EN to ZH order),
CALL_027 and CALL_034 (distractor amounts are disclaimed as "not the amount for this order" before it is stated), CALL_036 (the
recommendation wording cites a preference the client states later in the plan).
