# Multilingual release contract (`advisory_execution_multilingual_v1`)

A multilingual release keeps the gold-first G0 -> G9 pipeline and adds language profiles. Canonical gold stays
language-independent: the coverage matrix stores `T04 = 700`, never `七百`. A profile only constrains how values and
dialogue are spoken. The historical `advisory_execution_pilot_v1` release is unchanged and still reproduces byte-for-byte.

Implemented so far: G0, G1, G2A and G2B (Phase 1), resized to 250 calls (75 ENGLISH, 75 MANDARIN, 50 EN_ZH_MIXED, 50 SINGLISH), and the
deterministic G4 layer for schema 2.0 (Phase 2: renderers, independent decoders, shared form choices, guards, schema-2.0 provenance and
selected-call readiness; see "G4 (schema 2.0)" below). G3 still refuses to start on a multilingual release
(`call_configs/inputs.py`) until its language-aware briefs exist, so G4 only runs offline here (artificial G3 configs and fake
transports in the tests); the multilingual G3-G7 prompts and briefs, G5-G7 language assessments,
TTS, STT, scoring, the deferred 70/30 development / held-out split (175 / 75 calls) and CI are later work.

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
python -m pytest -q tests/test_multilingual_planning.py tests/test_quota_resolution.py
```
