# G4 check_plan authoring instructions (template v3, multilingual releases)

You write the wording of planned evidence for ONE synthetic private-banking telephone call. The compliance verdicts, the extraction values, the events, their order, their speakers, their languages and every structural constraint were fixed by people before you were called. You are not writing the dialogue, you are not deciding anything about the call, and you must not change anything that is fixed. A later stage turns your wording into dialogue turns; an independent verifier will later judge whether your wording really establishes the intended behaviour, so write each event to genuinely do what its condition describes (or, for a violation event, what the fail condition describes).

The user message is a JSON brief for one call. Reply with one JSON object that follows the response schema exactly: one entry per event in the brief's `events`, using exactly the event IDs given, no extra and no missing events.

## The language of this call

The brief's `language` block gives the call's language profile, its mixing style, the script rules and, for the client and the employee, the languages each may speak and must speak. Every event lists `spoken_languages` (the languages that event may be written in) and, when its language is fixed, `required_language`.

- Write every event's `text` in a language it may use. When an event has `required_language`, write the whole event in that language (the value slots aside). Never change an event's language and never write a different language "to be natural".
- ENGLISH: standard, courteous English.
- MANDARIN: natural conversational Simplified Chinese, in the right register for the speaker (a private-banking employee is professional). Write no Latin letters and no Arabic digits yourself: the only Latin letters in the whole call are inside the identifier that Python inserts through `[[K01]]`. Do not use Chinese numerals for values either (see below).
- SINGLISH: the client speaks colloquial Singapore English; the employee speaks professional English with no Singlish phrasing. Write no Chinese characters and no romanised Mandarin, Hokkien, Cantonese or Malay sentences. The brief carries a `register_guide`: use it as reference material, not as a checklist. Colloquial grammar and particles belong where they carry meaning, in client events where they fit; many client lines can stay plain, and short neutral replies are fine. Do not force particles into every line.
- EN_ZH_MIXED: the client must really use both English and Mandarin during the call. For an unpinned client event you may write English, Mandarin or a natural mixture where it fits; you never have to make an individual event bilingual, and you never override a pin. The client's within-turn switching is a property of the dialogue: the next stage builds each client turn from your events plus the speech around them, so two events in different languages, or an event and the connecting speech, can form a switch inside one turn. Switching only at turn boundaries is extra, never a substitute. There are no quotas of any kind, and the employee may answer in either language and is never required to switch.
- Use the names in `scenario.client.spoken_names` and `scenario.employee.spoken_names` for the language you are writing, when a name is needed. Do not invent other names.

## What you write for each event

- `event_id`: copy it exactly.
- `speaker`: `null` when the brief says the speaker is fixed. When the brief lists more than one allowed speaker, choose one of them.
- `text`: a template for what is said in this event: one short, natural utterance (typically one or two sentences), spoken by the event's speaker in the event's language. Each event is later realised as one contiguous passage inside a single turn, so write it as that speaker's own words, not as stage directions.
- `value_forms`: only for a placeholder whose `form_choices` lists more than one form: one `{placeholder, form}` entry choosing one of the listed forms. Each choice shows the form's language: choose the form that fits the language of the sentence you write around it. Otherwise an empty list.
- `readback_question_ids`: `null`, except for a READBACK event (see below).

## Placeholders: you never write values

Every structured value is inserted by Python through a slot written `[[NAME]]`, for example `[[T04]]`. The brief lists, per event, the placeholders that are `required` (use each exactly once), `optional` (use at most once, only if natural) and, by omission, forbidden. Never write a value yourself: no digits, no spelled-out or Chinese-numeral amounts or quantities, no magnitude words (hundred, thousand, million, 百, 千, 万, 亿...), no percentages, no currency names, codes or symbols (including 元, 块 and 刀), no product names, no identifiers and no dates in your wording. Python replaces each slot with the surface form shown in `rendered_surfaces`, so write the sentence around it so that it reads naturally with that surface. `[[LIMIT_PRICE]]` and `[[FEE]]` render the scenario's limit price and fee, and `[[CURRENCY]]` the currency of a monetary mention, in the language of the form chosen for them.

## Per-kind guidance

- COMPLIANCE_ACT: write what the speaker actually says so that the event does what its `condition` (the check's pass or fail condition) describes. Paraphrase the condition in natural speech; where `wording_constraint` says so, do not copy the checklist wording. An EXCHANGE check has an EMPLOYEE act and a CLIENT response: make the response answer the act clearly and unambiguously (the C03 and C06 and C08 responses must be plain, unambiguous answers, never a bare "okay").
- VALUE_STATEMENT / ANTECEDENT_MENTION / DISTRACTOR_MENTION: the speaker states or mentions the value(s) of the listed placeholders. A DISTRACTOR_MENTION mentions a value for a different purpose (its `distractor_role`); make the role clear in the wording, so it cannot be mistaken for the transaction value. An ANTECEDENT_MENTION is a candidate that is only mentioned, not chosen.
- VALUE_CORRECTION: the client corrects the earlier value of the event it `corrects_event_id` points at, giving the new value through the placeholder, with a correction cue. English: "actually", "sorry, I meant". Mandarin: for example "不对，我是说", "抱歉，我刚才说错了". The brief's constraints decide whether the correction happens in the same turn or later.
- ORDER_STATEMENT: the client states a complete order using every required placeholder. A SUPERSEDED order is the earlier order; ORDER_INVALIDATION (no placeholders) withdraws the earlier order; the FINAL order replaces it.
- REFERENCE_SELECTION: the client picks the earlier-mentioned candidate by reference only (English: "the second one", "the one you mentioned first"; Mandarin: "第二只", "您刚才先提到的那只"). No placeholders: never name the product or state the quantity.
- READBACK: the employee reads back what they have understood. Choose the fields in `readback_question_ids`: it must contain every `must_include` field, at least `min_field_count` fields in all, and only fields from `selectable_fields` (a field that cannot be spoken in this event's language is not offered). Use each chosen question ID as its placeholder name, exactly once (for example, `[[T04]]` for chosen field `T04`), and no other placeholders. The letter Q is only a metavariable, never a literal placeholder name. The `wrong_question_id` field is deliberately read back wrongly (Python renders the wrong value); the other fields are read back correctly. Do not signal in the wording that anything is wrong: the client corrects it in a later event.
- FALSE_START: the client starts a statement and trails off or abandons it: end the text with a dash or an ellipsis (English "—" or "...", Mandarin "——" or "……"). No placeholders.
- GUARANTEE_REQUEST (client) asks whether returns are guaranteed or certain; GUARANTEE_REFUSAL (employee) declines to promise returns. No placeholders.
- WEAK_ACKNOWLEDGEMENT: a vague, non-committal client acknowledgement that does NOT confirm anything, for example "Mm, I see, let me think about it" or "嗯，我先想想". The whole text must never be a bare, unqualified agreement (English yes, okay, sure, can; Mandarin 可以, 好, 好的, 行, 对, 没问题), and never use confirm, correct, right, agree, go ahead or proceed. Hedged wording stays acceptable ("可以吧", "Can ah?" asks rather than agrees). No placeholders.

## Checks that have no events (absence)

The brief's `checks` lists every check. A check whose `evidence_mode` is ABSENCE has no event, on purpose: its behaviour must not be expressed anywhere in the call (`must_not_be_expressed_anywhere` gives the condition that must not happen). Never add or imply that behaviour in any event. In particular, do not mention guarantees of returns outside the guarantee events, do not hint at money laundering, concealed funds or inside/confidential information unless an event for that check exists, and mention a fee only through `[[FEE]]` in the one event that carries it. A colloquial register never adds a confirmation, a risk acceptance, a guarantee or a fee mention to an event whose planned act is something else.

## Consistency with the scenario

The brief's `scenario` and `products` are fixed facts: use them to make the wording consistent (the client's and employee's names and manner, the product, the reasons). Treat the final T01 product as the recommended product; for a SELL the recommendation may be to sell, reduce or exit that holding. Capital loss is a general disclosure topic. Do not invent products, currencies, amounts, dates, identifiers or other figures. A fact in the scenario (a fee, a risk, a limit price) is not said by anyone unless an event carries it. Products are named only through `[[T01]]`.

## The checklist (fixed semantics)

{{CHECKLIST}}

## Ordering and dependency rules (fixed)

{{RULES}}
