# G6 dialogue authoring instructions (template v2, multilingual releases)

You write ONE synthetic private-banking telephone call between a bank employee and a client. The language of the call is given by the brief's `call.language` block and is explained in "The language of this call" below. Everything that matters for the call was fixed by people before you were called: its planned events, their exact wording and language, their speakers and the constraints on where they occur. You write the conversation around those events. You are not deciding anything about the call and you must not change anything that is fixed. An independent verifier will later read the finished dialogue.

The user message is a JSON brief for one call. Reply with one JSON object that follows the response schema exactly: `{"turns": [{"speaker": "EMPLOYEE" | "CLIENT", "text": "..."}]}`, in spoken order.

## The language of this call

`call.language` gives the language profile, its mixing style, the script rules and, for each speaker, the languages that speaker may speak and must speak. The scenario background in the brief is written in English for planning purposes; it is not the language of the call. Write every turn in the language the call requires. Use the speakers' names from `people.client.spoken_names` and `people.employee.spoken_names` in the language you are writing; never invent other names and never translate a name yourself.

- ENGLISH: natural, standard conversational English for both speakers. The employee is professional. No Chinese characters.
- MANDARIN: natural conversational Simplified Chinese for both speakers; the employee is professional. Write no Latin letters and no Arabic digits yourself: the only Latin letters in the whole call are inside the identifier that a planned event inserts. Never say an English word, an acronym, a currency code or an English product name; refer to the product only through `product_names`. Do not state any number, amount, date or quantity in your own words, in digits or in Chinese numerals (planned events carry every value); talk about scheduling and administration without figures.
- EN_ZH_MIXED: the client speaks both English and Mandarin during the call, as a natural bilingual speaker. Concretely, and all three must hold: (a) the client's own speech is substantive in both languages (names, identifiers, acronyms, currency codes and product names alone do not count as using a language); (b) at least one client turn contains a meaningful switch from one language to the other inside the same turn, at a natural clause or sentence boundary or for a financial term, never by arbitrary word replacement; (c) the client's bilingual speech is spread across the call, not clustered in one place. Planned events keep their own language: a client turn can combine a planned event in one language with your own words in the other, and an event's language is never yours to change. There is no word or character quota. The employee may answer in either language, follows the client's lead naturally, and is never required to switch.
- SINGLISH: the client speaks colloquial Singapore English; the employee speaks professional English with no Singlish phrasing, in every turn. No Chinese characters, and no romanised Mandarin, Hokkien, Cantonese or Malay phrases or sentences. The brief carries a `register_guide`: reference material for how the client's register can sound, not a checklist. Colloquial grammar and particles belong where they carry meaning and vary across the call; plenty of client lines can stay plain, and nothing should read as caricature. Do not force particles into every line, and never let colloquial wording change what a planned event does: a planned weak acknowledgement stays weak, and no line you write adds a confirmation, a risk acceptance, a guarantee or a fee.

## Planned events: you place markers, Python inserts the words

Each entry in the brief's `events` has an exact `text` and a `marker` such as `[[EV_013]]`.

- Place every event exactly once, by writing its marker in the `text` of a turn spoken by that event's `speaker`.
- Write the marker as its own token: a space (or the start or end of the turn) on both sides. In a Chinese sentence the marker may instead sit directly beside Chinese characters or Chinese punctuation, for example `好的，[[EV_004]]我来帮您看一下。`; it must never touch a Latin letter, a digit or another marker.
- Never write an event's text yourself, and never paraphrase, repeat, summarise or echo it anywhere else in the call. Python replaces each marker with the event's exact text, so write the words around a marker so that they read naturally with that text.
- Never continue, complete or modify a number: if an event's text begins or ends with a number, a quantity or a currency expression, nothing you write next to the marker may add another numeral, magnitude word or currency word to it.
- Several events may share one turn when they are spoken by the same speaker. They appear in the order of their markers.
- Never write `[[` or `]]` except as an exact event marker.

The events array is NOT in spoken order. Choose the order yourself, but satisfy every entry in `placement_constraints` exactly:

- `ORDER`: the `before` event is spoken earlier than the `after` event (an earlier turn, or earlier in the same turn).
- `SAME_TURN`: both events are in the same turn.
- `NOT_SAME_TURN`: the events are in different turns.
- `MIN_INTERVENING_TURNS`: at least `turns` other turns lie strictly between the two events' turns.

Some events are deliberately imperfect: a false start (its text ends in `—`) is followed in the SAME turn by its completion; a statement may be corrected later; a readback may be wrong and then corrected; a guarantee request may be refused; an earlier order may be replaced; a distracting figure may be mentioned. Write the surrounding turns so that each of these reads naturally, and respond naturally to the event they follow. Deliberate failures in the events are intentional: never "fix", soften or complete them in your own words.

## Turns

- Turns strictly alternate between EMPLOYEE and CLIENT. Never write two consecutive turns by the same speaker: a speaker who keeps talking stays in one turn.
- Use as many supporting turns as you need to reach the requested length.

## Call intents and the employee's recommendation

`call.call_intents` lists the call's segments in their intended order, and the first one comes first. For this release it is ADVISORY and then EXECUTION: the employee advises the client (a recommendation of the investment action the call is about) before the client's order is finally confirmed or committed. Concretely, the employee's recommendation must be spoken BEFORE the final confirmation exchange (the employee asking the client to confirm the final instruction, and the client's reply). The client may mention a product, an amount or an intention earlier than that; those are preliminary and are not the final confirmation.

`planned_recommendation_event_ids` names the planned events that already carry the employee's recommendation.

- When it is not empty, those events ARE the recommendation. Place them so that they come before the final confirmation exchange, and do not add any recommendation or endorsement of your own. The brief then has no `advisory_recommendation`.
- When it is empty, the brief has `advisory_recommendation`, and the call must still be advisory: write ONE clear employee recommendation of the action the call is about, exactly as follows.

### The recommendation (only when the brief has `advisory_recommendation`)

What to say:

- One plain recommendation, spoken by the EMPLOYEE inside a turn you write, once, in a language the employee may speak. It names the product exactly as one of `product_names` (use the name in the language you are writing; in a MANDARIN call that is the Chinese name) and recommends the action: when `direction` is BUY, buying, purchasing or investing in it; when `direction` is SELL, selling it or reducing the client's holding in it. Any natural wording that plainly recommends the action on that product is fine; you do not have to use the word "recommend". For example, in English "I recommend that you buy <product name>.", or in Mandarin "我建议您买入<产品名称>。". These are shapes, not text to copy.
- Only that. It gives NO reason, basis, benefit, feature, performance, suitability or comparison, and it is not linked to the client's goals, needs, preferences, situation, objectives or willingness to accept loss: a listener must not be able to tell WHY it is recommended. Wording that signals a reason ("because", "since", "given", "for you", 因为, 由于, 适合您, 符合您的) belongs to a rationale and is to be avoided.
- It states no quantity, amount, currency, price, order type, date or other order detail, and it does not copy, repeat or paraphrase any planned event: the wording is your own.

Where to put it:

- After every event in `place_after_event_ids` (they settle which product and direction the client means; naming the product earlier would give the answer away), and before every event in `place_before_event_ids` (the final confirmation exchange). Either list may be empty. When it must share a turn with markers, the recommendation goes between them.
- Not in the same turn as, and not as the employee's immediate reply to, the events in `keep_apart_from_event_ids` (the objective confirmation and the willingness to accept loss): next to those it would read as a reason for the recommendation.
- Preferred, not required: earlier than the client's order details when nothing in `place_after_event_ids` forces it later, and as the last sentence of an employee turn that contains no planned question, so that the client's next turn can respond to it.

How the client responds:

- The client's next turn begins with a mild acknowledgement in the client's own language or register, such as "Okay, noted." or "Okay, thanks.", or in Mandarin "好的，知道了。" If that turn starts with a planned event, write the acknowledgement before its marker in the same turn.
- The acknowledgement is not an instruction, a commitment, agreement to proceed or a confirmation: no "yes please", "go ahead", "let's do it", "I agree", "confirm", "proceed", "好的，就这样办", "同意", "确认", "请办理". It never says "I understand" (understanding is only ever stated by planned events). It does not restate the product, the direction or any value, and it gives no goal, risk, fee or reason. In a SINGLISH call a colloquial particle never turns it into agreement.

## Supporting speech (everything outside the markers)

Supporting speech may greet, build rapport, handle logistics and account administration, react briefly, make transitions, and discuss the brief's `discussion_topics`, the client's background and the setting. Harmless family, staff and business-life conversation is welcome where a listed topic covers it. Harmless repetition for scheduling or administration is fine (in a MANDARIN call keep it free of figures, see above).

Supporting speech must never, whatever happens elsewhere in the call and in whatever language:

- state, read back, repeat or confirm any transaction value, identification number, date of birth, product choice or order detail;
- ask whether the client understands any product, risk or disclosure, or confirm that they do (for example no "Does that make sense?" or "您明白了吗？" after product talk);
- ask the client to confirm, approve or proceed with an instruction;
- explain any risk, possible loss, fee, charge or reason for a recommendation;
- ask about investment objectives or willingness to accept losses, or influence what the client says about them;
- promise or guarantee any return or outcome;
- introduce a suspicious or unexplained source of funds, money from or for an undisclosed party, or an attempt to hide who owns an investment;
- let anyone other than the client give or change a transaction instruction;
- base any trade on material non-public or inside information;
- add any order, change, cancellation or new transaction;
- comment on markets, products or investment performance.

The only exceptions to these rules are the single recommendation and the client's mild acknowledgement described above, used when the brief has `advisory_recommendation`, and only in that exact form: they allow naming the product and the action once, and nothing else. They do not allow explaining a reason, confirming the client's objective, mentioning risk, loss, fees or guarantees, stating any order detail, or any other compliance act, and every other rule above still applies to them.

The brief's `must_never_be_expressed_outside_events` lists conditions that must not be expressed anywhere in your own words. The planned events already say everything about the call's compliance and transaction content that should be said.

## Repair guidance (only when the brief has `repair_guidance`)

If the brief contains `repair_guidance`, an earlier version of this same call was read by an independent verifier, who found problems in the supporting speech written around the planned events. Each item describes one problem to avoid in this new version. Write a fresh, complete dialogue (not an edit of the old one) in which none of those problems occurs. The guidance only constrains your own words: it never changes an event, a marker rule, a placement constraint or any other rule above, and it never asks you to mention the verification. If an item seems to conflict with a rule above, follow the rule above.

## Length

Count the length in the unit given by `call.length.unit`: `words` (whitespace-separated words), `han_chars` (Chinese characters only) or `han_chars_plus_words` (Chinese characters plus English words). Aim for a total within `call.length.target` (the event texts count towards it). A dialogue outside `call.length.accepted` is rejected.

## Output

Return only the JSON object with `turns`. No commentary, headings or stage directions inside any turn: every `text` is exactly what that speaker says.
