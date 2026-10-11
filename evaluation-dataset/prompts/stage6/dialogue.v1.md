# G6 dialogue authoring instructions (template v1)

You write ONE synthetic private-banking telephone call between a bank employee and a client, in natural conversational Singapore English (en-SG). Everything that matters for the call was fixed by people before you were called: its planned events, their exact wording, their speakers and the constraints on where they occur. You write the conversation around those events. You are not deciding anything about the call and you must not change anything that is fixed. An independent verifier will later read the finished dialogue.

The user message is a JSON brief for one call. Reply with one JSON object that follows the response schema exactly: `{"turns": [{"speaker": "EMPLOYEE" | "CLIENT", "text": "..."}]}`, in spoken order.

## Planned events: you place markers, Python inserts the words

Each entry in the brief's `events` has an exact `text` and a `marker` such as `[[EV_013]]`.

- Place every event exactly once, by writing its marker in the `text` of a turn spoken by that event's `speaker`.
- Write the marker as its own token, with a space (or the start or end of the turn) on both sides, for example `Sure. [[EV_004]] Let me check that for you.` Never attach it to a word or punctuation mark.
- Never write an event's text yourself, and never paraphrase, repeat, summarise or echo it anywhere else in the call. Python replaces each marker with the event's exact text, so write the words around a marker so that they read naturally with that text.
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

## Supporting speech (everything outside the markers)

Supporting speech may greet, build rapport, handle logistics and account administration, react briefly, make transitions, and discuss the brief's `discussion_topics`, the client's background and the setting. Harmless family, staff and business-life conversation is welcome where a listed topic covers it. Harmless repetition for scheduling or administration (a travel date, a callback time) is fine.

Supporting speech must never, whatever happens elsewhere in the call:

- state, read back, repeat or confirm any transaction value, identification number, date of birth, product choice or order detail;
- ask whether the client understands any product, risk or disclosure, or confirm that they do (for example, no "Does that make sense?" or "Is that clear?" after product talk);
- ask the client to confirm, approve or proceed with an instruction;
- explain any risk, possible loss, fee, charge or reason for a recommendation;
- ask about investment objectives or willingness to accept losses, or influence what the client says about them;
- promise or guarantee any return or outcome;
- introduce a suspicious or unexplained source of funds, money from or for an undisclosed party, or an attempt to hide who owns an investment;
- let anyone other than the client give or change a transaction instruction;
- base any trade on material non-public or inside information;
- add any order, change, cancellation or new transaction;
- comment on markets, products or investment performance.

The brief's `must_never_be_expressed_outside_events` lists conditions that must not be expressed anywhere in your own words. The planned events already say everything about the call's compliance and transaction content that should be said.

## Length

Aim for a total length within the brief's `call.target_words` (the event texts count towards it). A dialogue outside `call.accepted_words` is rejected.

## Output

Return only the JSON object with `turns`. No commentary, headings or stage directions inside any turn: every `text` is exactly what that speaker says.
