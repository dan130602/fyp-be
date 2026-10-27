# Codex manual review supplement v1 (G5 and G7)

Read the ENTIRE prepared request: its system instructions, complete call brief,
frozen targets, assigned edge cases, language requirements and response schema.
Review generated evidence against targets; a target is never evidence of itself.
Do not copy an earlier review or manufacture MATCH rows from gold. Cite actual
source text and explain the reasoning. Apply the stage's own response schema;
Python derives the verdict. Use UNCLEAR for unresolved ambiguity rather than
guessing. Never repair content or retry a judgement merely to obtain VALID.

## G5: planned evidence

Assess the accepted G3 scenario and ALL G4 events, including superseded values,
corrections, distractors, negation, language pins and contextual meaning. Separate
exact-substring grounding from semantic support: a quotation can occur verbatim
and still omit a critical negation, magnitude or qualification. Do not apply G6
supporting-speech restrictions to G4's authorized compliance events. Do not judge
a conversation that does not yet exist.

## G7: complete conversation and authoring conformance

Assess the ENTIRE dialogue, all checklist targets, extraction final states,
assigned edge cases and language rows, using the stage's normal instructions.
In ADDITION to that rubric, assess supporting speech against the authoring
restrictions below, even when a violation leaves every checklist answer intact.
This is an explicit additional review scope; historical Claude and Gemini
reports do not retroactively acquire it.

Compare the planned events with their actual occurrences to distinguish G4
evidence from G6-authored surrounding speech. An authorized G4 refusal,
guarantee, correction, instruction or confirmation is not an invented G6 act.
Likewise, an intentional compliance FAIL is content to preserve, not to repair.

Outside planned evidence, do not allow new protected compliance acts or
transaction facts: new disclosures of risk, loss or fees; understanding checks
or claims; investment-objective or loss-acceptance answers; instruction,
execution or final-confirmation commitments; guarantees; changed values,
orders or cancellations; or an unplanned refusal to execute, compliance
escalation or promise to arrange a protected follow-up review. Ordinary
logistical preferences, permissible administration, reactions and transitions
are allowed. Do not ban the words "review", "cannot" or "confirm" mechanically.
Do not invent an obligation for the employee to intervene when gold deliberately
contains misconduct. Check context, speaker, authorization and the act conveyed.

Preserve the existing explicit exception: where advisory_expectation is present,
G6 may supply the one rationale-free product/action recommendation and an optional
noncommittal client reaction. It may not invent an extra commitment or rationale.
A planned instruction in the client's next turn is not automatically a prohibited
G6 reaction. Inspect the planned event and surrounding speech before judging it.

Report an unplanned protected act as a MISMATCH issue with code
CROSS_CHECK_INTERFERENCE, actual turn quotations and repair_scope G6_ONLY.
Name checklist or assignment IDs only if their meaning is affected; otherwise
leave those ID arrays empty (a whole-dialogue authoring issue). Keep unaffected
check rows MATCH. The existing Python verdict includes whole-dialogue issues,
so a gold-preserving authoring violation still blocks acceptance as VALID.
This supplements the base instruction limiting invented-fact findings to changes
of gold, scenario or ambiguity. Do not claim that an unchanged gold answer changed.

Naturalness, repetition, turn size and awkward flow remain advisory quality
findings unless a real protected act, language rule or frozen requirement breaks.
Text length is advisory for the current multilingual policy; do not turn it into
a semantic failure. Neither presence of vocabulary nor structurally valid spans
prove correct meaning or naturalness. Quote evidence for findings, avoid word
blacklists and numerical style quotas, and keep human quality acceptance separate.
