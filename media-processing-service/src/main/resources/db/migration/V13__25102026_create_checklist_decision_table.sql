-- Append-only log of reviewer decisions per checklist item; replaces t_checklist.review.
-- A question's current override is its newest row; decision NULL = reverted to the AI verdict.
CREATE TABLE t_checklist_decision(
    id                UUID PRIMARY KEY,
    checklist_id      UUID         NOT NULL REFERENCES t_checklist(id),
    checklist_item_id VARCHAR(100) NOT NULL,
    ai_verdict        VARCHAR(20)  NOT NULL,
    decision          VARCHAR(20),
    comment           TEXT,
    decided_by        UUID         NOT NULL REFERENCES t_user(id),
    decided_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_checklist_decision_checklist ON t_checklist_decision(checklist_id, decided_at);

-- Reviews saved so far keep their state, each override starting with one history entry.
INSERT INTO t_checklist_decision(id, checklist_id, checklist_item_id, ai_verdict, decision, comment, decided_by, decided_at)
SELECT gen_random_uuid(), c.id, o->>'checklistItemId', o->>'aiVerdict', o->>'decision', o->>'comment', c.reviewed_by, c.reviewed_at
FROM t_checklist c, jsonb_array_elements(c.review) AS o
WHERE c.review IS NOT NULL AND jsonb_typeof(c.review) = 'array' AND c.reviewed_by IS NOT NULL AND c.reviewed_at IS NOT NULL;

ALTER TABLE t_checklist DROP COLUMN review;
