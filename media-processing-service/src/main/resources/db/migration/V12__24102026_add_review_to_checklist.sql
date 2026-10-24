-- review is a JSON array of the reviewer's overrides only (with a server-side snapshot of the AI
-- verdict at submit time); an empty array means the AI review was approved as-is. NULL = never reviewed.
ALTER TABLE t_checklist
    ADD COLUMN review      JSONB,
    ADD COLUMN reviewed_by UUID REFERENCES t_user(id),
    ADD COLUMN reviewed_at TIMESTAMP;
