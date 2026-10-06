CREATE TABLE t_checklist(
    id              UUID PRIMARY KEY,
    media_asset_id  UUID         NOT NULL UNIQUE REFERENCES t_media_asset(id),
    submitted_by    UUID         REFERENCES t_user(id),
    checklist       JSONB        NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'PENDING_REVIEW',
    created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_checklist_submitted_by ON t_checklist(submitted_by);
