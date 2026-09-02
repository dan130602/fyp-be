CREATE TABLE t_media_asset(
    id                 UUID PRIMARY KEY,
    user_id            UUID         NOT NULL REFERENCES t_user(id),
    object_key         TEXT         NOT NULL,
    original_filename  VARCHAR(255),
    content_type       VARCHAR(100),
    file_size_bytes    BIGINT,
    status             VARCHAR(30)  NOT NULL DEFAULT 'PENDING',
    created_at         TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_media_asset_user_id ON t_media_asset(user_id);
