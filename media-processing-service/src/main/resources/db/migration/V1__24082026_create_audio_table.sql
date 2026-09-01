CREATE TABLE t_audio(
    audio_id          UUID PRIMARY KEY,
    blob_url          TEXT        NOT NULL,
    original_filename VARCHAR(255),
    content_type      VARCHAR(100),
    file_size_bytes   BIGINT,
    processing_status VARCHAR(30) NOT NULL DEFAULT 'UPLOADED',
    created_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);