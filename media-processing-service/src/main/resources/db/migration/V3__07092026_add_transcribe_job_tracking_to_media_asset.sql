ALTER TABLE t_media_asset
    ADD COLUMN transcribe_job_name VARCHAR(255),
    ADD COLUMN transcript_uri      TEXT,
    ADD COLUMN failure_reason      TEXT;

CREATE UNIQUE INDEX idx_media_asset_transcribe_job_name
    ON t_media_asset (transcribe_job_name)
    WHERE transcribe_job_name IS NOT NULL;
