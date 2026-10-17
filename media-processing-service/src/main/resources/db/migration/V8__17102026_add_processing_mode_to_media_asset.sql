-- ADD COLUMN ... NOT NULL DEFAULT backfills every existing row as IMMEDIATE.
ALTER TABLE t_media_asset
    ADD COLUMN processing_mode VARCHAR(20) NOT NULL DEFAULT 'IMMEDIATE';

ALTER TABLE t_media_asset
    ADD CONSTRAINT chk_media_asset_processing_mode
        CHECK (processing_mode IN ('IMMEDIATE', 'BATCH'));

-- Existing rows only ever hold PENDING / COMPLETED / FAILED; QUEUED and PROCESSING are
-- written by the batch enqueuer / worker Lambdas (lambda/transcribe-trigger).
ALTER TABLE t_media_asset
    ADD CONSTRAINT chk_media_asset_status
        CHECK (status IN ('PENDING', 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED'));
