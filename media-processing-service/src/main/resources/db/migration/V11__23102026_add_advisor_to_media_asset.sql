-- Nullable: rows uploaded before advisor selection existed have no advisor.
-- New uploads are required to carry one at the API layer (FileUploadRequest.advisorId).
ALTER TABLE t_media_asset
    ADD COLUMN advisor_id UUID REFERENCES t_advisor(id);

CREATE INDEX idx_media_asset_advisor_id ON t_media_asset(advisor_id);
