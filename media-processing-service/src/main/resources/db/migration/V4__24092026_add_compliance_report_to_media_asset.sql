ALTER TABLE t_media_asset
    ADD COLUMN compliance_report      JSONB,
    ADD COLUMN compliance_analyzed_at TIMESTAMP;
