package com.fyp.be.media_processing_service.models;

/**
 * IMMEDIATE assets go PENDING -> COMPLETED | FAILED. BATCH assets go
 * PENDING -> QUEUED -> PROCESSING -> COMPLETED | FAILED; the QUEUED / PROCESSING transitions are
 * written by the batch Lambdas, not this service.
 */
public enum MediaAssetStatus {
    PENDING,
    QUEUED,
    PROCESSING,
    COMPLETED,
    FAILED
}
