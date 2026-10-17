package com.fyp.be.media_processing_service.models;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Locale;

/**
 * IMMEDIATE: uploaded to the main bucket, whose S3 event starts Transcribe straight away.
 * BATCH: uploaded to the separate batch bucket (no S3 trigger) and picked up by the nightly
 * enqueuer. Stored uppercase like MediaAssetStatus; the API uses "immediate" / "batch".
 */
public enum ProcessingMode {
    IMMEDIATE,
    BATCH;

    @JsonValue
    public String value() {
        return name().toLowerCase(Locale.ROOT);
    }

    @JsonCreator
    public static ProcessingMode fromValue(String value) {
        if (value == null) {
            return IMMEDIATE;
        }
        return switch (value.toLowerCase(Locale.ROOT)) {
            case "immediate" -> IMMEDIATE;
            case "batch" -> BATCH;
            default -> throw new IllegalArgumentException("Unknown processing_mode: " + value);
        };
    }
}
