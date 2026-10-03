package com.fyp.be.media_processing_service.dtos;

import java.time.OffsetDateTime;
import java.util.List;

public record ErrorResponse(
        OffsetDateTime timestamp,
        int status,
        String error,
        String message,
        List<String> details
) {
    public ErrorResponse(int status, String error, String message) {
        this(OffsetDateTime.now(), status, error, message, List.of());
    }

    public ErrorResponse(int status, String error, String message, List<String> details) {
        this(OffsetDateTime.now(), status, error, message, details);
    }
}
