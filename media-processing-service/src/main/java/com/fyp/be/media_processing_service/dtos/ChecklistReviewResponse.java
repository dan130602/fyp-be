package com.fyp.be.media_processing_service.dtos;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

public record ChecklistReviewResponse(
        UUID reviewedBy,
        OffsetDateTime reviewedAt,
        List<ReviewOverrideResponse> overrides
) {}
