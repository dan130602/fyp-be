package com.fyp.be.media_processing_service.dtos;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

/** overrides is the current state; history is every logged decision, oldest first. */
public record ChecklistReviewResponse(
        UUID reviewedBy,
        OffsetDateTime reviewedAt,
        List<ReviewOverrideResponse> overrides,
        List<ChecklistDecisionResponse> history
) {}
