package com.fyp.be.media_processing_service.dtos;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

public record ChecklistReviewResponse(
        UUID reviewedBy,
        LocalDateTime reviewedAt,
        List<ReviewOverrideResponse> overrides
) {}
