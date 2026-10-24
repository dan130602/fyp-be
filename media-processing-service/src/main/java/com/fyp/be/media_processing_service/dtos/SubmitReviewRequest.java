package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

import java.util.List;

/** Overrides only - an empty list approves the AI verdicts as-is. */
public record SubmitReviewRequest(
        @NotNull @Valid List<ReviewOverrideRequest> overrides
) {}
