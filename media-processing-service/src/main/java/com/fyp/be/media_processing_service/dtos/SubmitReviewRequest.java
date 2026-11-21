package com.fyp.be.media_processing_service.dtos;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

import java.util.List;

/** Overrides only - an empty list approves the AI verdicts as-is. */
public record SubmitReviewRequest(
        @Schema(description = "Only the items the reviewer disagrees with. An empty list accepts every AI verdict.")
        @NotNull @Valid List<ReviewOverrideRequest> overrides
) {}
