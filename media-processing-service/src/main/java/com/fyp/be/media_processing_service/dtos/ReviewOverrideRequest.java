package com.fyp.be.media_processing_service.dtos;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;

public record ReviewOverrideRequest(
        @NotBlank String checklistItemId,
        @Schema(description = "PASS, FAIL, PARTIAL, CANNOT_DETERMINE or NOT_APPLICABLE; must differ from the AI verdict.")
        @NotBlank String decision,
        @NotBlank String comment
) {}
