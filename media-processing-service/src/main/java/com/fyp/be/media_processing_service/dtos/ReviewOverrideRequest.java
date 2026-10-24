package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.constraints.NotBlank;

public record ReviewOverrideRequest(
        @NotBlank String checklistItemId,
        @NotBlank String decision,
        @NotBlank String comment
) {}
