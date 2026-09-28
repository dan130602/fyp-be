package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.constraints.NotBlank;

public record ChecklistItemRequest(
        @NotBlank String id,
        @NotBlank String question
) {}
