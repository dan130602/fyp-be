package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;

import java.util.List;

public record FileUploadRequest(
        @NotBlank String fileName,
        @NotBlank String contentType,
        @Positive Long fileSize,
        /** Reviewer's edited checklist for this file - null/empty falls back to the deployment default. */
        @Valid List<ChecklistItemRequest> checklist
) {}
