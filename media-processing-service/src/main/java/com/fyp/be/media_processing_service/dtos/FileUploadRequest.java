package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;

public record FileUploadRequest(
        @NotBlank String fileName,
        @NotBlank String contentType,
        @Positive Long fileSize
) {}
