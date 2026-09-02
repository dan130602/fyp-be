package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;

import java.util.List;

public record PresignedUploadRequest(
        @NotEmpty @Valid List<FileUploadRequest> files
) {}
