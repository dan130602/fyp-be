package com.fyp.be.media_processing_service.dtos;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fyp.be.media_processing_service.models.ProcessingMode;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;

import java.util.List;

public record PresignedUploadRequest(
        @NotEmpty @Valid List<FileUploadRequest> files,
        /** "immediate" | "batch"; omitted means immediate so existing clients are unaffected. */
        @JsonProperty("processing_mode") ProcessingMode processingMode
) {
    public PresignedUploadRequest {
        if (processingMode == null) {
            processingMode = ProcessingMode.IMMEDIATE;
        }
    }
}
