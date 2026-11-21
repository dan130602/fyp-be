package com.fyp.be.media_processing_service.dtos;

import io.swagger.v3.oas.annotations.media.Schema;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fyp.be.media_processing_service.models.ProcessingMode;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;

import java.util.List;

public record PresignedUploadRequest(
        @NotEmpty @Valid List<FileUploadRequest> files,
        /** "immediate" | "batch"; omitted means immediate so existing clients are unaffected. */
        @Schema(description = "immediate | batch. Defaults to immediate when omitted.", allowableValues = {"immediate", "batch"})
        @JsonProperty("processing_mode") ProcessingMode processingMode
) {
    public PresignedUploadRequest {
        if (processingMode == null) {
            processingMode = ProcessingMode.IMMEDIATE;
        }
    }
}
