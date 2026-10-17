package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.models.ProcessingMode;

import java.time.OffsetDateTime;
import java.util.UUID;

public record MediaAssetResponse(
        UUID id,
        String originalFilename,
        String contentType,
        Long fileSizeBytes,
        MediaAssetStatus status,
        ProcessingMode processingMode,
        OffsetDateTime createdAt
) {}
