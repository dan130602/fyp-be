package com.fyp.be.media_processing_service.dtos;

import java.time.LocalDateTime;
import java.util.UUID;

public record AudioResponse(
        UUID audioId,
        String blobUrl,
        String originalFilename,
        String contentType,
        Long fileSizeBytes,
        String processingStatus,
        LocalDateTime createdAt,
        LocalDateTime updatedAt
) {}
