package com.fyp.be.media_processing_service.dtos;

import java.util.UUID;

public record FileUploadResponse(
        UUID id,
        String fileName,
        String objectKey,
        String presignedUrl,
        long expiresIn
) {}
