package com.fyp.be.media_processing_service.dtos;

public record FileUploadResponse(
        String fileName,
        String objectKey,
        String presignedUrl,
        long expiresIn
) {}
