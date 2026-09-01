package com.fyp.be.media_processing_service.dtos;

public record AudioRequest(
        String blobUrl,
        String originalFilename,
        String contentType,
        Long fileSizeBytes
) {}
