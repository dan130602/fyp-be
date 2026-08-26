package com.fyp.be.ingestion_service.dtos;

public record AudioRequest(
        String blobUrl,
        String originalFilename,
        String contentType,
        Long fileSizeBytes
) {}
