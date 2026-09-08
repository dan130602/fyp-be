package com.fyp.be.media_processing_service.dtos;

public record PlaybackUrlResponse(
        String url,
        long expiresIn
) {}
