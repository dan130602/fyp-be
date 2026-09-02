package com.fyp.be.media_processing_service.configs;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.List;

@ConfigurationProperties(prefix = "app.upload")
public record UploadProperties(
        int maxFilesPerRequest,
        long maxFileSizeBytes,
        List<String> allowedContentTypes,
        List<String> allowedExtensions
) {}
