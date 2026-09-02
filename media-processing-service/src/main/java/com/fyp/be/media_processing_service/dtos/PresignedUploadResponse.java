package com.fyp.be.media_processing_service.dtos;

import java.util.List;

public record PresignedUploadResponse(
        List<FileUploadResponse> uploads
) {}
