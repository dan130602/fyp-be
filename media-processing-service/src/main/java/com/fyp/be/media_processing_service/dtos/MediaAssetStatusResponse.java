package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;

import java.time.OffsetDateTime;
import java.util.UUID;

public record MediaAssetStatusResponse(
        UUID id,
        String fileName,
        String objectKey,
        MediaAssetStatus status,
        String transcriptUri,
        String failureReason,
        OffsetDateTime updatedAt
) {
    public static MediaAssetStatusResponse from(MediaAsset asset) {
        return new MediaAssetStatusResponse(
                asset.getId(),
                asset.getOriginalFilename(),
                asset.getObjectKey(),
                asset.getStatus(),
                asset.getTranscriptUri(),
                asset.getFailureReason(),
                ApiTime.of(asset.getUpdatedAt())
        );
    }
}
