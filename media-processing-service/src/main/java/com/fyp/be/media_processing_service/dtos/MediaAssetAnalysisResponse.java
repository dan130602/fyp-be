package com.fyp.be.media_processing_service.dtos;

import io.swagger.v3.oas.annotations.media.Schema;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import tools.jackson.databind.JsonNode;

import java.util.UUID;

public record MediaAssetAnalysisResponse(
        UUID mediaAssetId,
        String fileName,
        MediaAssetStatus assetStatus,
        AnalysisStatus analysisStatus,
        String failureReason,
        @Schema(description = "Raw fyp-ai output, passed through unchanged. Null unless analysisStatus is READY.", implementation = Object.class)
        JsonNode result,
        ChecklistReviewResponse review
) {}
