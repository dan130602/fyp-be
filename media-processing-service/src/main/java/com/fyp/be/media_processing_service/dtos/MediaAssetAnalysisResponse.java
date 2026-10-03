package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import tools.jackson.databind.JsonNode;

import java.util.UUID;

public record MediaAssetAnalysisResponse(
        UUID mediaAssetId,
        String fileName,
        MediaAssetStatus assetStatus,
        AnalysisStatus analysisStatus,
        String failureReason,
        JsonNode result
) {}
