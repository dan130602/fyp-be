package com.fyp.be.media_processing_service.dtos;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

/** outcome is null until the call is reviewed; advisor is null for uploads that predate advisor selection. */
public record DashboardCallResponse(
        UUID mediaAssetId,
        String fileName,
        OffsetDateTime createdAt,
        AdvisorResponse advisor,
        ReviewStage reviewStage,
        CallOutcome outcome,
        List<DashboardCallItemResponse> violatedItems,
        int redFlags
) {
}
