package com.fyp.be.media_processing_service.dtos;

import java.time.OffsetDateTime;
import java.util.UUID;

/** One entry of the decision history; a null decision means the item was reverted to the AI verdict. */
public record ChecklistDecisionResponse(
        String checklistItemId,
        String aiVerdict,
        String decision,
        String comment,
        UUID decidedBy,
        OffsetDateTime decidedAt
) {}
