package com.fyp.be.media_processing_service.dtos;

import java.time.OffsetDateTime;
import java.util.Map;

/** proposedActionsByOwner counts what the AI proposed; whether an action was completed is not recorded anywhere. */
public record OutstandingItemsResponse(
        long awaitingReview,
        OffsetDateTime oldestAwaitingReviewAt,
        long overdueReviews,
        int overdueAfterDays,
        long unreviewedEscalations,
        long unreviewedHolds,
        long reviewedWithUndeterminedItems,
        Map<String, Long> proposedActionsByOwner
) {
}
