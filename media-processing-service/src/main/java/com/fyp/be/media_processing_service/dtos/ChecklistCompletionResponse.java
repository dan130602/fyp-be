package com.fyp.be.media_processing_service.dtos;

/** Reviewed calls only. determinedItems reached PASS, FAIL or PARTIAL; applicableItems excludes NOT_APPLICABLE. */
public record ChecklistCompletionResponse(
        long determinedItems,
        long applicableItems,
        double completionPct
) {
}
