package com.fyp.be.media_processing_service.dtos;

/** A checklist item with its effective verdict: the reviewer's current decision, otherwise the AI's. */
public record DashboardCallItemResponse(
        String itemId,
        String question,
        String verdict
) {
}
