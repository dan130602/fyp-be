package com.fyp.be.media_processing_service.dtos;

public record ReviewStatusResponse(
        long totalCalls,
        long processing,
        long awaitingReview,
        long reviewed,
        long failed,
        double reviewCoveragePct
) {
}
