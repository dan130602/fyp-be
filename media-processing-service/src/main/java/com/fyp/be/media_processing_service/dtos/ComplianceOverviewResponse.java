package com.fyp.be.media_processing_service.dtos;

/** Reviewed calls only; the three counts add up to reviewedCalls. */
public record ComplianceOverviewResponse(
        long reviewedCalls,
        long compliantCalls,
        long violationCalls,
        long inconclusiveCalls,
        double compliancePct,
        double violationPct,
        double inconclusivePct
) {
}
