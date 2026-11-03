package com.fyp.be.media_processing_service.dtos;

import java.util.UUID;

public record AdvisorComplianceRow(
        UUID advisorId,
        String employeeId,
        String fullName,
        String department,
        String status,
        long totalCalls,
        long reviewedCalls,
        long compliantCalls,
        long violationCalls,
        long inconclusiveCalls,
        long redFlagCalls,
        double compliancePct,
        double violationPct,
        double inconclusivePct,
        double reviewCoveragePct
) {
}
