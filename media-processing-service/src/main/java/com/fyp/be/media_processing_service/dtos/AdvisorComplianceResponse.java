package com.fyp.be.media_processing_service.dtos;

import java.time.LocalDate;
import java.util.List;

public record AdvisorComplianceResponse(
        LocalDate from,
        LocalDate to,
        List<AdvisorComplianceRow> advisors
) {
}
