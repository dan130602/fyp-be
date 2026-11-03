package com.fyp.be.media_processing_service.dtos;

import java.util.List;

public record DashboardCallsResponse(
        List<DashboardCallResponse> calls,
        int page,
        int size,
        long totalCalls
) {
}
