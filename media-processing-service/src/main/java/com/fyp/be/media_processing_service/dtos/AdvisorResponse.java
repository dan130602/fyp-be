package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.Advisor;

import java.util.UUID;

public record AdvisorResponse(
        UUID id,
        String employeeId,
        String fullName,
        String email,
        String department
) {
    public static AdvisorResponse from(Advisor advisor) {
        return new AdvisorResponse(
                advisor.getId(),
                advisor.getEmployeeId(),
                advisor.getFullName(),
                advisor.getEmail(),
                advisor.getDepartment()
        );
    }
}
