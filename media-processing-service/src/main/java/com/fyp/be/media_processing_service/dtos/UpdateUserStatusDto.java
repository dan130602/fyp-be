package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.UserStatus;
import jakarta.validation.constraints.NotNull;

/**
 * Only accepted on the admin-only status-management endpoint - never bound
 * from a self-service/profile-update request, so users cannot reactivate themselves.
 */
public record UpdateUserStatusDto(
        @NotNull UserStatus status
) {}
