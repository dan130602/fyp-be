package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.Role;
import jakarta.validation.constraints.NotNull;

/**
 * Only accepted on the admin-only role-management endpoint - never bound
 * from a self-service/profile-update request, so users cannot elevate themselves.
 */
public record UpdateUserRoleDto(
        @NotNull Role role
) {}
