package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.UserStatus;

import java.util.UUID;

public record UserResponseDto(
        UUID id,
        String email,
        String name,
        Role role,
        UserStatus status
) {}
