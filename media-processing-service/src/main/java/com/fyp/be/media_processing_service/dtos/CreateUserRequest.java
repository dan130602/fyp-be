package com.fyp.be.media_processing_service.dtos;

import com.fyp.be.media_processing_service.models.Role;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;


public record CreateUserRequest(
        @NotBlank @Email String email,
        String name,
        @NotNull Role role
) {}
