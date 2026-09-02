package com.fyp.be.media_processing_service.dtos;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;

public record FirstAdminRequest(
        @NotBlank @Email String email,
        String name
) {}
