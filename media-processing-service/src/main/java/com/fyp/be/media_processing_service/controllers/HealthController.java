package com.fyp.be.media_processing_service.controllers;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirements;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "Health", description = "Liveness probe")
@RestController
@RequestMapping("/api/health")
public class HealthController {
    @Operation(summary = "Liveness check", description = "Public. Returns the plain-text body ok.")
    @SecurityRequirements
    @GetMapping
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("ok");
    }
}
