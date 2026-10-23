package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.AdvisorResponse;
import com.fyp.be.media_processing_service.services.AdvisorService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/advisors")
public class AdvisorController {

    private final AdvisorService advisorService;

    public AdvisorController(AdvisorService advisorService) {
        this.advisorService = advisorService;
    }

    /** Full active list - small enough (tens of rows) that the upload dialog searches it client-side. */
    @GetMapping
    @PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<List<AdvisorResponse>> list() {
        return ResponseEntity.ok(advisorService.listActive());
    }
}
