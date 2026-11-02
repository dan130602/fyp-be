package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.AdvisorComplianceResponse;
import com.fyp.be.media_processing_service.dtos.CallOutcome;
import com.fyp.be.media_processing_service.dtos.ChecklistCompletionResponse;
import com.fyp.be.media_processing_service.dtos.ComplianceOverviewResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallsResponse;
import com.fyp.be.media_processing_service.dtos.OutstandingItemsResponse;
import com.fyp.be.media_processing_service.dtos.ReviewStatusResponse;
import com.fyp.be.media_processing_service.services.DashboardService;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.util.UUID;

/** Org-wide on purpose: unlike /api/media-assets, these figures cover every user's uploads. from and to are inclusive. */
@RestController
@RequestMapping("/api/dashboard")
@PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
public class DashboardController {

    private final DashboardService dashboardService;

    public DashboardController(DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    @GetMapping("/compliance-overview")
    public ResponseEntity<ComplianceOverviewResponse> complianceOverview(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getComplianceOverview(from, to, department));
    }

    @GetMapping("/review-status")
    public ResponseEntity<ReviewStatusResponse> reviewStatus(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getReviewStatus(from, to, department));
    }

    @GetMapping("/outstanding-items")
    public ResponseEntity<OutstandingItemsResponse> outstandingItems(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getOutstandingItems(from, to, department));
    }

    @GetMapping("/checklist-completion")
    public ResponseEntity<ChecklistCompletionResponse> checklistCompletion(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getChecklistCompletion(from, to, department));
    }

    @GetMapping("/advisor-compliance")
    public ResponseEntity<AdvisorComplianceResponse> advisorCompliance(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getAdvisorCompliance(from, to, department));
    }

    @GetMapping("/calls")
    public ResponseEntity<DashboardCallsResponse> calls(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @RequestParam(required = false) String department,
            @RequestParam(required = false) UUID advisorId,
            @RequestParam(required = false) CallOutcome outcome,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(dashboardService.getCalls(from, to, department, advisorId, outcome, page, size));
    }
}
