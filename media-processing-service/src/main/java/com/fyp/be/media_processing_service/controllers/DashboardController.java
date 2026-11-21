package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.AdvisorComplianceResponse;
import com.fyp.be.media_processing_service.dtos.CallOutcome;
import com.fyp.be.media_processing_service.dtos.ChecklistCompletionResponse;
import com.fyp.be.media_processing_service.dtos.ComplianceOverviewResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallsResponse;
import com.fyp.be.media_processing_service.dtos.OutstandingItemsResponse;
import com.fyp.be.media_processing_service.dtos.ReviewStatusResponse;
import com.fyp.be.media_processing_service.services.DashboardService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
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
@Tag(name = "Dashboard", description = "Org-wide compliance figures; unlike /api/media-assets they cover every user's uploads")
@RestController
@RequestMapping("/api/dashboard")
@PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'MANAGER', 'COMPLIANCE_OFFICER')")
public class DashboardController {

    private final DashboardService dashboardService;

    public DashboardController(DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    @Operation(summary = "Compliance overview", description = "Reviewed calls only. Compliant, violation and inconclusive counts add up to reviewedCalls.")
    @GetMapping("/compliance-overview")
    public ResponseEntity<ComplianceOverviewResponse> complianceOverview(
            @Parameter(description = "Inclusive start date (YYYY-MM-DD), filtered on upload time. Omitted means no lower bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @Parameter(description = "Inclusive end date (YYYY-MM-DD), filtered on upload time. Omitted means no upper bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @Parameter(description = "Exact match on the advisor's department. Blank means all departments.")
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getComplianceOverview(from, to, department));
    }

    @Operation(summary = "Review status counts", description = "Calls by review stage (processing, awaiting review, reviewed, failed) plus review coverage.")
    @GetMapping("/review-status")
    public ResponseEntity<ReviewStatusResponse> reviewStatus(
            @Parameter(description = "Inclusive start date (YYYY-MM-DD), filtered on upload time. Omitted means no lower bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @Parameter(description = "Inclusive end date (YYYY-MM-DD), filtered on upload time. Omitted means no upper bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @Parameter(description = "Exact match on the advisor's department. Blank means all departments.")
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getReviewStatus(from, to, department));
    }

    @Operation(summary = "Outstanding review work", description = "Calls awaiting review, overdue reviews (DASHBOARD_OVERDUE_REVIEW_DAYS), unreviewed escalations and holds, and the AI's proposed actions by owner.")
    @GetMapping("/outstanding-items")
    public ResponseEntity<OutstandingItemsResponse> outstandingItems(
            @Parameter(description = "Inclusive start date (YYYY-MM-DD), filtered on upload time. Omitted means no lower bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @Parameter(description = "Inclusive end date (YYYY-MM-DD), filtered on upload time. Omitted means no upper bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @Parameter(description = "Exact match on the advisor's department. Blank means all departments.")
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getOutstandingItems(from, to, department));
    }

    @Operation(summary = "Checklist completion", description = "Items on reviewed calls only. Not-applicable items are left out of applicableItems; determinedItems excludes CANNOT_DETERMINE.")
    @GetMapping("/checklist-completion")
    public ResponseEntity<ChecklistCompletionResponse> checklistCompletion(
            @Parameter(description = "Inclusive start date (YYYY-MM-DD), filtered on upload time. Omitted means no lower bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @Parameter(description = "Inclusive end date (YYYY-MM-DD), filtered on upload time. Omitted means no upper bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @Parameter(description = "Exact match on the advisor's department. Blank means all departments.")
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getChecklistCompletion(from, to, department));
    }

    @Operation(summary = "Per-advisor compliance", description = "Advisors with at least one call in range, highest violation percentage first, then by name.")
    @GetMapping("/advisor-compliance")
    public ResponseEntity<AdvisorComplianceResponse> advisorCompliance(
            @Parameter(description = "Inclusive start date (YYYY-MM-DD), filtered on upload time. Omitted means no lower bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @Parameter(description = "Inclusive end date (YYYY-MM-DD), filtered on upload time. Omitted means no upper bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @Parameter(description = "Exact match on the advisor's department. Blank means all departments.")
            @RequestParam(required = false) String department) {
        return ResponseEntity.ok(dashboardService.getAdvisorCompliance(from, to, department));
    }

    @Operation(summary = "List calls", description = "Paginated, newest first. Outcome is null until the call is reviewed; violatedItems are items whose effective verdict (reviewer override, else AI) is FAIL or PARTIAL.")
    @GetMapping("/calls")
    public ResponseEntity<DashboardCallsResponse> calls(
            @Parameter(description = "Inclusive start date (YYYY-MM-DD), filtered on upload time. Omitted means no lower bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate from,
            @Parameter(description = "Inclusive end date (YYYY-MM-DD), filtered on upload time. Omitted means no upper bound.")
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate to,
            @Parameter(description = "Exact match on the advisor's department. Blank means all departments.")
            @RequestParam(required = false) String department,
            @Parameter(description = "Only calls attributed to this advisor.")
            @RequestParam(required = false) UUID advisorId,
            @Parameter(description = "Only reviewed calls have an outcome, so this filter returns only reviewed calls.")
            @RequestParam(required = false) CallOutcome outcome,
            @Parameter(description = "Zero-based page index. Negative values are treated as 0.")
            @RequestParam(defaultValue = "0") int page,
            @Parameter(description = "Page size, clamped to 1-100.")
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(dashboardService.getCalls(from, to, department, advisorId, outcome, page, size));
    }
}
