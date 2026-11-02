package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AdvisorComplianceResponse;
import com.fyp.be.media_processing_service.dtos.AdvisorComplianceRow;
import com.fyp.be.media_processing_service.dtos.AdvisorResponse;
import com.fyp.be.media_processing_service.dtos.ApiTime;
import com.fyp.be.media_processing_service.dtos.CallOutcome;
import com.fyp.be.media_processing_service.dtos.ChecklistCompletionResponse;
import com.fyp.be.media_processing_service.dtos.ComplianceOverviewResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallItemResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallsResponse;
import com.fyp.be.media_processing_service.dtos.OutstandingItemsResponse;
import com.fyp.be.media_processing_service.dtos.ReviewStage;
import com.fyp.be.media_processing_service.dtos.ReviewStatusResponse;
import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.repositories.ChecklistDecisionRepository;
import com.fyp.be.media_processing_service.repositories.DashboardCallRow;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Org-wide figures for the officer dashboard. Every endpoint is computed from the same per-call
 * view built in load(), so the verdict, outcome and review stage rules live in one place.
 */
@Slf4j
@Service
public class DashboardService {

    private static final Set<String> VIOLATIONS = Set.of("FAIL", "PARTIAL");
    private static final String CANNOT_DETERMINE = "CANNOT_DETERMINE";
    private static final String NOT_APPLICABLE = "NOT_APPLICABLE";
    private static final LocalDate EARLIEST = LocalDate.of(1970, 1, 1);
    private static final LocalDate LATEST = LocalDate.of(9999, 1, 1);
    private static final int MAX_PAGE_SIZE = 100;

    private final MediaAssetRepository mediaAssetRepository;
    private final ChecklistDecisionRepository checklistDecisionRepository;
    private final ObjectMapper objectMapper;
    private final int overdueReviewDays;

    public DashboardService(
            MediaAssetRepository mediaAssetRepository,
            ChecklistDecisionRepository checklistDecisionRepository,
            ObjectMapper objectMapper,
            @Value("${app.dashboard.overdue-review-days}") int overdueReviewDays
    ) {
        this.mediaAssetRepository = mediaAssetRepository;
        this.checklistDecisionRepository = checklistDecisionRepository;
        this.objectMapper = objectMapper;
        this.overdueReviewDays = overdueReviewDays;
    }

    private record Item(String itemId, String question, String verdict) {
    }

    private record Call(DashboardCallRow row, ReviewStage stage, CallOutcome outcome, List<Item> items, int redFlags,
                        boolean escalated, boolean held, Map<String, Long> actionsByOwner) {
    }

    public ComplianceOverviewResponse getComplianceOverview(LocalDate from, LocalDate to, String department) {
        List<Call> calls = load(from, to, department);
        long reviewed = count(calls, ReviewStage.REVIEWED);
        long compliant = count(calls, CallOutcome.COMPLIANT);
        long violation = count(calls, CallOutcome.VIOLATION);
        long inconclusive = count(calls, CallOutcome.INCONCLUSIVE);
        return new ComplianceOverviewResponse(reviewed, compliant, violation, inconclusive,
                pct(compliant, reviewed), pct(violation, reviewed), pct(inconclusive, reviewed));
    }

    public ReviewStatusResponse getReviewStatus(LocalDate from, LocalDate to, String department) {
        List<Call> calls = load(from, to, department);
        long reviewed = count(calls, ReviewStage.REVIEWED);
        return new ReviewStatusResponse(calls.size(), count(calls, ReviewStage.PROCESSING),
                count(calls, ReviewStage.AWAITING_REVIEW), reviewed, count(calls, ReviewStage.FAILED),
                pct(reviewed, calls.size()));
    }

    public OutstandingItemsResponse getOutstandingItems(LocalDate from, LocalDate to, String department) {
        List<Call> calls = load(from, to, department);
        List<Call> awaiting = calls.stream().filter(call -> call.stage() == ReviewStage.AWAITING_REVIEW).toList();
        LocalDateTime overdueBefore = LocalDateTime.now().minusDays(overdueReviewDays);

        Map<String, Long> actionsByOwner = new TreeMap<>();
        calls.forEach(call -> call.actionsByOwner().forEach((owner, count) -> actionsByOwner.merge(owner, count, Long::sum)));

        return new OutstandingItemsResponse(
                awaiting.size(),
                ApiTime.of(awaiting.stream().map(call -> call.row().getAnalysedAt())
                        .min(Comparator.naturalOrder()).orElse(null)),
                awaiting.stream().filter(call -> call.row().getAnalysedAt().isBefore(overdueBefore)).count(),
                overdueReviewDays,
                awaiting.stream().filter(Call::escalated).count(),
                awaiting.stream().filter(Call::held).count(),
                calls.stream().filter(call -> call.stage() == ReviewStage.REVIEWED
                        && call.items().stream().anyMatch(item -> CANNOT_DETERMINE.equals(item.verdict()))).count(),
                actionsByOwner);
    }

    public ChecklistCompletionResponse getChecklistCompletion(LocalDate from, LocalDate to, String department) {
        List<Item> items = load(from, to, department).stream()
                .filter(call -> call.stage() == ReviewStage.REVIEWED)
                .flatMap(call -> call.items().stream())
                .filter(item -> !NOT_APPLICABLE.equals(item.verdict()))
                .toList();
        long determined = items.stream().filter(item -> !CANNOT_DETERMINE.equals(item.verdict())).count();
        return new ChecklistCompletionResponse(determined, items.size(), pct(determined, items.size()));
    }

    /** Advisors with at least one call in range, highest violation rate first. */
    public AdvisorComplianceResponse getAdvisorCompliance(LocalDate from, LocalDate to, String department) {
        Map<UUID, List<Call>> byAdvisor = load(from, to, department).stream()
                .filter(call -> call.row().getAdvisorId() != null)
                .collect(Collectors.groupingBy(call -> call.row().getAdvisorId()));
        List<AdvisorComplianceRow> rows = byAdvisor.values().stream()
                .map(DashboardService::advisorRow)
                .sorted(Comparator.comparingDouble(AdvisorComplianceRow::violationPct).reversed()
                        .thenComparing(AdvisorComplianceRow::fullName))
                .toList();
        return new AdvisorComplianceResponse(from, to, rows);
    }

    public DashboardCallsResponse getCalls(LocalDate from, LocalDate to, String department,
                                           UUID advisorId, CallOutcome outcome, int page, int size) {
        int safePage = Math.max(page, 0);
        int safeSize = Math.clamp(size, 1, MAX_PAGE_SIZE);
        List<Call> matching = load(from, to, department).stream()
                .filter(call -> advisorId == null || advisorId.equals(call.row().getAdvisorId()))
                .filter(call -> outcome == null || outcome == call.outcome())
                .toList();
        List<DashboardCallResponse> calls = matching.stream()
                .skip((long) safePage * safeSize)
                .limit(safeSize)
                .map(DashboardService::callResponse)
                .toList();
        return new DashboardCallsResponse(calls, safePage, safeSize, matching.size());
    }

    private List<Call> load(LocalDate from, LocalDate to, String department) {
        LocalDateTime start = (from == null ? EARLIEST : from).atStartOfDay();
        LocalDateTime end = (to == null ? LATEST : to.plusDays(1)).atStartOfDay();
        List<DashboardCallRow> rows = mediaAssetRepository.findDashboardCalls(start, end).stream()
                .filter(row -> department == null || department.isBlank() || department.equals(row.getDepartment()))
                .toList();

        List<UUID> reviewedChecklists = rows.stream()
                .filter(DashboardService::isReviewed)
                .map(DashboardCallRow::getChecklistId)
                .toList();
        Map<UUID, List<ChecklistDecision>> logs = reviewedChecklists.isEmpty() ? Map.of()
                : checklistDecisionRepository.findByChecklistIdInOrderByDecidedAtAsc(reviewedChecklists).stream()
                        .collect(Collectors.groupingBy(ChecklistDecision::getChecklistId));

        return rows.stream()
                .map(row -> toCall(row, row.getChecklistId() == null ? List.of()
                        : logs.getOrDefault(row.getChecklistId(), List.of())))
                .toList();
    }

    private Call toCall(DashboardCallRow row, List<ChecklistDecision> decisions) {
        if (MediaAssetStatus.FAILED.name().equals(row.getMediaStatus())) {
            return unanalysed(row, ReviewStage.FAILED);
        }
        if (row.getAnalysedAt() == null) {
            return unanalysed(row, ReviewStage.PROCESSING);
        }

        JsonNode analysis;
        try {
            analysis = objectMapper.readTree(row.getAnalysis());
        } catch (JacksonException ex) {
            log.error("Stored AI output for media asset {} is not valid JSON", row.getMediaAssetId(), ex);
            return unanalysed(row, ReviewStage.FAILED);
        }
        if (analysis.hasNonNull("error")) {
            return unanalysed(row, ReviewStage.FAILED);
        }

        boolean reviewed = isReviewed(row);
        List<Item> items = effectiveItems(analysis, decisions);
        JsonNode summary = analysis.path("summary");
        JsonNode escalation = summary.path("escalation");
        Map<String, Long> actionsByOwner = new LinkedHashMap<>();
        for (Map.Entry<String, JsonNode> owner : summary.path("actions_by_owner").properties()) {
            actionsByOwner.put(owner.getKey(), owner.getValue().asLong(0));
        }
        return new Call(row,
                reviewed ? ReviewStage.REVIEWED : ReviewStage.AWAITING_REVIEW,
                reviewed ? outcomeOf(items) : null,
                items,
                analysis.path("final_red_flags").size(),
                escalation.path("required").asBoolean(false),
                escalation.path("hold_transaction").asBoolean(false),
                actionsByOwner);
    }

    private static Call unanalysed(DashboardCallRow row, ReviewStage stage) {
        return new Call(row, stage, null, List.of(), 0, false, false, Map.of());
    }

    private static boolean isReviewed(DashboardCallRow row) {
        return ChecklistStatus.REVIEWED.name().equals(row.getChecklistStatus());
    }

    /** The reviewer's current decision where there is one, otherwise the AI verdict. */
    private static List<Item> effectiveItems(JsonNode analysis, List<ChecklistDecision> decisions) {
        Map<String, String> overrides = new HashMap<>();
        for (ChecklistDecision row : ChecklistService.currentOverrides(decisions)) {
            overrides.put(row.getChecklistItemId(), row.getDecision());
        }
        Map<String, String> questions = new HashMap<>();
        for (JsonNode entry : analysis.path("checklist")) {
            questions.put(entry.path("id").asText(""), entry.path("question").asText(""));
        }

        List<Item> items = new ArrayList<>();
        for (JsonNode result : analysis.path("final_results")) {
            String itemId = result.path("item_id").asText("");
            String question = questions.getOrDefault(itemId, "");
            items.add(new Item(itemId, question.isBlank() ? itemId : question,
                    overrides.getOrDefault(itemId, result.path("status").asText(""))));
        }
        return items;
    }

    /** A call with no results has nothing that shows it was compliant, so it counts as inconclusive. */
    private static CallOutcome outcomeOf(List<Item> items) {
        if (items.stream().anyMatch(item -> VIOLATIONS.contains(item.verdict()))) {
            return CallOutcome.VIOLATION;
        }
        if (items.isEmpty() || items.stream().anyMatch(item -> CANNOT_DETERMINE.equals(item.verdict()))) {
            return CallOutcome.INCONCLUSIVE;
        }
        return CallOutcome.COMPLIANT;
    }

    private static AdvisorComplianceRow advisorRow(List<Call> calls) {
        DashboardCallRow advisor = calls.getFirst().row();
        long reviewed = count(calls, ReviewStage.REVIEWED);
        long compliant = count(calls, CallOutcome.COMPLIANT);
        long violation = count(calls, CallOutcome.VIOLATION);
        long inconclusive = count(calls, CallOutcome.INCONCLUSIVE);
        return new AdvisorComplianceRow(advisor.getAdvisorId(), advisor.getEmployeeId(), advisor.getFullName(),
                advisor.getDepartment(), advisor.getAdvisorStatus(), calls.size(), reviewed, compliant, violation,
                inconclusive, calls.stream().filter(call -> call.redFlags() > 0).count(),
                pct(compliant, reviewed), pct(violation, reviewed), pct(inconclusive, reviewed),
                pct(reviewed, calls.size()));
    }

    private static DashboardCallResponse callResponse(Call call) {
        DashboardCallRow row = call.row();
        AdvisorResponse advisor = row.getAdvisorId() == null ? null : new AdvisorResponse(row.getAdvisorId(),
                row.getEmployeeId(), row.getFullName(), row.getEmail(), row.getDepartment());
        List<DashboardCallItemResponse> violated = call.items().stream()
                .filter(item -> VIOLATIONS.contains(item.verdict()))
                .map(item -> new DashboardCallItemResponse(item.itemId(), item.question(), item.verdict()))
                .toList();
        return new DashboardCallResponse(row.getMediaAssetId(), row.getFileName(), ApiTime.of(row.getCreatedAt()),
                advisor, call.stage(), call.outcome(), violated, call.redFlags());
    }

    private static long count(List<Call> calls, ReviewStage stage) {
        return calls.stream().filter(call -> call.stage() == stage).count();
    }

    private static long count(List<Call> calls, CallOutcome outcome) {
        return calls.stream().filter(call -> call.outcome() == outcome).count();
    }

    private static double pct(long part, long whole) {
        return whole == 0 ? 0.0 : Math.round(1000.0 * part / whole) / 10.0;
    }
}
