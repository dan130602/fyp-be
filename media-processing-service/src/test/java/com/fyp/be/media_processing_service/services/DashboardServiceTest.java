package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AdvisorComplianceRow;
import com.fyp.be.media_processing_service.dtos.CallOutcome;
import com.fyp.be.media_processing_service.dtos.ChecklistCompletionResponse;
import com.fyp.be.media_processing_service.dtos.ComplianceOverviewResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallResponse;
import com.fyp.be.media_processing_service.dtos.DashboardCallsResponse;
import com.fyp.be.media_processing_service.dtos.OutstandingItemsResponse;
import com.fyp.be.media_processing_service.dtos.ReviewStage;
import com.fyp.be.media_processing_service.dtos.ReviewStatusResponse;
import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.repositories.ChecklistDecisionRepository;
import com.fyp.be.media_processing_service.repositories.DashboardCallRow;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DashboardServiceTest {

    @Mock
    private MediaAssetRepository mediaAssetRepository;

    @Mock
    private ChecklistDecisionRepository checklistDecisionRepository;

    private final UUID anna = UUID.randomUUID();
    private final UUID hugh = UUID.randomUUID();

    private record Row(UUID getMediaAssetId, String getFileName, String getMediaStatus, LocalDateTime getCreatedAt,
                       UUID getAdvisorId, String getEmployeeId, String getFullName, String getEmail,
                       String getDepartment, String getAdvisorStatus, UUID getChecklistId, String getChecklistStatus,
                       LocalDateTime getAnalysedAt, String getAnalysis) implements DashboardCallRow {
    }

    private DashboardService service() {
        return new DashboardService(new DashboardDataReader(mediaAssetRepository, checklistDecisionRepository), new ObjectMapper(), 3);
    }

    private void given(Row... rows) {
        when(mediaAssetRepository.findDashboardCalls(any(), any())).thenReturn(List.of(rows));
    }

    private void givenDecisions(ChecklistDecision... decisions) {
        lenient().when(checklistDecisionRepository.findByChecklistIdInOrderByDecidedAtAsc(anyCollection()))
                .thenReturn(List.of(decisions));
    }

    /** verdicts are alternating item id and status. */
    private static String analysis(String... verdicts) {
        StringBuilder results = new StringBuilder();
        StringBuilder checklist = new StringBuilder();
        for (int i = 0; i < verdicts.length; i += 2) {
            String separator = i == 0 ? "" : ",";
            results.append(separator).append("{\"item_id\":\"%s\",\"status\":\"%s\"}".formatted(verdicts[i], verdicts[i + 1]));
            checklist.append(separator).append("{\"id\":\"%s\",\"question\":\"Question %s?\"}".formatted(verdicts[i], verdicts[i]));
        }
        return "{\"final_results\":[%s],\"checklist\":[%s],\"final_red_flags\":null,\"summary\":null,\"error\":null}"
                .formatted(results, checklist);
    }

    private Row call(UUID advisorId, String department, String checklistStatus, String analysis) {
        return new Row(UUID.randomUUID(), "call.mp3", "COMPLETED", LocalDateTime.now(),
                advisorId, advisorId == null ? null : "E100001", advisorId == anna ? "Anna Lyst" : "Hugh Mungus",
                "advisor@clearcall-bank.example", department, "ACTIVE",
                UUID.randomUUID(), checklistStatus, LocalDateTime.now(), analysis);
    }

    private Row reviewed(UUID advisorId, String... verdicts) {
        return call(advisorId, "Private Banking", "REVIEWED", analysis(verdicts));
    }

    private Row awaiting(String analysis) {
        return call(anna, "Private Banking", "SUBMITTED", analysis);
    }

    private Row withAnalysedAt(Row row, LocalDateTime analysedAt) {
        return new Row(row.getMediaAssetId(), row.getFileName(), row.getMediaStatus(), row.getCreatedAt(),
                row.getAdvisorId(), row.getEmployeeId(), row.getFullName(), row.getEmail(), row.getDepartment(),
                row.getAdvisorStatus(), row.getChecklistId(), row.getChecklistStatus(), analysedAt, row.getAnalysis());
    }

    private static ChecklistDecision decision(UUID checklistId, String itemId, String verdict, LocalDateTime at) {
        return ChecklistDecision.builder().id(UUID.randomUUID()).checklistId(checklistId).checklistItemId(itemId)
                .decision(verdict).decidedAt(at).build();
    }

    private ComplianceOverviewResponse overview() {
        return service().getComplianceOverview(null, null, null);
    }

    @Test
    void outcomesSplitReviewedCallsAndIgnoreUnreviewedOnes() {
        given(reviewed(anna, "a", "PASS", "b", "NOT_APPLICABLE"),
                reviewed(anna, "a", "PASS", "b", "PARTIAL"),
                reviewed(anna, "a", "FAIL", "b", "CANNOT_DETERMINE"),
                reviewed(anna, "a", "PASS", "b", "CANNOT_DETERMINE"),
                awaiting(analysis("a", "FAIL")));

        ComplianceOverviewResponse result = overview();

        assertThat(result.reviewedCalls()).isEqualTo(4);
        assertThat(result.compliantCalls()).isEqualTo(1);
        assertThat(result.violationCalls()).isEqualTo(2);
        assertThat(result.inconclusiveCalls()).isEqualTo(1);
        assertThat(result.violationPct()).isEqualTo(50.0);
    }

    @Test
    void reviewerOverrideBeatsTheAiVerdict() {
        Row row = reviewed(anna, "a", "FAIL");
        given(row);
        givenDecisions(decision(row.getChecklistId(), "a", "PASS", LocalDateTime.now()));

        assertThat(overview().compliantCalls()).isEqualTo(1);
    }

    @Test
    void newestDecisionWins() {
        Row row = reviewed(anna, "a", "PASS");
        LocalDateTime earlier = LocalDateTime.now().minusDays(1);
        given(row);
        givenDecisions(decision(row.getChecklistId(), "a", "PASS", earlier),
                decision(row.getChecklistId(), "a", "FAIL", earlier.plusHours(1)));

        assertThat(overview().violationCalls()).isEqualTo(1);
    }

    @Test
    void revertedOverrideFallsBackToTheAiVerdict() {
        Row row = reviewed(anna, "a", "FAIL");
        LocalDateTime earlier = LocalDateTime.now().minusDays(1);
        given(row);
        givenDecisions(decision(row.getChecklistId(), "a", "PASS", earlier),
                decision(row.getChecklistId(), "a", null, earlier.plusHours(1)));

        assertThat(overview().violationCalls()).isEqualTo(1);
    }

    @Test
    void reviewedCallWithNoResultsIsInconclusiveNotCompliant() {
        given(reviewed(anna));

        ComplianceOverviewResponse result = overview();

        assertThat(result.inconclusiveCalls()).isEqualTo(1);
        assertThat(result.compliantCalls()).isZero();
    }

    @Test
    void noCallsGiveZeroRatesRatherThanDividingByZero() {
        given();

        assertThat(overview().compliancePct()).isZero();
    }

    @Test
    void reviewStagesFollowThePrecedenceAndTreatSubmittedAsNotReviewed() {
        Row failedMedia = new Row(UUID.randomUUID(), "f.mp3", "FAILED", LocalDateTime.now(), anna, "E1", "Anna", "e",
                "Private Banking", "ACTIVE", UUID.randomUUID(), "REVIEWED", LocalDateTime.now(), analysis("a", "PASS"));
        Row errored = call(anna, "Private Banking", "REVIEWED",
                "{\"final_results\":null,\"checklist\":null,\"final_red_flags\":null,\"summary\":null,\"error\":\"boom\"}");
        Row processing = withAnalysedAt(call(anna, "Private Banking", "SUBMITTED", null), null);
        given(failedMedia, errored, processing, awaiting(analysis("a", "PASS")), reviewed(anna, "a", "PASS"));

        ReviewStatusResponse result = service().getReviewStatus(null, null, null);

        assertThat(result.totalCalls()).isEqualTo(5);
        assertThat(result.failed()).isEqualTo(2);
        assertThat(result.processing()).isEqualTo(1);
        assertThat(result.awaitingReview()).isEqualTo(1);
        assertThat(result.reviewed()).isEqualTo(1);
        assertThat(result.reviewCoveragePct()).isEqualTo(20.0);
    }

    @Test
    void erroredOutputIsLeftOutOfTheRates() {
        given(call(anna, "Private Banking", "REVIEWED",
                "{\"final_results\":[{\"item_id\":\"a\",\"status\":\"FAIL\"}],\"error\":\"boom\"}"));

        assertThat(overview().reviewedCalls()).isZero();
    }

    @Test
    void outstandingItemsCoverBacklogEscalationsAndActions() {
        LocalDateTime old = LocalDateTime.now().minusDays(5);
        String escalatedOutput = """
                {"final_results":[{"item_id":"a","status":"FAIL"}],"checklist":[],"final_red_flags":[{}],
                 "summary":{"escalation":{"required":true,"hold_transaction":true},"actions_by_owner":{"Compliance":1,"Advisor":2}},
                 "error":null}""";
        given(withAnalysedAt(awaiting(escalatedOutput), old),
                awaiting(analysis("a", "PASS")),
                call(anna, "Private Banking", "REVIEWED", escalatedOutput.replace("\"status\":\"FAIL\"", "\"status\":\"CANNOT_DETERMINE\"")));

        OutstandingItemsResponse result = service().getOutstandingItems(null, null, null);

        assertThat(result.awaitingReview()).isEqualTo(2);
        assertThat(result.oldestAwaitingReviewAt().toLocalDateTime()).isEqualTo(old);
        assertThat(result.overdueReviews()).isEqualTo(1);
        assertThat(result.overdueAfterDays()).isEqualTo(3);
        assertThat(result.unreviewedEscalations()).isEqualTo(1);
        assertThat(result.unreviewedHolds()).isEqualTo(1);
        assertThat(result.reviewedWithUndeterminedItems()).isEqualTo(1);
        assertThat(result.proposedActionsByOwner()).containsEntry("Compliance", 2L).containsEntry("Advisor", 4L);
    }

    @Test
    void completionExcludesNotApplicableAndUnreviewedCalls() {
        given(reviewed(anna, "a", "PASS", "b", "FAIL", "c", "CANNOT_DETERMINE", "d", "NOT_APPLICABLE"),
                awaiting(analysis("a", "CANNOT_DETERMINE")));

        ChecklistCompletionResponse result = service().getChecklistCompletion(null, null, null);

        assertThat(result.determinedItems()).isEqualTo(2);
        assertThat(result.applicableItems()).isEqualTo(3);
        assertThat(result.completionPct()).isEqualTo(66.7);
    }

    @Test
    void advisorRowsSkipCallsWithoutAnAdvisorAndSortByViolationRate() {
        given(reviewed(hugh, "a", "PASS"),
                reviewed(anna, "a", "FAIL"),
                reviewed(anna, "a", "PASS"),
                awaiting(analysis("a", "PASS")),
                reviewed(null, "a", "FAIL"));

        List<AdvisorComplianceRow> rows = service().getAdvisorCompliance(null, null, null).advisors();

        assertThat(rows).extracting(AdvisorComplianceRow::advisorId).containsExactly(anna, hugh);
        assertThat(rows.getFirst().totalCalls()).isEqualTo(3);
        assertThat(rows.getFirst().reviewedCalls()).isEqualTo(2);
        assertThat(rows.getFirst().violationPct()).isEqualTo(50.0);
        assertThat(rows.getFirst().reviewCoveragePct()).isEqualTo(66.7);
    }

    @Test
    void departmentFilterAlsoDropsCallsWithoutAnAdvisor() {
        given(call(anna, "Private Banking", "REVIEWED", analysis("a", "PASS")),
                call(hugh, "Wealth Management", "REVIEWED", analysis("a", "PASS")),
                call(null, null, "REVIEWED", analysis("a", "PASS")));

        assertThat(service().getComplianceOverview(null, null, "Private Banking").reviewedCalls()).isEqualTo(1);
    }

    @Test
    void toDateIsInclusiveAndMissingDatesAreOpenEnded() {
        given();

        service().getReviewStatus(LocalDate.of(2026, 10, 1), LocalDate.of(2026, 10, 31), null);

        verify(mediaAssetRepository).findDashboardCalls(
                LocalDate.of(2026, 10, 1).atStartOfDay(), LocalDate.of(2026, 11, 1).atStartOfDay());
    }

    @Test
    void callsFilterByAdvisorAndOutcomeAndNameTheViolatedQuestions() {
        Row target = reviewed(anna, "a", "PASS", "b", "FAIL");
        given(target, reviewed(anna, "a", "PASS"), reviewed(hugh, "a", "FAIL"));

        DashboardCallsResponse result = service().getCalls(null, null, null, anna, CallOutcome.VIOLATION, 0, 20);

        assertThat(result.totalCalls()).isEqualTo(1);
        DashboardCallResponse call = result.calls().getFirst();
        assertThat(call.mediaAssetId()).isEqualTo(target.getMediaAssetId());
        assertThat(call.reviewStage()).isEqualTo(ReviewStage.REVIEWED);
        assertThat(call.violatedItems()).singleElement().satisfies(item -> {
            assertThat(item.itemId()).isEqualTo("b");
            assertThat(item.question()).isEqualTo("Question b?");
            assertThat(item.verdict()).isEqualTo("FAIL");
        });
    }

    @Test
    void callsPageAndClampOutOfRangeValues() {
        given(reviewed(anna, "a", "PASS"), reviewed(anna, "a", "PASS"), reviewed(anna, "a", "PASS"));

        DashboardCallsResponse second = service().getCalls(null, null, null, null, null, 1, 2);
        DashboardCallsResponse clamped = service().getCalls(null, null, null, null, null, -1, 1000);

        assertThat(second.calls()).hasSize(1);
        assertThat(second.totalCalls()).isEqualTo(3);
        assertThat(clamped.page()).isZero();
        assertThat(clamped.size()).isEqualTo(100);
    }
}
