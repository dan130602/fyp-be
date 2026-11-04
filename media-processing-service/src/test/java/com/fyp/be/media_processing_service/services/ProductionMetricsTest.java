package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.*;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;
import java.time.LocalDateTime;
import java.util.*;
import static org.assertj.core.api.Assertions.assertThat;

class ProductionMetricsTest {
    private final ObjectMapper mapper = new ObjectMapper();
    private final ProductionMetrics metrics = new ProductionMetrics(mapper);
    private final LocalDateTime now = LocalDateTime.of(2026, 10, 31, 10, 0);
    private final UUID checklistId = UUID.randomUUID(), runId = UUID.randomUUID();
    private LanggraphOutput run(UUID id, String output, int minutesAgo) {
        return LanggraphOutput.builder().id(id).checklistId(checklistId).output(output)
                .createdAt(now.minusMinutes(minutesAgo + 1)).updatedAt(now.minusMinutes(minutesAgo)).build();
    }
    private Checklist checklist(UUID reviewedRun) {
        return Checklist.builder().id(checklistId).mediaAssetId(UUID.randomUUID()).checklist("[]")
                .reviewedAt(reviewedRun == null ? null : now).build();
    }
    private ChecklistDecision decision(String item, String ai, String officer, int secondsAgo) {
        return ChecklistDecision.builder().id(UUID.randomUUID()).checklistId(checklistId)
                .checklistItemId(item).aiVerdict(ai).decision(officer).decidedBy(UUID.randomUUID()).decidedAt(now.minusSeconds(secondsAgo)).build();
    }
    @Test void a_saved_error_is_reported_without_inventing_a_successful_prior_attempt() {
        var failed = run(runId, "{\"__error__\":\"AI_REQUEST_UNAVAILABLE\"}", 1);
        var result = mapper.valueToTree(metrics.summarize(List.of(failed), List.of(checklist(runId)), List.of(), 7, now));
        assertThat(result.at("/counts/binaryDecisions").asInt()).isZero();
        assertThat(result.at("/counts/reviewedCalls").asInt()).isZero();
        assertThat(result.at("/counts/failedResults").asInt()).isEqualTo(1);
        assertThat(result.at("/counts/savedResults").asInt()).isEqualTo(1);
        assertThat(result.at("/cost/unknownResults").asInt()).isEqualTo(1);
    }

    @Test void denominators_include_unclear_and_failures_in_full_breach_detection() {
        String output = """
          {"final_results":[
            {"item_id":"A","status":"PASS"},{"item_id":"B","status":"FAIL"},
            {"item_id":"C","status":"PASS"},{"item_id":"D","status":"CANNOT_DETERMINE","execution_status":"prediction_failure"},
            {"item_id":"E","status":"PARTIAL"},{"item_id":"F","status":"FAIL"}]}
          """;
        var result = metrics.summarize(List.of(run(runId, output, 1)), List.of(checklist(runId)), List.of(
                decision("A", "PASS", "FAIL", 50), decision("B", "FAIL", "PASS", 40),
                decision("D", "CANNOT_DETERMINE", "FAIL", 30)), 7, now);
        var json = mapper.valueToTree(result);
        assertThat(json.at("/counts/checks").asInt()).isEqualTo(6);
        assertThat(json.at("/counts/binaryDecisions").asInt()).isEqualTo(4);
        assertThat(json.at("/counts/predictionFailures").asInt()).isEqualTo(1);
        assertThat(json.at("/metrics/reviewedBreachRecall/numerator").asInt()).isEqualTo(1);
        assertThat(json.at("/metrics/reviewedBreachRecall/denominator").asInt()).isEqualTo(2);
        assertThat(json.at("/metrics/reviewedBreachDetection/denominator").asInt()).isEqualTo(3);
        assertThat(json.at("/reviewedBreaches/predictionFailures").asInt()).isEqualTo(1);
    }

    @Test void unreviewed_predictions_never_become_ground_truth() {
        var result = metrics.summarize(List.of(run(runId, "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"FAIL\"}]}", 1)),
                List.of(checklist(null)), List.of(), 7, now);
        var json = mapper.valueToTree(result);
        assertThat(json.at("/metrics/decisionCoverage/percentage").asDouble()).isEqualTo(100);
        assertThat(json.at("/counts/reviewedChecks").asInt()).isZero();
        assertThat(json.at("/metrics/reviewedBreachRecall/percentage").isNull()).isTrue();
        assertThat(json.at("/cost/reportedUsd").isNull()).isTrue();
    }

    @Test void rerun_invalidates_an_older_review() {
        UUID newer = UUID.randomUUID();
        var result = metrics.summarize(List.of(
                run(newer, "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"FAIL\"}],\"observability\":{\"cost\":{\"amount\":2}}}", 1)),
                List.of(Checklist.builder().id(checklistId).checklist("[]").reviewedAt(now.minusMinutes(3)).build()), List.of(decision("A", "PASS", "FAIL", 20)), 7, now);
        var json = mapper.valueToTree(result);
        assertThat(json.at("/counts/calls").asInt()).isEqualTo(1);
        assertThat(json.at("/counts/checks").asInt()).isEqualTo(1);
        assertThat(json.at("/counts/reviewedChecks").asInt()).isZero();
        assertThat(json.at("/cost/reportedUsd").asDouble()).isEqualTo(2);
    }

    @Test void comment_edits_and_reverts_do_not_inflate_active_overrides() {
        var first = decision("A", "PASS", "FAIL", 50);
        var comment = decision("A", "PASS", "FAIL", 40);
        var revert = decision("A", "PASS", null, 30);
        var result = metrics.summarize(List.of(run(runId, "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"PASS\"}]}", 1)),
                List.of(checklist(runId)), List.of(first, comment, revert), 7, now);
        var json = mapper.valueToTree(result);
        assertThat(json.at("/overrides/activeItems").asInt()).isZero();
        assertThat(json.at("/overrides/eventCount").asInt()).isEqualTo(2);
        assertThat(json.at("/overrides/recent/0/id").asText()).isEqualTo(revert.getId().toString());
        assertThat(json.at("/matrix/agreedPass").asInt()).isEqualTo(1);
    }

    @Test void duplicate_and_missing_results_are_failures_not_extra_decisions() {
        var result = metrics.summarize(List.of(run(runId, """
          {"checklist":[{"id":"A"},{"id":"B"}],"final_results":[{"item_id":"A","status":"PASS"},{"item_id":"A","status":"PASS"}]}
          """, 1)), List.of(checklist(null)), List.of(), 7, now);
        var json = mapper.valueToTree(result);
        assertThat(json.at("/counts/checks").asInt()).isEqualTo(2);
        assertThat(json.at("/counts/predictionFailures").asInt()).isEqualTo(2);
        assertThat(json.at("/counts/binaryDecisions").asInt()).isZero();
    }

    @Test void empty_window_has_unknown_rates_and_latency() {
        var json = mapper.valueToTree(metrics.summarize(List.of(), List.of(), List.of(), 7, now));
        assertThat(json.at("/metrics/decisionCoverage/percentage").isNull()).isTrue();
        assertThat(json.at("/latency/p95Ms").isNull()).isTrue();
        assertThat(json.at("/overrides/recent").isEmpty()).isTrue();
    }

    @Test void a_shortened_output_checklist_cannot_hide_a_missing_configured_question() {
        var configured = Checklist.builder().id(checklistId).checklist("""
          [{"id":"A","stage":2},{"id":"B","stage":2}]
          """).build();
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId, """
          {"checklist":[{"id":"A","stage":2}],
           "final_results":[{"item_id":"A","status":"PASS"}]}
          """, 1)), List.of(configured), List.of(), 7, now));
        assertThat(json.at("/metrics/decisionCoverage/numerator").asInt()).isEqualTo(1);
        assertThat(json.at("/metrics/decisionCoverage/denominator").asInt()).isEqualTo(2);
        assertThat(json.at("/metrics/decisionCoverage/percentage").asDouble()).isEqualTo(50);
        assertThat(json.at("/counts/predictionFailures").asInt()).isEqualTo(1);
    }
    @Test void production_fp_fn_use_final_officer_decisions_with_breach_as_the_positive_class() {
        var output = """
          {"final_results":[{"item_id":"TN","status":"PASS"},{"item_id":"FN","status":"PASS"},
            {"item_id":"FP","status":"FAIL"},{"item_id":"TP","status":"FAIL"},
            {"item_id":"UNCLEAR","status":"CANNOT_DETERMINE"}]}
          """;
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId, output, 1)), List.of(checklist(runId)), List.of(
                decision("FN", "PASS", "FAIL", 30), decision("FP", "FAIL", "PASS", 20),
                decision("UNCLEAR", "CANNOT_DETERMINE", "FAIL", 10)), 7, now));
        assertThat(json.path("source").asText()).isEqualTo("production");
        assertThat(json.at("/matrix/agreedPass").asInt()).isEqualTo(1);
        assertThat(json.at("/matrix/agreedBreach").asInt()).isEqualTo(1);
        assertThat(json.at("/matrix/aiPassOfficerFail").asInt()).isEqualTo(1); // FN
        assertThat(json.at("/matrix/aiFailOfficerPass").asInt()).isEqualTo(1); // FP
        assertThat(json.at("/matrix/binaryPairs").asInt()).isEqualTo(4);
        assertThat(json.at("/overrides/activeItems").asInt()).isEqualTo(3); // unclear resolution is not FP/FN
        assertThat(json.at("/reviewedBreaches/unclear").asInt()).isEqualTo(1);
        assertThat(json.at("/metrics/reviewedBreachRecall/percentage").asDouble()).isEqualTo(50);
    }
    @Test void reverting_a_live_override_removes_the_false_positive_without_deleting_history() {
        var first = decision("A", "FAIL", "PASS", 30);
        var revert = decision("A", "FAIL", null, 10);
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId,
                "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"FAIL\"}]}", 1)),
                List.of(checklist(runId)), List.of(first, revert), 7, now));
        assertThat(json.at("/matrix/aiFailOfficerPass").asInt()).isZero();
        assertThat(json.at("/matrix/agreedBreach").asInt()).isEqualTo(1);
        assertThat(json.at("/overrides/activeItems").asInt()).isZero();
        assertThat(json.at("/overrides/eventCount").asInt()).isEqualTo(2);
    }

    @Test void a_finalized_review_accepts_unchanged_answers_without_override_rows() {
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId, """
          {"final_results":[{"item_id":"A","status":"PASS"},{"item_id":"B","status":"FAIL"}]}
          """, 1)), List.of(checklist(runId)), List.of(), 7, now));
        assertThat(json.at("/matrix/agreedPass").asInt()).isEqualTo(1);
        assertThat(json.at("/matrix/agreedBreach").asInt()).isEqualTo(1);
        assertThat(json.at("/counts/reviewedChecks").asInt()).isEqualTo(2);
        assertThat(json.at("/overrides/activeItems").asInt()).isZero();
    }

    @Test void an_older_active_override_still_counts_after_a_recent_finalized_review() {
        var override = decision("A", "PASS", "FAIL", 10);
        override.setDecidedAt(now.minusDays(10));
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId,
                "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"PASS\"}]}", 1)),
                List.of(checklist(runId)), List.of(override), 7, now));
        assertThat(json.at("/matrix/aiPassOfficerFail").asInt()).isEqualTo(1);
        assertThat(json.at("/overrides/eventCount").asInt()).isZero();
    }

    @Test void a_conflicting_ai_snapshot_is_excluded_from_reference_totals() {
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId,
                "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"FAIL\"}]}", 1)),
                List.of(checklist(runId)), List.of(decision("A", "PASS", "FAIL", 10)), 7, now));
        assertThat(json.at("/counts/staleReviewItems").asInt()).isEqualTo(1);
        assertThat(json.at("/matrix/binaryPairs").asInt()).isZero();
    }

    @Test void conflicting_latest_decisions_with_identical_timestamps_do_not_get_an_arbitrary_ground_truth() {
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId,
                "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"PASS\"}]}", 1)),
                List.of(checklist(runId)), List.of(decision("A", "PASS", "FAIL", 10), decision("A", "PASS", null, 10)), 7, now));
        assertThat(json.at("/counts/staleReviewItems").asInt()).isEqualTo(1);
        assertThat(json.at("/matrix/binaryPairs").asInt()).isZero();
    }

    @Test void processing_latency_uses_reported_telemetry_instead_of_database_timestamp_differences() {
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId,
                "{\"final_results\":[],\"observability\":{\"duration_ms\":1234}}", 100)), List.of(), List.of(), 7, now));
        assertThat(json.at("/latency/p95Ms").asDouble()).isEqualTo(1234);
        assertThat(json.at("/latency/measuredResults").asInt()).isEqualTo(1);
        assertThat(json.at("/cost/reportedUsd").isNull()).isTrue();
    }
    @Test void simultaneous_comment_only_rows_do_not_duplicate_override_warnings() {
        var json = mapper.valueToTree(metrics.summarize(List.of(run(runId,
                "{\"final_results\":[{\"item_id\":\"A\",\"status\":\"PASS\"}]}", 1)),
                List.of(checklist(runId)), List.of(decision("A", "PASS", "FAIL", 10), decision("A", "PASS", "FAIL", 10)), 7, now));
        assertThat(json.at("/matrix/aiPassOfficerFail").asInt()).isEqualTo(1);
        assertThat(json.at("/overrides/eventCount").asInt()).isEqualTo(1);
    }

}
