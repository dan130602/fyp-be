package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ApiTime;
import com.fyp.be.media_processing_service.models.*;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import java.time.LocalDateTime;
import java.util.*;

/** Production counts only: officer reviews are a reference, never benchmark ground truth. */
public final class ProductionMetrics {
    private final ObjectMapper mapper;
    public ProductionMetrics(ObjectMapper mapper) { this.mapper = mapper; }

    public static Map<String, Object> map(Object... values) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < values.length; i += 2) result.put((String) values[i], values[i + 1]);
        return result;
    }
    public static Map<String, Object> ratio(long numerator, long denominator) {
        return map("numerator", numerator, "denominator", denominator,
                "percentage", denominator == 0 ? null : 100.0 * numerator / denominator);
    }
    private JsonNode json(String value) {
        try { var parsed = mapper.readTree(value); return parsed == null ? mapper.readTree("{}") : parsed; }
        catch (Exception ignored) { return mapper.readTree("{}"); }
    }
    private static Double number(JsonNode value) {
        return value.isNumber() && Double.isFinite(value.asDouble()) && value.asDouble() >= 0 ? value.asDouble() : null;
    }
    private static String bin(JsonNode result, boolean duplicate) {
        if (duplicate || result == null || !result.path("execution_status").asText("success").equals("success")) return "prediction_failure";
        return switch (result.path("status").asText()) {
            case "PASS", "FAIL" -> "binary";
            case "CANNOT_DETERMINE", "PARTIAL" -> "unclear";
            case "NOT_APPLICABLE" -> "not_applicable";
            default -> "prediction_failure";
        };
    }
    private static Double percentile(List<Double> numbers, double p) {
        if (numbers.isEmpty()) return null;
        var sorted = numbers.stream().sorted().toList();
        return sorted.get(Math.min(sorted.size() - 1, (int) Math.ceil(p * sorted.size()) - 1));
    }

    public Map<String, Object> summarize(List<LanggraphOutput> savedOutputs, List<Checklist> checklists,
                                          List<ChecklistDecision> history, int days, LocalDateTime now) {
        Map<UUID, LanggraphOutput> latest = new LinkedHashMap<>();
        var ordered = savedOutputs.stream().sorted(Comparator.comparing(LanggraphOutput::getUpdatedAt).reversed()
                .thenComparing(LanggraphOutput::getId)).toList();
        // The existing table retains one saved output per checklist, not every historical attempt.
        ordered.forEach(run -> latest.putIfAbsent(run.getChecklistId(), run));
        Map<UUID, Checklist> checks = new HashMap<>();
        checklists.forEach(row -> checks.put(row.getId(), row));
        Map<UUID, Map<String, ChecklistDecision>> current = new HashMap<>();
        Map<UUID, Set<String>> ambiguous = new HashMap<>();
        List<Map<String, Object>> recent = new ArrayList<>();
        for (ChecklistDecision row : history.stream().sorted(Comparator.comparing(ChecklistDecision::getDecidedAt)
                .thenComparing(ChecklistDecision::getId)).toList()) {
            var rows = current.computeIfAbsent(row.getChecklistId(), ignored -> new LinkedHashMap<>());
            var previous = rows.put(row.getChecklistItemId(), row);
            var ties = ambiguous.computeIfAbsent(row.getChecklistId(), ignored -> new HashSet<>());
            if (previous != null && previous.getDecidedAt().equals(row.getDecidedAt())) {
                if (!Objects.equals(previous.getDecision(), row.getDecision())
                        || !Objects.equals(previous.getAiVerdict(), row.getAiVerdict())) ties.add(row.getChecklistItemId());
            } else ties.remove(row.getChecklistItemId());
            String from = previous == null || previous.getDecision() == null
                    || !Objects.equals(previous.getAiVerdict(), row.getAiVerdict())
                    || previous.getDecidedAt().equals(row.getDecidedAt())
                    && !Objects.equals(previous.getDecision(), row.getDecision()) ? row.getAiVerdict() : previous.getDecision();
            String to = row.getDecision() == null ? row.getAiVerdict() : row.getDecision();
            if (!Objects.equals(from, to) && !row.getDecidedAt().isBefore(now.minusDays(days))) {
                Checklist checklist = checks.get(row.getChecklistId());
                recent.add(map("id", row.getId(),
                        "mediaAssetId", checklist == null ? null : checklist.getMediaAssetId(),
                        "checklistItemId", row.getChecklistItemId(), "aiVerdict", row.getAiVerdict(),
                        "from", from, "to", to, "reverted", row.getDecision() == null,
                        "decidedBy", row.getDecidedBy(), "decidedAt", ApiTime.of(row.getDecidedAt()),
                        "priority", row.getDecision() != null && row.getAiVerdict().equals("PASS") && to.equals("FAIL") ? "high" : "normal"));
            }
        }
        long total = 0, binary = 0, unclear = 0, failed = 0, na = 0, reviewed = 0, reviewedCalls = 0;
        long tp = 0, fp = 0, tn = 0, fn = 0, breachTotal = 0, breachUnclear = 0, breachFailed = 0, breachNa = 0;
        long activeOverrides = 0, staleReviewItems = 0;
        for (LanggraphOutput run : latest.values()) {
            JsonNode output = json(run.getOutput());
            Map<String, JsonNode> findings = new LinkedHashMap<>();
            Set<String> duplicates = new HashSet<>();
            for (JsonNode finding : output.path("final_results")) {
                String id = finding.path("item_id").asText();
                if (!id.isBlank() && findings.put(id, finding) != null) duplicates.add(id);
            }
            Set<String> expected = new LinkedHashSet<>();
            Checklist checklist = checks.get(run.getChecklistId());
            // The stored submission is authoritative. A shortened output echo must not
            // silently remove a configured question from the coverage denominator.
            if (checklist != null) for (JsonNode item : json(checklist.getChecklist())) {
                String id = item.path("id").asText();
                if (!id.isBlank() && item.path("stage").asInt(2) == 2) expected.add(id);
            }
            if (expected.isEmpty()) for (JsonNode item : output.path("checklist")) {
                String id = item.path("id").asText(); if (!id.isBlank() && item.path("stage").asInt(2) == 2) expected.add(id);
            }
            if (expected.isEmpty()) expected.addAll(findings.keySet());
            boolean reviewedRun = checklist != null && checklist.getReviewedAt() != null
                    && !checklist.getReviewedAt().isBefore(run.getUpdatedAt())
                    && !checklist.getReviewedAt().isAfter(now) && !output.has("__error__");
            if (!reviewedRun && checklist != null && checklist.getReviewedAt() != null) staleReviewItems += expected.size();
            if (reviewedRun) reviewedCalls++;
            for (String id : expected) {
                JsonNode finding = findings.get(id);
                String outcome = bin(finding, duplicates.contains(id));
                String ai = finding == null ? "CANNOT_DETERMINE" : finding.path("status").asText();
                total++;
                switch (outcome) { case "binary" -> binary++; case "unclear" -> unclear++; case "not_applicable" -> na++; default -> failed++; }
                if (!reviewedRun) continue;
                ChecklistDecision decision = current.getOrDefault(run.getChecklistId(), Map.of()).get(id);
                if (ambiguous.getOrDefault(run.getChecklistId(), Set.of()).contains(id)
                        || decision != null && decision.getDecision() != null && !Objects.equals(decision.getAiVerdict(), ai)) {
                    staleReviewItems++; continue;
                }
                String officer = decision == null || decision.getDecision() == null ? ai : decision.getDecision();
                reviewed++;
                if (!officer.equals(ai)) activeOverrides++;
                if (outcome.equals("binary")) {
                    if (ai.equals("FAIL") && officer.equals("FAIL")) tp++;
                    if (ai.equals("FAIL") && officer.equals("PASS")) fp++;
                    if (ai.equals("PASS") && officer.equals("PASS")) tn++;
                    if (ai.equals("PASS") && officer.equals("FAIL")) fn++;
                }
                if (officer.equals("FAIL")) {
                    breachTotal++;
                    switch (outcome) { case "unclear" -> breachUnclear++; case "prediction_failure" -> breachFailed++; case "not_applicable" -> breachNa++; default -> {} }
                }
            }
        }
        double reportedCost = 0; long priced = 0, legacy = 0, failedAttempts = 0;
        List<Double> durations = new ArrayList<>();
        Map<String, List<Double>> stageDurations = new TreeMap<>();
        Map<String, long[]> modelTokens = new TreeMap<>();
        Map<String, Long> providerErrors = new TreeMap<>();
        for (LanggraphOutput run : savedOutputs) {
            JsonNode body = json(run.getOutput());
            JsonNode telemetry = body.path("observability");
            if (body.has("__error__") || telemetry.path("status").asText().equals("failed")) failedAttempts++;
            Double amount = number(telemetry.path("cost").path("amount"));
            if (amount != null) { reportedCost += amount; priced++; }
            if (!telemetry.isObject()) legacy++;
            Double graphDuration = number(telemetry.path("duration_ms"));
            if (graphDuration != null) durations.add(graphDuration);
            for (JsonNode stage : telemetry.path("stages")) {
                Double duration = number(stage.path("duration_ms"));
                if (duration != null) stageDurations.computeIfAbsent(stage.path("stage").asText("unknown"), k -> new ArrayList<>()).add(duration);
                for (JsonNode error : stage.path("error_codes")) if (error.asText().matches("[0-9]{3}")) providerErrors.merge(error.asText(), 1L, Long::sum);
            }
            for (JsonNode transport : telemetry.path("usage_by_transport")) transport.properties().forEach(entry -> {
                var tokens = modelTokens.computeIfAbsent(entry.getKey(), key -> new long[3]);
                tokens[0] += entry.getValue().path("input_tokens").asLong(0);
                tokens[1] += entry.getValue().path("output_tokens").asLong(0);
                tokens[2] += entry.getValue().path("input_token_details").path("cache_read").asLong(0);
            });
        }
        List<Map<String, Object>> stages = stageDurations.entrySet().stream().map(e -> map("stage", e.getKey(),
                "p50Ms", percentile(e.getValue(), .5), "p95Ms", percentile(e.getValue(), .95), "count", e.getValue().size())).toList();
        List<Map<String, Object>> models = modelTokens.entrySet().stream().map(e -> map("model", e.getKey(),
                "inputTokens", e.getValue()[0], "outputTokens", e.getValue()[1], "cacheReadTokens", e.getValue()[2])).toList();
        long pairs = tp + fp + tn + fn;
        Collections.reverse(recent);
        return map("source", "production", "generatedAt", ApiTime.of(now), "windowDays", days,
                "reference", "Finalized officer decisions; unchanged reviewed answers are accepted AI verdicts. Timestamp and verdict checks exclude stale reviews; this is not independent benchmark ground truth.",
                "population", "Latest saved AI output per checklist analysed or reviewed in this window; all Stage 2 checks. Earlier overwritten attempts are unavailable.",
                "counts", map("calls", latest.size(), "savedResults", savedOutputs.size(), "failedResults", failedAttempts, "checks", total,
                        "binaryDecisions", binary, "unclear", unclear, "predictionFailures", failed, "notApplicable", na,
                        "reviewedCalls", reviewedCalls, "reviewedChecks", reviewed, "unreviewedChecks", total - reviewed, "staleReviewItems", staleReviewItems),
                "metrics", map("decisionCoverage", ratio(binary, total), "predictionFailureRate", ratio(failed, total),
                        "abstentionRate", ratio(unclear, total), "reviewedBreachRecall", ratio(tp, tp + fn),
                        "reviewedBreachPrecision", ratio(tp, tp + fp), "reviewAgreement", ratio(tp + tn, pairs),
                        "reviewedBreachDetection", ratio(tp, breachTotal), "reviewedFalseClearanceRate", ratio(fn, breachTotal)),
                "matrix", map("agreedPass", tn, "aiPassOfficerFail", fn, "aiFailOfficerPass", fp, "agreedBreach", tp, "binaryPairs", pairs),
                "reviewedBreaches", map("total", breachTotal, "detected", tp, "unclear", breachUnclear,
                        "predictionFailures", breachFailed, "notApplicable", breachNa, "falseClearances", fn),
                "cost", map("reportedUsd", priced == 0 ? null : reportedCost, "pricedResults", priced, "unknownResults", savedOutputs.size() - priced,
                        "scope", "Reported chat-cost estimates for the latest saved results in this cohort, including older calls reviewed in the window. Excludes overwritten attempts, embeddings, transcription and hosting; not total production spend."),
                "latency", map("p50Ms", percentile(durations, .5), "p95Ms", percentile(durations, .95),
                        "measuredResults", durations.size(), "unknownResults", savedOutputs.size() - durations.size(),
                        "scope", "Reported AI graph processing time for saved results; excludes queue, writer, upload, transcription and backend network time. Database timestamps are not used as latency."),
                "overrides", map("activeItems", activeOverrides, "reviewedItems", reviewed, "rate", ratio(activeOverrides, reviewed),
                        "eventCount", recent.size(), "highPriorityEvents", recent.stream().filter(e -> e.get("priority").equals("high")).count(),
                        "recent", recent.stream().limit(20).toList()),
                "stages", stages, "models", models, "providerErrors", providerErrors, "legacyResults", legacy);
    }
}
