package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ChecklistItemRequest;
import com.fyp.be.media_processing_service.dtos.ChecklistItemResponse;
import com.fyp.be.media_processing_service.dtos.ChecklistReviewResponse;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideRequest;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideResponse;
import com.fyp.be.media_processing_service.exceptions.AnalysisNotReadyException;
import com.fyp.be.media_processing_service.exceptions.InvalidReviewException;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Owns t_checklist (per-asset, possibly reviewer-edited checklist) and t_langgraph_output
 * (raw fyp-ai response, 1:1 with a Checklist).
 */
@Service
public class ChecklistService {

    // Verdict vocabulary of the AI pipeline (final_results[].status); reviewer decisions use the same one.
    private static final Set<String> VERDICTS = Set.of("PASS", "FAIL", "PARTIAL", "CANNOT_DETERMINE", "NOT_APPLICABLE");

    private final ChecklistRepository checklistRepository;
    private final LanggraphOutputRepository langgraphOutputRepository;
    private final ObjectMapper objectMapper;
    private final String defaultChecklistJson;

    public ChecklistService(
            ChecklistRepository checklistRepository,
            LanggraphOutputRepository langgraphOutputRepository,
            ObjectMapper objectMapper,
            @Value("${app.checklist-json}") String defaultChecklistJson
    ) {
        this.checklistRepository = checklistRepository;
        this.langgraphOutputRepository = langgraphOutputRepository;
        this.objectMapper = objectMapper;
        this.defaultChecklistJson = defaultChecklistJson;
    }

    /** The deployment default checklist (app.checklist-json); empty when unset or not a JSON array. */
    public List<ChecklistItemResponse> getDefaultChecklist() {
        if (defaultChecklistJson == null || defaultChecklistJson.isBlank()) return List.of();
        JsonNode root = objectMapper.readTree(defaultChecklistJson);
        if (!root.isArray()) return List.of();
        List<ChecklistItemResponse> items = new ArrayList<>();
        for (JsonNode node : root) {
            items.add(new ChecklistItemResponse(node.path("id").asText(), node.path("question").asText()));
        }
        return items;
    }

    /**
     * Creates the checklist for a newly-uploaded MediaAsset, called at upload time (not lazily
     * at transcript handoff). If the uploader supplied their own checklist items, those are
     * recorded as their submission; otherwise falls back to the deployment default, still
     * attributed to the uploader since they're the one who submitted this upload request.
     */
    @Transactional
    public Checklist createForUpload(UUID mediaAssetId, UUID submittedBy, List<ChecklistItemRequest> checklistItems) {
        LocalDateTime now = LocalDateTime.now();
        String checklistJson = (checklistItems == null || checklistItems.isEmpty())
                ? defaultChecklistJson
                : objectMapper.writeValueAsString(checklistItems);

        Checklist checklist = Checklist.builder()
                .id(UUID.randomUUID())
                .mediaAssetId(mediaAssetId)
                .submittedBy(submittedBy)
                .checklist(checklistJson)
                .status(ChecklistStatus.SUBMITTED)
                .createdAt(now)
                .updatedAt(now)
                .build();
        return checklistRepository.save(checklist);
    }

    /**
     * Returns this asset's checklist row, creating one seeded from the deployment-wide
     * default (app.checklist-json) if nobody has reviewed/edited this asset's checklist yet.
     * Called from TranscriptHandoffService.publish, inside the same transaction as the
     * triggering MediaAsset status update.
     */
    @Transactional
    public Checklist getOrCreateForAsset(UUID mediaAssetId) {
        return checklistRepository.findByMediaAssetId(mediaAssetId)
                .orElseGet(() -> {
                    LocalDateTime now = LocalDateTime.now();
                    Checklist checklist = Checklist.builder()
                            .id(UUID.randomUUID())
                            .mediaAssetId(mediaAssetId)
                            .submittedBy(null)
                            .checklist(defaultChecklistJson)
                            .status(ChecklistStatus.PENDING_REVIEW)
                            .createdAt(now)
                            .updatedAt(now)
                            .build();
                    return checklistRepository.save(checklist);
                });
    }

    /** Called by ComplianceAnalysisService once fyp-ai returns a result for this checklist. */
    @Transactional
    public void saveLanggraphOutput(UUID checklistId, String output) {
        LocalDateTime now = LocalDateTime.now();
        LanggraphOutput row = langgraphOutputRepository.findByChecklistId(checklistId)
                .map(existing -> {
                    existing.setOutput(output);
                    existing.setUpdatedAt(now);
                    return existing;
                })
                .orElseGet(() -> LanggraphOutput.builder()
                        .id(UUID.randomUUID())
                        .checklistId(checklistId)
                        .output(output)
                        .createdAt(now)
                        .updatedAt(now)
                        .build());
        langgraphOutputRepository.save(row);
    }

    /**
     * Records the reviewer's overrides of the AI verdicts (empty list = AI approved as-is),
     * replacing any earlier review. Each override's aiVerdict is snapshotted here from the stored
     * AI output, since saveLanggraphOutput overwrites it on an AI re-run.
     */
    @Transactional
    public ChecklistReviewResponse submitReview(UUID checklistId, UUID reviewerId, List<ReviewOverrideRequest> overrides) {
        Checklist checklist = checklistRepository.findById(checklistId)
                .orElseThrow(() -> new ResourceNotFoundException("Checklist not found: " + checklistId));

        LanggraphOutput output = langgraphOutputRepository.findByChecklistId(checklistId)
                .orElseThrow(() -> new AnalysisNotReadyException("AI analysis is not ready for review"));
        JsonNode result;
        try {
            result = objectMapper.readTree(output.getOutput());
        } catch (JacksonException ex) {
            throw new AnalysisNotReadyException("AI analysis is not ready for review");
        }
        if (result.has("__error__")) {
            throw new AnalysisNotReadyException("AI analysis is not ready for review");
        }

        Map<String, String> aiVerdicts = new HashMap<>();
        for (JsonNode finding : result.path("final_results")) {
            aiVerdicts.put(finding.path("item_id").asText(), finding.path("status").asText());
        }

        Set<String> seen = new HashSet<>();
        List<ReviewOverrideResponse> reviewed = new ArrayList<>();
        for (ReviewOverrideRequest override : overrides) {
            String itemId = override.checklistItemId();
            String aiVerdict = aiVerdicts.get(itemId);
            if (aiVerdict == null) {
                throw new InvalidReviewException("Unknown checklist item: " + itemId);
            }
            if (!seen.add(itemId)) {
                throw new InvalidReviewException("Duplicate override for checklist item: " + itemId);
            }
            if (!VERDICTS.contains(override.decision())) {
                throw new InvalidReviewException("Invalid decision for checklist item " + itemId + ": " + override.decision());
            }
            if (override.decision().equals(aiVerdict)) {
                throw new InvalidReviewException("Decision for checklist item " + itemId + " matches the AI verdict");
            }
            reviewed.add(new ReviewOverrideResponse(itemId, aiVerdict, override.decision(), override.comment()));
        }

        LocalDateTime now = LocalDateTime.now();
        checklist.setReview(objectMapper.writeValueAsString(reviewed));
        checklist.setReviewedBy(reviewerId);
        checklist.setReviewedAt(now);
        checklist.setStatus(ChecklistStatus.REVIEWED);
        checklist.setUpdatedAt(now);
        checklistRepository.save(checklist);
        return new ChecklistReviewResponse(reviewerId, now, reviewed);
    }

    /** The stored review of this checklist, or null when it has never been reviewed. */
    public ChecklistReviewResponse toReviewResponse(Checklist checklist) {
        if (checklist.getReview() == null) return null;
        List<ReviewOverrideResponse> overrides = new ArrayList<>();
        for (JsonNode node : objectMapper.readTree(checklist.getReview())) {
            overrides.add(new ReviewOverrideResponse(
                    node.path("checklistItemId").asText(),
                    node.path("aiVerdict").asText(),
                    node.path("decision").asText(),
                    node.path("comment").asText()));
        }
        return new ChecklistReviewResponse(checklist.getReviewedBy(), checklist.getReviewedAt(), overrides);
    }
}
