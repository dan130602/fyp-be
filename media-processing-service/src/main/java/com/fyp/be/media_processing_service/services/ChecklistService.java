package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ChecklistItemRequest;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * Owns t_checklist (per-asset, possibly reviewer-edited checklist) and t_langgraph_output
 * (raw fyp-ai response, 1:1 with a Checklist).
 */
@Service
public class ChecklistService {

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
}
