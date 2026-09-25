package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Owns t_checklist (per-asset, possibly reviewer-edited checklist) and t_langgraph_output
 * (raw fyp-ai response, 1:1 with a Checklist).
 */
@Service
public class ChecklistService {

    private final ChecklistRepository checklistRepository;
    private final LanggraphOutputRepository langgraphOutputRepository;
    private final String defaultChecklistJson;

    public ChecklistService(
            ChecklistRepository checklistRepository,
            LanggraphOutputRepository langgraphOutputRepository,
            @Value("${app.checklist-json}") String defaultChecklistJson
    ) {
        this.checklistRepository = checklistRepository;
        this.langgraphOutputRepository = langgraphOutputRepository;
        this.defaultChecklistJson = defaultChecklistJson;
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
