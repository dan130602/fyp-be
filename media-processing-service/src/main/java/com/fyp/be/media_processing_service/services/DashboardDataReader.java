package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.repositories.ChecklistDecisionRepository;
import com.fyp.be.media_processing_service.repositories.DashboardCallRow;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * Loads the calls created in [from, to) and the decision logs of their reviewed checklists.
 * Default source is the application's own datasource; RdsDashboardDataReader replaces it when
 * app.dashboard.source=rds-data.
 */
@Service
public class DashboardDataReader {

    public record Snapshot(List<DashboardCallRow> rows, List<ChecklistDecision> decisions) {
    }

    private final MediaAssetRepository mediaAssetRepository;
    private final ChecklistDecisionRepository checklistDecisionRepository;

    public DashboardDataReader(MediaAssetRepository mediaAssetRepository,
                               ChecklistDecisionRepository checklistDecisionRepository) {
        this.mediaAssetRepository = mediaAssetRepository;
        this.checklistDecisionRepository = checklistDecisionRepository;
    }

    public Snapshot read(LocalDateTime from, LocalDateTime to) {
        List<DashboardCallRow> rows = mediaAssetRepository.findDashboardCalls(from, to);
        List<UUID> reviewed = rows.stream()
                .filter(row -> ChecklistStatus.REVIEWED.name().equals(row.getChecklistStatus()))
                .map(DashboardCallRow::getChecklistId)
                .toList();
        return new Snapshot(rows, reviewed.isEmpty() ? List.of()
                : checklistDecisionRepository.findByChecklistIdInOrderByDecidedAtAsc(reviewed));
    }
}
