package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.ChecklistDecision;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.List;
import java.util.UUID;

public interface ChecklistDecisionRepository extends JpaRepository<ChecklistDecision, UUID> {
    List<ChecklistDecision> findByChecklistIdOrderByDecidedAtAsc(UUID checklistId);
    List<ChecklistDecision> findByChecklistIdInOrderByDecidedAtAsc(Collection<UUID> checklistIds);
}
