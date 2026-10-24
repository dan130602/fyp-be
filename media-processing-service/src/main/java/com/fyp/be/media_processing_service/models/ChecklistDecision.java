package com.fyp.be.media_processing_service.models;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Append-only log of reviewer decisions on a Checklist's items. A question's current override is
 * its newest row; a row with a null decision is a revert to the AI verdict. See ChecklistService.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "t_checklist_decision")
public class ChecklistDecision {

    @Id
    @Column(name = "id", nullable = false, updatable = false)
    private UUID id;

    @Column(name = "checklist_id", nullable = false, updatable = false)
    private UUID checklistId;

    @Column(name = "checklist_item_id", nullable = false, updatable = false, length = 100)
    private String checklistItemId;

    // Snapshot of the AI verdict when the decision was made.
    @Column(name = "ai_verdict", nullable = false, updatable = false, length = 20)
    private String aiVerdict;

    // Null = reverted to the AI verdict.
    @Column(name = "decision", updatable = false, length = 20)
    private String decision;

    @Column(name = "comment", updatable = false, columnDefinition = "text")
    private String comment;

    @Column(name = "decided_by", nullable = false, updatable = false)
    private UUID decidedBy;

    @Column(name = "decided_at", nullable = false, updatable = false)
    private LocalDateTime decidedAt;
}
