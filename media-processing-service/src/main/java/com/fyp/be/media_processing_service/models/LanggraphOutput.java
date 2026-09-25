package com.fyp.be.media_processing_service.models;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Raw fyp-ai /runs/wait response for a Checklist, 1:1. No status column - AI-run
 * completion is signalled purely by this row's existence; human review status
 * lives exclusively on Checklist.status.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "t_langgraph_output")
public class LanggraphOutput {

    @Id
    @Column(name = "id", nullable = false, updatable = false)
    private UUID id;

    @Column(name = "checklist_id", nullable = false, updatable = false)
    private UUID checklistId;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "output", nullable = false, columnDefinition = "jsonb")
    private String output;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;
}
