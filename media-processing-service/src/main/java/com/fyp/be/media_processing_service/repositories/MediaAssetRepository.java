package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.MediaAsset;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.List;
import java.util.UUID;

public interface MediaAssetRepository extends JpaRepository<MediaAsset, UUID> {
    List<MediaAsset> findByUserIdOrderByCreatedAtDesc(UUID userId);
    Optional<MediaAsset> findByTranscribeJobName(String transcribeJobName);

    /** Every user's calls created in [from, to), newest first. Aliases are quoted so Postgres keeps their case. */
    @Query(value = """
            SELECT m.id AS "mediaAssetId",
                   m.original_filename AS "fileName",
                   m.status AS "mediaStatus",
                   m.created_at AS "createdAt",
                   a.id AS "advisorId",
                   a.employee_id AS "employeeId",
                   a.full_name AS "fullName",
                   a.email AS "email",
                   a.department AS "department",
                   a.status AS "advisorStatus",
                   c.id AS "checklistId",
                   c.status AS "checklistStatus",
                   o.created_at AS "analysedAt",
                   CAST(jsonb_build_object(
                       'final_results', o.output -> 'final_results',
                       'checklist', o.output -> 'checklist',
                       'final_red_flags', o.output -> 'final_red_flags',
                       'summary', o.output -> 'summary',
                       'error', o.output -> '__error__') AS text) AS "analysis"
            FROM t_media_asset m
            LEFT JOIN t_advisor a ON a.id = m.advisor_id
            LEFT JOIN t_checklist c ON c.media_asset_id = m.id
            LEFT JOIN t_langgraph_output o ON o.checklist_id = c.id
            WHERE m.created_at >= :from AND m.created_at < :to
            ORDER BY m.created_at DESC
            """, nativeQuery = true)
    List<DashboardCallRow> findDashboardCalls(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);
}
