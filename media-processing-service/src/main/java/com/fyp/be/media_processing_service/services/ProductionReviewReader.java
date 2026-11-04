package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** SELECT-only reporting over the existing V13 schema. Never saves entities or invokes AI. */
@Service
public class ProductionReviewReader {
    private final NamedParameterJdbcTemplate jdbc;

    public ProductionReviewReader(NamedParameterJdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public record Snapshot(List<LanggraphOutput> outputs, List<Checklist> checklists,
                           List<ChecklistDecision> history) {}
    private record SavedResult(LanggraphOutput output, Checklist checklist) {}

    // Review activity brings an older AI result into the cohort without rerunning it.
    static final String RESULTS = """
        SELECT o.id AS output_id, o.checklist_id, o.output::text AS output_json,
               o.created_at AS output_created_at, o.updated_at AS output_updated_at,
               c.media_asset_id, c.checklist::text AS checklist_json, c.reviewed_at
        FROM t_langgraph_output o
        JOIN t_checklist c ON c.id = o.checklist_id
        WHERE o.updated_at <= :now
          AND (o.updated_at >= :since OR c.reviewed_at BETWEEN :since AND :now)
        ORDER BY o.updated_at DESC, o.id
        """;
    static final String DECISIONS = """
        SELECT id, checklist_id, checklist_item_id, ai_verdict, decision, decided_by, decided_at
        FROM t_checklist_decision
        WHERE checklist_id IN (:ids) AND decided_at <= :now
        ORDER BY decided_at ASC, id ASC
        """;

    private static LocalDateTime time(ResultSet row, String column) throws SQLException {
        Timestamp value = row.getTimestamp(column);
        return value == null ? null : value.toLocalDateTime();
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Snapshot read(int days, LocalDateTime now) {
        var results = jdbc.query(RESULTS, Map.of("since", now.minusDays(days), "now", now), (row, index) -> {
            UUID checklistId = row.getObject("checklist_id", UUID.class);
            return new SavedResult(
                LanggraphOutput.builder().id(row.getObject("output_id", UUID.class))
                    .checklistId(checklistId).output(row.getString("output_json"))
                    .createdAt(time(row, "output_created_at")).updatedAt(time(row, "output_updated_at")).build(),
                Checklist.builder().id(checklistId).mediaAssetId(row.getObject("media_asset_id", UUID.class))
                    .checklist(row.getString("checklist_json")).reviewedAt(time(row, "reviewed_at")).build());
        });
        if (results.isEmpty()) return new Snapshot(List.of(), List.of(), List.of());
        var ids = results.stream().map(result -> result.checklist().getId()).toList();
        var history = jdbc.query(DECISIONS, Map.of("ids", ids, "now", now), (row, index) ->
            ChecklistDecision.builder().id(row.getObject("id", UUID.class))
                .checklistId(row.getObject("checklist_id", UUID.class))
                .checklistItemId(row.getString("checklist_item_id")).aiVerdict(row.getString("ai_verdict"))
                .decision(row.getString("decision")).decidedBy(row.getObject("decided_by", UUID.class))
                .decidedAt(time(row, "decided_at")).build());
        return new Snapshot(results.stream().map(SavedResult::output).toList(),
                results.stream().map(SavedResult::checklist).toList(), history);
    }
}
