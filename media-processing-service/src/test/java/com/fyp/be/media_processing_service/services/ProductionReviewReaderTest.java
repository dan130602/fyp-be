package com.fyp.be.media_processing_service.services;

import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.sql.ResultSet;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ProductionReviewReaderTest {
    private final NamedParameterJdbcTemplate jdbc = mock(NamedParameterJdbcTemplate.class);
    private final ProductionReviewReader reader = new ProductionReviewReader(jdbc);
    private final LocalDateTime now = LocalDateTime.of(2026, 11, 2, 2, 0);

    @Test void empty_cohort_does_not_issue_an_invalid_empty_in_query() {
        when(jdbc.query(eq(ProductionReviewReader.RESULTS), anyMap(), any(RowMapper.class))).thenReturn(List.of());
        var data = reader.read(7, now);
        assertThat(data.outputs()).isEmpty();
        assertThat(data.history()).isEmpty();
        verify(jdbc).query(eq(ProductionReviewReader.RESULTS), eq(Map.of("since", now.minusDays(7), "now", now)), any(RowMapper.class));
        verifyNoMoreInteractions(jdbc);
    }

    @Test void existing_schema_maps_original_decision_ids_and_null_reverts_without_writes() throws Exception {
        UUID outputId = UUID.randomUUID(), checklistId = UUID.randomUUID(), decisionId = UUID.randomUUID();
        ResultSet output = mock(ResultSet.class), decision = mock(ResultSet.class);
        when(output.getObject("output_id", UUID.class)).thenReturn(outputId);
        when(output.getObject("checklist_id", UUID.class)).thenReturn(checklistId);
        when(output.getObject("media_asset_id", UUID.class)).thenReturn(UUID.randomUUID());
        when(output.getString("output_json")).thenReturn("{\"final_results\":[]}");
        when(output.getString("checklist_json")).thenReturn("[]");
        when(output.getTimestamp("output_created_at")).thenReturn(Timestamp.valueOf(now.minusDays(10)));
        when(output.getTimestamp("output_updated_at")).thenReturn(Timestamp.valueOf(now.minusDays(8)));
        when(output.getTimestamp("reviewed_at")).thenReturn(Timestamp.valueOf(now));
        when(decision.getObject("id", UUID.class)).thenReturn(decisionId);
        when(decision.getObject("checklist_id", UUID.class)).thenReturn(checklistId);
        when(decision.getString("checklist_item_id")).thenReturn("C01");
        when(decision.getString("ai_verdict")).thenReturn("PASS");
        when(decision.getTimestamp("decided_at")).thenReturn(Timestamp.valueOf(now.minusDays(9)));
        when(jdbc.query(anyString(), anyMap(), any(RowMapper.class))).thenAnswer(invocation -> {
            RowMapper<?> mapper = invocation.getArgument(2);
            return List.of(mapper.mapRow(invocation.getArgument(0).equals(ProductionReviewReader.RESULTS) ? output : decision, 0));
        });
        var data = reader.read(7, now);
        assertThat(data.outputs().getFirst().getId()).isEqualTo(outputId);
        assertThat(data.checklists().getFirst().getReviewedAt()).isEqualTo(now);
        assertThat(data.history().getFirst().getId()).isEqualTo(decisionId);
        assertThat(data.history().getFirst().getDecision()).isNull();
        verify(jdbc).query(eq(ProductionReviewReader.RESULTS), anyMap(), any(RowMapper.class));
        verify(jdbc).query(eq(ProductionReviewReader.DECISIONS), eq(Map.of("ids", List.of(checklistId), "now", now)), any(RowMapper.class));
        verifyNoMoreInteractions(jdbc);
        assertThat(ProductionReviewReader.RESULTS).contains("c.reviewed_at BETWEEN :since AND :now");
        // Old active overrides must still be loaded when a recent review brings a call into the cohort.
        assertThat(ProductionReviewReader.DECISIONS).doesNotContain(":since", "comment", "analysis_run_id");
    }

    @Test void reads_share_one_read_only_database_snapshot() throws Exception {
        var transaction = ProductionReviewReader.class.getMethod("read", int.class, LocalDateTime.class)
                .getAnnotation(Transactional.class);
        assertThat(transaction.readOnly()).isTrue();
        assertThat(transaction.isolation()).isEqualTo(Isolation.REPEATABLE_READ);
    }
}
