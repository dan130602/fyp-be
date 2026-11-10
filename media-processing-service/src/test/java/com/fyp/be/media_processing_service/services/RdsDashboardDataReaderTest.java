package com.fyp.be.media_processing_service.services;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import software.amazon.awssdk.services.rdsdata.RdsDataClient;
import software.amazon.awssdk.services.rdsdata.model.BeginTransactionRequest;
import software.amazon.awssdk.services.rdsdata.model.BeginTransactionResponse;
import software.amazon.awssdk.services.rdsdata.model.ExecuteStatementRequest;
import software.amazon.awssdk.services.rdsdata.model.ExecuteStatementResponse;
import software.amazon.awssdk.services.rdsdata.model.Field;
import software.amazon.awssdk.services.rdsdata.model.RollbackTransactionRequest;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RdsDashboardDataReaderTest {

    final RdsDataClient client = mock(RdsDataClient.class);
    final RdsDashboardDataReader reader =
            new RdsDashboardDataReader(new ObjectMapper(), client, "cluster", "secret", "ingestion");
    final LocalDateTime from = LocalDateTime.of(2026, 10, 1, 0, 0);
    final LocalDateTime to = LocalDateTime.of(2026, 11, 1, 0, 0);

    @AfterEach
    void cleanup() {
        SecurityContextHolder.clearContext();
    }

    void login() {
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken("cognito-sub", "", List.of()));
        when(client.beginTransaction(any(BeginTransactionRequest.class)))
                .thenReturn(BeginTransactionResponse.builder().transactionId("tx").build());
    }

    static ExecuteStatementResponse rows(String... json) {
        return ExecuteStatementResponse.builder().records(java.util.Arrays.stream(json)
                .map(value -> List.of(Field.builder().stringValue(value).build())).toList()).build();
    }

    void liveRole(String role, String status, ExecuteStatementResponse calls, ExecuteStatementResponse decisions) {
        when(client.executeStatement(any(ExecuteStatementRequest.class))).thenAnswer(call -> {
            ExecuteStatementRequest request = call.getArgument(0);
            assertEquals("tx", request.transactionId());
            if (request.sql().contains("FROM t_user")) {
                assertEquals("cognito-sub", request.parameters().getFirst().value().stringValue());
                return rows("{\"role\":\"" + role + "\",\"status\":\"" + status + "\"}");
            }
            if (request.sql().contains("FROM t_media_asset m")) return calls;
            if (request.sql().contains("FROM t_checklist_decision")) return decisions;
            return ExecuteStatementResponse.builder().build();
        });
    }

    @Test
    void anonymous_never_queries_aws() {
        assertThrows(AccessDeniedException.class, () -> reader.read(from, to));
        verifyNoInteractions(client);
    }

    @Test
    void role_without_live_dashboard_access_is_denied_and_rolled_back() {
        login();
        liveRole("DEVELOPER", "ACTIVE", rows(), rows());
        assertThrows(AccessDeniedException.class, () -> reader.read(from, to));
        verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
    }

    @Test
    void disabled_admin_is_denied() {
        login();
        liveRole("ADMIN", "DISABLED", rows(), rows());
        assertThrows(AccessDeniedException.class, () -> reader.read(from, to));
        verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
    }

    @Test
    void active_admin_reads_calls_and_decisions_in_read_only_snapshot() {
        login();
        UUID asset = UUID.randomUUID();
        UUID checklist = UUID.randomUUID();
        UUID decision = UUID.randomUUID();
        liveRole("ADMIN", "ACTIVE",
                rows("{\"mediaAssetId\":\"" + asset + "\",\"fileName\":\"call.wav\",\"mediaStatus\":\"COMPLETED\","
                        + "\"createdAt\":\"2026-10-20T09:30:00.123456\",\"advisorId\":null,\"checklistId\":\"" + checklist
                        + "\",\"checklistStatus\":\"REVIEWED\",\"analysedAt\":\"2026-10-20T09:45:00\","
                        + "\"analysis\":{\"final_results\":[{\"item_id\":\"q1\",\"status\":\"PASS\"}],\"error\":null}}"),
                rows("{\"id\":\"" + decision + "\",\"checklist_id\":\"" + checklist + "\",\"checklist_item_id\":\"q1\","
                        + "\"ai_verdict\":\"PASS\",\"decision\":null,\"decided_by\":\"" + UUID.randomUUID()
                        + "\",\"decided_at\":\"2026-10-21T10:00:00\"}"));

        var snapshot = reader.read(from, to);

        assertEquals(1, snapshot.rows().size());
        var row = snapshot.rows().getFirst();
        assertEquals(asset, row.getMediaAssetId());
        assertNull(row.getAdvisorId());
        assertEquals(LocalDateTime.of(2026, 10, 20, 9, 30, 0, 123456000), row.getCreatedAt());
        assertTrue(row.getAnalysis().contains("\"item_id\":\"q1\""));
        assertEquals(1, snapshot.decisions().size());
        assertEquals(decision, snapshot.decisions().getFirst().getId());
        assertNull(snapshot.decisions().getFirst().getDecision());

        var order = inOrder(client);
        order.verify(client).beginTransaction(any(BeginTransactionRequest.class));
        order.verify(client).executeStatement(argThat((ExecuteStatementRequest r) ->
                r.sql().equals("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")));
        order.verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
    }

    @Test
    void midnight_bounds_keep_seconds_for_the_data_api() {
        assertEquals("2026-10-01 00:00:00.000000",
                RdsDashboardDataReader.time("from", from.withDayOfMonth(1)).value().stringValue());
    }

    @Test
    void call_without_output_has_no_analysis() {
        var row = new RdsDashboardDataReader.JsonCallRow(new ObjectMapper().readTree(
                "{\"mediaAssetId\":\"" + UUID.randomUUID() + "\",\"analysedAt\":null,\"analysis\":null}"));
        assertNull(row.getAnalysedAt());
        assertNull(row.getAnalysis());
    }
}
