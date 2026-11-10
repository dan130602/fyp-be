package com.fyp.be.media_processing_service.services;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import software.amazon.awssdk.services.rdsdata.RdsDataClient;
import software.amazon.awssdk.services.rdsdata.model.*;
import tools.jackson.databind.ObjectMapper;
import java.time.LocalDateTime;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RdsProductionReviewReaderTest {
    final RdsDataClient client = mock(RdsDataClient.class);
    final RdsProductionReviewReader reader = new RdsProductionReviewReader(null,new ObjectMapper(),client,"cluster","secret","ingestion");
    final LocalDateTime now = LocalDateTime.of(2026,11,5,0,0);
    @AfterEach void cleanup() { SecurityContextHolder.clearContext(); }
    void login() {
        SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken("cognito-sub","",List.of()));
        when(client.beginTransaction(any(BeginTransactionRequest.class)))
            .thenReturn(BeginTransactionResponse.builder().transactionId("tx").build());
    }
    static ExecuteStatementResponse row(String json) {
        return ExecuteStatementResponse.builder().records(List.of(List.of(Field.builder().stringValue(json).build()))).build();
    }
    static ExecuteStatementResponse empty() { return ExecuteStatementResponse.builder().build(); }
    @Test void anonymous_never_queries_aws() {
        assertThrows(AccessDeniedException.class,()->reader.read(7,now)); verifyNoInteractions(client);
    }
    @Test void local_role_does_not_grant_live_access() {
        login(); when(client.executeStatement(any(ExecuteStatementRequest.class)))
            .thenReturn(empty(), row("{\"role\":\"COMPLIANCE_OFFICER\",\"status\":\"ACTIVE\"}"));
        assertThrows(AccessDeniedException.class,()->reader.read(7,now));
        verify(client,times(2)).executeStatement(any(ExecuteStatementRequest.class));
        verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
    }
    @Test void disabled_developer_is_denied() {
        login(); when(client.executeStatement(any(ExecuteStatementRequest.class)))
            .thenReturn(empty(), row("{\"role\":\"DEVELOPER\",\"status\":\"DISABLED\"}"));
        assertThrows(AccessDeniedException.class,()->reader.read(7,now));
        verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
    }
    @Test void active_developer_reads_in_readonly_snapshot_and_rolls_back() {
        login(); when(client.executeStatement(any(ExecuteStatementRequest.class))).thenAnswer(call -> {
            ExecuteStatementRequest req = call.getArgument(0);
            assertEquals("tx",req.transactionId());
            if(req.sql().contains("FROM t_user")) {
                assertEquals("cognito-sub",req.parameters().getFirst().value().stringValue());
                return row("{\"role\":\"DEVELOPER\",\"status\":\"ACTIVE\"}");
            }
            return empty();
        });
        var result=reader.read(7,now); assertTrue(result.outputs().isEmpty());
        var order=inOrder(client);
        order.verify(client).beginTransaction(any(BeginTransactionRequest.class));
        order.verify(client).executeStatement(argThat((ExecuteStatementRequest r)->r.sql().equals("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")));
        verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
        verify(client,never()).commitTransaction(any(CommitTransactionRequest.class));
    }
    @Test void database_error_cannot_fall_back_to_local_metrics() {
        login(); when(client.executeStatement(any(ExecuteStatementRequest.class))).thenThrow(new IllegalStateException("unavailable"));
        assertThrows(IllegalStateException.class,()->reader.read(7,now));
        verify(client).rollbackTransaction(any(RollbackTransactionRequest.class));
    }
}
