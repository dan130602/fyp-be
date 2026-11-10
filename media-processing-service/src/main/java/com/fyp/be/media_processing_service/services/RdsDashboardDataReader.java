package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.repositories.DashboardCallRow;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.rdsdata.RdsDataClient;
import software.amazon.awssdk.services.rdsdata.model.BeginTransactionRequest;
import software.amazon.awssdk.services.rdsdata.model.ExecuteStatementRequest;
import software.amazon.awssdk.services.rdsdata.model.Field;
import software.amazon.awssdk.services.rdsdata.model.RollbackTransactionRequest;
import software.amazon.awssdk.services.rdsdata.model.SqlParameter;
import software.amazon.awssdk.services.rdsdata.model.TypeHint;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;

/**
 * Local-development option: reads the management dashboard's calls from the Aurora cluster through the
 * RDS Data API instead of the application datasource. Same rows as MediaAssetRepository.findDashboardCalls,
 * read in one read-only transaction that is always rolled back. The caller must also hold an ACTIVE
 * dashboard role in the live t_user table, so a local role alone never exposes live data.
 */
@Service
@Primary
@ConditionalOnProperty(name = "app.dashboard.source", havingValue = "rds-data")
public class RdsDashboardDataReader extends DashboardDataReader {

    static final Set<String> DASHBOARD_ROLES = Set.of("ADMIN", "COMPLIANCE_OFFICER");
    static final int CALL_PAGE = 25;
    static final int DECISION_PAGE = 100;
    static final DateTimeFormatter DATA_API_TIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS");

    // Mirrors findDashboardCalls; results and checklist are trimmed to the fields DashboardService reads.
    static final String CALLS = """
        SELECT jsonb_build_object(
          'mediaAssetId', m.id, 'fileName', m.original_filename, 'mediaStatus', m.status,
          'createdAt', m.created_at, 'advisorId', a.id, 'employeeId', a.employee_id,
          'fullName', a.full_name, 'email', a.email, 'department', a.department,
          'advisorStatus', a.status, 'checklistId', c.id, 'checklistStatus', c.status,
          'analysedAt', o.created_at,
          'analysis', CASE WHEN o.id IS NULL THEN NULL ELSE jsonb_build_object(
             'final_results', (SELECT jsonb_agg(jsonb_build_object('item_id', r->'item_id', 'status', r->'status'))
                 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o.output::jsonb->'final_results') = 'array'
                      THEN o.output::jsonb->'final_results' ELSE '[]'::jsonb END) r),
             'checklist', (SELECT jsonb_agg(jsonb_build_object('id', q->'id', 'question', q->'question'))
                 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o.output::jsonb->'checklist') = 'array'
                      THEN o.output::jsonb->'checklist' ELSE '[]'::jsonb END) q),
             'final_red_flags', o.output::jsonb->'final_red_flags',
             'summary', o.output::jsonb->'summary',
             'error', o.output::jsonb->'__error__') END
        )::text
        FROM t_media_asset m
        LEFT JOIN t_advisor a ON a.id = m.advisor_id
        LEFT JOIN t_checklist c ON c.media_asset_id = m.id
        LEFT JOIN t_langgraph_output o ON o.checklist_id = c.id
        WHERE m.created_at >= :from AND m.created_at < :to
        ORDER BY m.created_at DESC, m.id, c.id
        LIMIT :limit OFFSET :offset
        """;

    // Decision logs of reviewed checklists in range, oldest first; officer comments are not read.
    static final String DECISIONS = """
        SELECT jsonb_build_object('id', d.id, 'checklist_id', d.checklist_id,
          'checklist_item_id', d.checklist_item_id, 'ai_verdict', d.ai_verdict, 'decision', d.decision,
          'decided_by', d.decided_by, 'decided_at', d.decided_at)::text
        FROM t_checklist_decision d
        JOIN t_checklist c ON c.id = d.checklist_id
        JOIN t_media_asset m ON m.id = c.media_asset_id
        WHERE c.status = 'REVIEWED' AND m.created_at >= :from AND m.created_at < :to
        ORDER BY d.decided_at, d.id
        LIMIT :limit OFFSET :offset
        """;

    private final RdsDataClient client;
    private final ObjectMapper mapper;
    private final String cluster;
    private final String secret;
    private final String database;

    @Autowired
    public RdsDashboardDataReader(ObjectMapper mapper,
            @Value("${app.observability.rds.cluster-arn}") String cluster,
            @Value("${app.observability.rds.secret-arn}") String secret,
            @Value("${app.observability.rds.database:ingestion}") String database,
            @Value("${app.aws.region:ap-southeast-1}") String region) {
        this(mapper, RdsDataClient.builder().region(Region.of(region)).build(), cluster, secret, database);
    }

    RdsDashboardDataReader(ObjectMapper mapper, RdsDataClient client, String cluster, String secret, String database) {
        super(null, null);
        this.mapper = mapper;
        this.client = client;
        this.cluster = cluster;
        this.secret = secret;
        this.database = database;
    }

    @Override
    public Snapshot read(LocalDateTime from, LocalDateTime to) {
        var auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated()) {
            throw new AccessDeniedException("Sign in required");
        }
        String tx = client.beginTransaction(BeginTransactionRequest.builder()
                .resourceArn(cluster).secretArn(secret).database(database).build()).transactionId();
        try {
            query(tx, "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY", List.of());
            List<JsonNode> roles = query(tx,
                    "SELECT jsonb_build_object('role', role, 'status', status)::text FROM t_user WHERE cognito_sub = :sub",
                    List.of(text("sub", auth.getName())));
            if (roles.size() != 1 || !"ACTIVE".equals(roles.getFirst().path("status").asText())
                    || !DASHBOARD_ROLES.contains(roles.getFirst().path("role").asText())) {
                throw new AccessDeniedException("An active dashboard role is required in the live database");
            }
            List<SqlParameter> range = List.of(time("from", from), time("to", to));
            List<DashboardCallRow> rows = pages(tx, CALLS, range, CALL_PAGE).stream()
                    .<DashboardCallRow>map(JsonCallRow::new).toList();
            List<ChecklistDecision> decisions = pages(tx, DECISIONS, range, DECISION_PAGE).stream()
                    .map(RdsDashboardDataReader::decision).toList();
            return new Snapshot(rows, decisions);
        } finally {
            client.rollbackTransaction(RollbackTransactionRequest.builder()
                    .resourceArn(cluster).secretArn(secret).transactionId(tx).build());
        }
    }

    private List<JsonNode> query(String tx, String sql, List<SqlParameter> parameters) {
        var response = client.executeStatement(ExecuteStatementRequest.builder().resourceArn(cluster)
                .secretArn(secret).database(database).transactionId(tx).sql(sql).parameters(parameters).build());
        if (!response.hasRecords()) {
            return List.of();
        }
        return response.records().stream().map(row -> mapper.readTree(row.getFirst().stringValue())).toList();
    }

    private List<JsonNode> pages(String tx, String sql, List<SqlParameter> range, int size) {
        List<JsonNode> result = new ArrayList<>();
        for (long offset = 0; ; offset += size) {
            List<SqlParameter> parameters = new ArrayList<>(range);
            parameters.add(number("limit", size));
            parameters.add(number("offset", offset));
            List<JsonNode> page = query(tx, sql, parameters);
            result.addAll(page);
            if (page.size() < size) {
                return result;
            }
        }
    }

    private static ChecklistDecision decision(JsonNode row) {
        return ChecklistDecision.builder()
                .id(uuid(row, "id"))
                .checklistId(uuid(row, "checklist_id"))
                .checklistItemId(row.path("checklist_item_id").asText())
                .aiVerdict(row.path("ai_verdict").asText())
                .decision(string(row, "decision"))
                .decidedBy(uuid(row, "decided_by"))
                .decidedAt(time(row, "decided_at"))
                .build();
    }

    private static SqlParameter text(String name, String value) {
        return SqlParameter.builder().name(name).value(Field.builder().stringValue(value).build()).build();
    }

    private static SqlParameter number(String name, long value) {
        return SqlParameter.builder().name(name).value(Field.builder().longValue(value).build()).build();
    }

    /** The Data API needs seconds; LocalDateTime.toString() drops them at midnight ("2026-10-01T00:00"). */
    static SqlParameter time(String name, LocalDateTime value) {
        return SqlParameter.builder().name(name).typeHint(TypeHint.TIMESTAMP)
                .value(Field.builder().stringValue(DATA_API_TIME.format(value)).build()).build();
    }

    private static String string(JsonNode row, String key) {
        JsonNode value = row.path(key);
        return value.isNull() || value.isMissingNode() ? null : value.asText();
    }

    private static UUID uuid(JsonNode row, String key) {
        String value = string(row, key);
        return value == null ? null : UUID.fromString(value);
    }

    private static LocalDateTime time(JsonNode row, String key) {
        String value = string(row, key);
        return value == null ? null : LocalDateTime.parse(value.replace(' ', 'T'));
    }

    /** One CALLS row exposed through the same projection the JPA query returns. */
    record JsonCallRow(JsonNode row) implements DashboardCallRow {
        @Override public UUID getMediaAssetId() { return uuid(row, "mediaAssetId"); }
        @Override public String getFileName() { return string(row, "fileName"); }
        @Override public String getMediaStatus() { return string(row, "mediaStatus"); }
        @Override public LocalDateTime getCreatedAt() { return time(row, "createdAt"); }
        @Override public UUID getAdvisorId() { return uuid(row, "advisorId"); }
        @Override public String getEmployeeId() { return string(row, "employeeId"); }
        @Override public String getFullName() { return string(row, "fullName"); }
        @Override public String getEmail() { return string(row, "email"); }
        @Override public String getDepartment() { return string(row, "department"); }
        @Override public String getAdvisorStatus() { return string(row, "advisorStatus"); }
        @Override public UUID getChecklistId() { return uuid(row, "checklistId"); }
        @Override public String getChecklistStatus() { return string(row, "checklistStatus"); }
        @Override public LocalDateTime getAnalysedAt() { return time(row, "analysedAt"); }
        @Override public String getAnalysis() {
            JsonNode analysis = row.path("analysis");
            return analysis.isObject() ? analysis.toString() : null;
        }
    }
}
