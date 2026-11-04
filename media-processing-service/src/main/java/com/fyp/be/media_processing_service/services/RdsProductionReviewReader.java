package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.rdsdata.RdsDataClient;
import software.amazon.awssdk.services.rdsdata.model.*;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import java.time.LocalDateTime;
import java.util.*;

/** Optional private-Aurora access for local reporting. Default deployment still uses JDBC.
 * RDS role is checked independently of the local login DB. No provisioning or writes.
 */
@Service @Primary
@ConditionalOnProperty(name = "app.observability.source", havingValue = "rds-data")
public class RdsProductionReviewReader extends ProductionReviewReader {
    private final RdsDataClient client;
    private final ObjectMapper mapper;
    private final String cluster, secret, database;

    @Autowired
    public RdsProductionReviewReader(NamedParameterJdbcTemplate jdbc, ObjectMapper mapper,
            @Value("${app.observability.rds.cluster-arn}") String cluster,
            @Value("${app.observability.rds.secret-arn}") String secret,
            @Value("${app.observability.rds.database:ingestion}") String database,
            @Value("${app.aws.region:ap-southeast-1}") String region) {
        this(jdbc, mapper, RdsDataClient.builder().region(Region.of(region)).build(), cluster, secret, database);
    }

    RdsProductionReviewReader(NamedParameterJdbcTemplate jdbc, ObjectMapper mapper, RdsDataClient client,
                             String cluster, String secret, String database) {
        super(jdbc); this.mapper = mapper; this.client = client;
        this.cluster = cluster; this.secret = secret; this.database = database;
    }

    static final String COHORT = """
        FROM t_langgraph_output o JOIN t_checklist c ON c.id = o.checklist_id
        WHERE o.updated_at <= :now
        AND (o.updated_at >= :since OR c.reviewed_at BETWEEN :since AND :now)
        """;
    // Exclude evidence, transcripts, comments and question wording at the database boundary.
    static final String OUTPUT = """
        SELECT jsonb_build_object('id',o.id,'checklist_id',o.checklist_id,
          'created_at',o.created_at,'updated_at',o.updated_at,'media_asset_id',c.media_asset_id,
          'reviewed_at',c.reviewed_at,
          'checklist', (SELECT jsonb_agg(jsonb_build_object('id',q->'id','stage',q->'stage'))
             FROM jsonb_array_elements(CASE WHEN jsonb_typeof(c.checklist::jsonb)='array'
                  THEN c.checklist::jsonb ELSE '[]'::jsonb END) q),
          'output', jsonb_build_object(
             'final_results',(SELECT jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
                 'item_id',r->'item_id','status',r->'status','execution_status',r->'execution_status')))
                 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o.output::jsonb->'final_results')='array'
                     THEN o.output::jsonb->'final_results' ELSE '[]'::jsonb END) r),
             'checklist',(SELECT jsonb_agg(jsonb_build_object('id',q->'id','stage',q->'stage'))
                 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(o.output::jsonb->'checklist')='array'
                     THEN o.output::jsonb->'checklist' ELSE '[]'::jsonb END) q),
             'observability',o.output::jsonb->'observability')
             || CASE WHEN o.output::jsonb ? '__error__' THEN '{"__error__":true}'::jsonb ELSE '{}'::jsonb END
          )::text
        """ + COHORT + " ORDER BY o.id LIMIT 1 OFFSET :offset";
    static final String HISTORY = """
        SELECT jsonb_build_object('id',d.id,'checklist_id',d.checklist_id,
          'checklist_item_id',d.checklist_item_id,'ai_verdict',d.ai_verdict,'decision',d.decision,
          'decided_by',d.decided_by,'decided_at',d.decided_at)::text
        FROM t_checklist_decision d WHERE d.decided_at <= :now AND d.checklist_id IN
        (SELECT o.checklist_id
        """ + COHORT + ") ORDER BY d.id LIMIT 100 OFFSET :offset";

    private static SqlParameter value(String name, String value) {
        return SqlParameter.builder().name(name).value(Field.builder().stringValue(value).build()).build();
    }
    private static SqlParameter time(String name, LocalDateTime value) {
        return SqlParameter.builder().name(name).typeHint(TypeHint.TIMESTAMP)
            .value(Field.builder().stringValue(value.toString().replace('T',' ')).build()).build();
    }
    private List<JsonNode> query(String transaction, String sql, List<SqlParameter> parameters) {
        var response = client.executeStatement(ExecuteStatementRequest.builder().resourceArn(cluster)
            .secretArn(secret).database(database).transactionId(transaction).sql(sql).parameters(parameters).build());
        return response.records().stream().map(row -> mapper.readTree(row.getFirst().stringValue())).toList();
    }
    private List<JsonNode> pages(String tx, String sql, List<SqlParameter> params, int size) {
        List<JsonNode> result = new ArrayList<>();
        for (int offset = 0; ; offset += size) {
            var pageParams = new ArrayList<>(params);
            pageParams.add(SqlParameter.builder().name("offset").value(Field.builder().longValue((long)offset).build()).build());
            var page = query(tx, sql, pageParams); result.addAll(page);
            if (page.size() < size) return result;
        }
    }
    private static UUID uuid(JsonNode row, String key) {
        return row.path(key).isNull() || row.path(key).isMissingNode() ? null : UUID.fromString(row.path(key).asText());
    }
    private static LocalDateTime date(JsonNode row, String key) {
        return row.path(key).isNull() || row.path(key).isMissingNode() ? null : LocalDateTime.parse(row.path(key).asText().replace(' ','T'));
    }

    @Override public Snapshot read(int days, LocalDateTime now) {
        if (days < 1 || days > 90) throw new IllegalArgumentException("days must be between 1 and 90");
        var auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated()) throw new AccessDeniedException("Sign in required");
        String tx = client.beginTransaction(BeginTransactionRequest.builder().resourceArn(cluster)
                .secretArn(secret).database(database).build()).transactionId();
        try {
            query(tx, "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY", List.of());
            var roles = query(tx, "SELECT jsonb_build_object('role',role,'status',status)::text FROM t_user WHERE cognito_sub=:sub",
                    List.of(value("sub", auth.getName())));
            if (roles.size()!=1 || !roles.getFirst().path("status").asText().equals("ACTIVE")
                || !Set.of("DEVELOPER","ADMIN","SUPER_ADMIN","MANAGER").contains(roles.getFirst().path("role").asText()))
                throw new AccessDeniedException("An active reporting role is required in the live database");
            var params = List.of(time("since",now.minusDays(days)),time("now",now));
            var rows = pages(tx,OUTPUT,params,1);
            var history = pages(tx,HISTORY,params,100);
            return new Snapshot(rows.stream().map(r -> LanggraphOutput.builder().id(uuid(r,"id"))
                .checklistId(uuid(r,"checklist_id")).output(r.path("output").toString())
                .createdAt(date(r,"created_at")).updatedAt(date(r,"updated_at")).build()).toList(),
                rows.stream().map(r -> Checklist.builder().id(uuid(r,"checklist_id"))
                .mediaAssetId(uuid(r,"media_asset_id")).checklist(r.path("checklist").toString())
                .reviewedAt(date(r,"reviewed_at")).build()).toList(),
                history.stream().map(r -> ChecklistDecision.builder().id(uuid(r,"id"))
                .checklistId(uuid(r,"checklist_id")).checklistItemId(r.path("checklist_item_id").asText())
                .aiVerdict(r.path("ai_verdict").asText()).decision(r.path("decision").isNull()?null:r.path("decision").asText())
                .decidedBy(uuid(r,"decided_by")).decidedAt(date(r,"decided_at")).build()).toList());
        } finally {
            client.rollbackTransaction(RollbackTransactionRequest.builder().resourceArn(cluster).secretArn(secret).transactionId(tx).build());
        }
    }
}
