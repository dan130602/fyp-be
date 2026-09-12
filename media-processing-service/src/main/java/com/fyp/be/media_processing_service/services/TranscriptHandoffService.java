package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.MediaAsset;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Publishes a completed MediaAsset's transcript + checklist to a downstream service's own SQS
 * queue - a separate, independently-owned queue from transcribe-status-updates (that one carries
 * the raw Transcribe/Call Analytics completion event for this service's own poller to consume;
 * this one carries the payload the *next* service actually needs, once we're done with it).
 *
 * The transcript itself is passed by S3 pointer, not inlined - Call Analytics output can be
 * large, and SQS caps message bodies at 256KB.
 *
 * The checklist is a single fixed JSON document for the whole deployment (not per-upload) -
 * configured via app.checklist-json (CHECKLIST_JSON env var), parsed once at startup rather than
 * per-publish. A malformed value fails fast at boot instead of surfacing only once a job completes.
 *
 * Deliberately propagates failures rather than swallowing them - called from inside
 * MediaAssetService.applyTranscribeStatusUpdate's @Transactional method, so a publish failure
 * rolls back that same DB update. The incoming SQS message TranscribeStatusPollingService is
 * processing then never gets deleted, so it's redelivered and the whole update+publish retried
 * together - reusing that queue's existing redelivery/DLQ machinery instead of building a second
 * one for this handoff specifically.
 *
 * No-ops when app.aws.sqs.transcript-handoff-queue-url is unset, matching
 * TranscribeStatusPollingService's guard for the same reason (local dev without this infra
 * wired up shouldn't fail).
 */
@Service
public class TranscriptHandoffService {

    private final SqsClient sqsClient;
    private final ObjectMapper objectMapper;
    private final String queueUrl;
    private final JsonNode checklist;

    public TranscriptHandoffService(
            SqsClient sqsClient,
            ObjectMapper objectMapper,
            @Value("${app.aws.sqs.transcript-handoff-queue-url:}") String queueUrl,
            @Value("${app.checklist-json}") String checklistJson
    ) {
        this.sqsClient = sqsClient;
        this.objectMapper = objectMapper;
        this.queueUrl = queueUrl;
        this.checklist = objectMapper.readTree(checklistJson);
    }

    public void publish(MediaAsset asset) {
        if (queueUrl == null || queueUrl.isBlank()) {
            return;
        }

        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("mediaAssetId", asset.getId().toString());
        payload.put("transcriptUri", asset.getTranscriptUri());
        payload.set("checklist", checklist);

        sqsClient.sendMessage(SendMessageRequest.builder()
                .queueUrl(queueUrl)
                .messageBody(objectMapper.writeValueAsString(payload))
                .build());
    }
}
