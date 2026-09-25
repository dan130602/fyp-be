package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.MediaAsset;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Publishes a completed MediaAsset's transcript + checklist reference to a downstream service's
 * own SQS queue - a separate, independently-owned queue from transcribe-status-updates (that one
 * carries the raw Transcribe/Call Analytics completion event for this service's own poller to
 * consume; this one carries the payload the *next* service actually needs, once we're done
 * with it).
 *
 * The transcript itself is passed by S3 pointer, not inlined - Call Analytics output can be
 * large, and SQS caps message bodies at 256KB.
 *
 * The checklist is passed by t_checklist row id, not inlined - ChecklistService.getOrCreateForAsset
 * lazily creates that row (seeded from app.checklist-json) on first handoff for this asset, so a
 * consumer re-reading it later always sees the latest reviewer edits rather than a stale snapshot
 * taken at enqueue time.
 *
 * Deliberately propagates failures rather than swallowing them - called from inside
 * MediaAssetService.applyTranscribeStatusUpdate's @Transactional method, so a publish failure
 * rolls back that same DB update (including the checklist row creation). The incoming SQS message
 * TranscribeStatusPollingService is processing then never gets deleted, so it's redelivered and
 * the whole update+publish retried together - reusing that queue's existing redelivery/DLQ
 * machinery instead of building a second one for this handoff specifically.
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
    private final ChecklistService checklistService;

    public TranscriptHandoffService(
            SqsClient sqsClient,
            ObjectMapper objectMapper,
            @Value("${app.aws.sqs.transcript-handoff-queue-url:}") String queueUrl,
            ChecklistService checklistService
    ) {
        this.sqsClient = sqsClient;
        this.objectMapper = objectMapper;
        this.queueUrl = queueUrl;
        this.checklistService = checklistService;
    }

    public void publish(MediaAsset asset) {
        if (queueUrl == null || queueUrl.isBlank()) {
            return;
        }

        Checklist checklist = checklistService.getOrCreateForAsset(asset.getId());

        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("mediaAssetId", asset.getId().toString());
        payload.put("transcriptUri", asset.getTranscriptUri());
        payload.put("checklistId", checklist.getId().toString());

        sqsClient.sendMessage(SendMessageRequest.builder()
                .queueUrl(queueUrl)
                .messageBody(objectMapper.writeValueAsString(payload))
                .build());
    }
}
