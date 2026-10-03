package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.DeleteMessageRequest;
import software.amazon.awssdk.services.sqs.model.Message;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageRequest;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageResponse;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.util.UUID;

/**
 * Consumes the transcript-handoff queue TranscriptHandoffService publishes to: for each
 * {mediaAssetId, transcriptUri, checklistId} message, loads the Transcribe output JSON from S3
 * and the referenced Checklist row, runs fyp-ai's compliance graph on it via POST /runs/wait, and
 * stores the returned report in t_langgraph_output.
 *
 * Takes one message per receive - each /runs/wait call can take minutes, and the queue's
 * visibility timeout (see infra) is sized for one run, not a batch of them.
 *
 * Failures leave the message in the queue, same as TranscribeStatusPollingService - SQS
 * redelivers it after the visibility timeout, and the queue's redrive policy moves it to the
 * DLQ after enough failed attempts.
 *
 * No-ops when app.aws.sqs.transcript-handoff-queue-url is unset, for local dev without the infra.
 */
@Slf4j
@Service
public class ComplianceAnalysisService {

    private static final String ASSISTANT_ID = "compliance";

    private final SqsClient sqsClient;
    private final S3Client s3Client;
    private final RestClient aiRestClient;
    private final ChecklistRepository checklistRepository;
    private final ChecklistService checklistService;
    private final ObjectMapper objectMapper;
    private final String queueUrl;

    public ComplianceAnalysisService(
            SqsClient sqsClient,
            S3Client s3Client,
            @Qualifier("aiRestClient") RestClient aiRestClient,
            ChecklistRepository checklistRepository,
            ChecklistService checklistService,
            ObjectMapper objectMapper,
            @Value("${app.aws.sqs.transcript-handoff-queue-url:}") String queueUrl
    ) {
        this.sqsClient = sqsClient;
        this.s3Client = s3Client;
        this.aiRestClient = aiRestClient;
        this.checklistRepository = checklistRepository;
        this.checklistService = checklistService;
        this.objectMapper = objectMapper;
        this.queueUrl = queueUrl;
    }

    @PostConstruct
    void warnIfQueueUnset() {
        if (queueUrl == null || queueUrl.isBlank()) {
            log.warn("TRANSCRIPT_HANDOFF_QUEUE_URL is not set; completed transcripts will not be analysed and no compliance report will be stored");
        }
    }

    @Scheduled(fixedDelay = 1000)
    public void pollHandoffQueue() {
        if (queueUrl == null || queueUrl.isBlank()) {
            return;
        }

        ReceiveMessageResponse response = sqsClient.receiveMessage(ReceiveMessageRequest.builder()
                .queueUrl(queueUrl)
                .maxNumberOfMessages(1)
                .waitTimeSeconds(20)
                .build());

        for (Message message : response.messages()) {
            try {
                processMessage(message);
                sqsClient.deleteMessage(DeleteMessageRequest.builder()
                        .queueUrl(queueUrl)
                        .receiptHandle(message.receiptHandle())
                        .build());
            } catch (Exception ex) {
                log.error("Failed to run compliance analysis for handoff message {}", message.messageId(), ex);
            }
        }
    }

    private void processMessage(Message message) {
        JsonNode body = objectMapper.readTree(message.body());
        // Kept for the log line only - no DB write in this class needs it anymore, since the
        // AI response is now stored against the checklist, not the media asset.
        UUID mediaAssetId = UUID.fromString(body.path("mediaAssetId").asText());
        UUID checklistId = UUID.fromString(body.path("checklistId").asText());
        String transcriptUri = body.path("transcriptUri").asText();

        Checklist checklist = checklistRepository.findById(checklistId)
                .orElseThrow(() -> new IllegalStateException("Checklist " + checklistId + " not found"));

        JsonNode results = readTranscript(transcriptUri).path("results");
        if (results.isMissingNode()) {
            throw new IllegalStateException("Transcript " + transcriptUri + " has no 'results' object");
        }

        ObjectNode input = objectMapper.createObjectNode();
        input.set("results", results);
        // The deployment-wide default checklist is "{}" - only forward a real list of items,
        // otherwise let fyp-ai fall back to its built-in checklist.
        JsonNode checklistItems = objectMapper.readTree(checklist.getChecklist());
        if (checklistItems.isArray() && !checklistItems.isEmpty()) {
            input.set("checklist", checklistItems);
        }

        ObjectNode request = objectMapper.createObjectNode();
        request.put("assistant_id", ASSISTANT_ID);
        request.set("input", input);

        log.info("Running compliance analysis for media asset {} (checklist {})", mediaAssetId, checklistId);
        String output = aiRestClient.post()
                .uri("/runs/wait")
                .contentType(MediaType.APPLICATION_JSON)
                .body(objectMapper.writeValueAsString(request))
                .retrieve()
                .body(String.class);

        checklistService.saveLanggraphOutput(checklistId, output);
    }

    /** transcriptUri is s3://bucket/key, as built by TranscribeStatusPollingService. */
    private JsonNode readTranscript(String transcriptUri) {
        URI uri = URI.create(transcriptUri);
        String json = s3Client.getObjectAsBytes(GetObjectRequest.builder()
                        .bucket(uri.getHost())
                        .key(uri.getPath().substring(1))
                        .build())
                .asUtf8String();
        return objectMapper.readTree(json);
    }
}
