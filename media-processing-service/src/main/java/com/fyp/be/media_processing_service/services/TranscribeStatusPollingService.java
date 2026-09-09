package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.DeleteMessageRequest;
import software.amazon.awssdk.services.sqs.model.Message;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageRequest;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageResponse;

/**
 * Polls the SQS queue an EventBridge rule delivers "Call Analytics Job State Change" events to
 * (see infra/modules/transcribe_status) and flips the matching MediaAsset's status. Uses SQS
 * long-polling (waitTimeSeconds), so @Scheduled's fixedDelay is a safety buffer between calls,
 * not the actual poll interval - each receiveMessage call already blocks up to 20s.
 *
 * No-ops when app.aws.sqs.transcribe-status-queue-url is unset, so local dev without this
 * infra wired up doesn't spam errors polling an empty queue URL. Guarded imperatively rather
 * than via @ConditionalOnProperty, since that annotation treats an empty-string property value
 * (the default this key falls back to when the env var is absent) as "present" - the bean would
 * still be created either way, so this is the reliable place to skip the no-op case.
 */
@Slf4j
@Service
public class TranscribeStatusPollingService {

    private final SqsClient sqsClient;
    private final MediaAssetService mediaAssetService;
    private final ObjectMapper objectMapper;
    private final String queueUrl;
    private final String transcriptOutputBucket;

    public TranscribeStatusPollingService(
            SqsClient sqsClient,
            MediaAssetService mediaAssetService,
            ObjectMapper objectMapper,
            @Value("${app.aws.sqs.transcribe-status-queue-url}") String queueUrl,
            @Value("${app.aws.s3.transcript-output-bucket}") String transcriptOutputBucket
    ) {
        this.sqsClient = sqsClient;
        this.mediaAssetService = mediaAssetService;
        this.objectMapper = objectMapper;
        this.queueUrl = queueUrl;
        this.transcriptOutputBucket = transcriptOutputBucket;
    }

    @Scheduled(fixedDelay = 1000)
    public void pollStatusQueue() {
        if (queueUrl == null || queueUrl.isBlank()) {
            return;
        }

        ReceiveMessageResponse response = sqsClient.receiveMessage(ReceiveMessageRequest.builder()
                .queueUrl(queueUrl)
                .maxNumberOfMessages(10)
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
                log.error("Failed to process transcribe status message {}", message.messageId(), ex);
                // Left in the queue on purpose - visibility timeout expires, SQS redelivers,
                // and the status queue's own redrive policy routes it to a DLQ after enough
                // failed attempts rather than looping forever.
            }
        }
    }

    private void processMessage(Message message) throws Exception {
        JsonNode detail = objectMapper.readTree(message.body()).path("detail");
        // Call Analytics events name these fields differently from standard Transcribe
        // ("TranscriptionJobName"/"TranscriptionJobStatus"); a mismatch here is silent,
        // since asText(null) just yields null and the event is dropped as malformed.
        String jobName = detail.path("JobName").asText(null);
        String status = detail.path("JobStatus").asText(null);
        if (jobName == null || status == null) {
            log.warn("Ignoring malformed Transcribe status event: {}", message.body());
            return;
        }

        MediaAssetStatus newStatus = "COMPLETED".equals(status) ? MediaAssetStatus.COMPLETED : MediaAssetStatus.FAILED;
        String transcriptUri = newStatus == MediaAssetStatus.COMPLETED
                ? "s3://%s/transcripts/%s.json".formatted(transcriptOutputBucket, jobName)
                : null;
        String failureReason = detail.path("FailureReason").asText(null);

        mediaAssetService.applyTranscribeStatusUpdate(jobName, newStatus, transcriptUri, failureReason);
    }
}
