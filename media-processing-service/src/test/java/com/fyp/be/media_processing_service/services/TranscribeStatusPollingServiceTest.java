package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.DeleteMessageRequest;
import software.amazon.awssdk.services.sqs.model.Message;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageRequest;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageResponse;
import tools.jackson.databind.ObjectMapper;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class TranscribeStatusPollingServiceTest {

    private static final String QUEUE = "https://sqs/status";

    private final SqsClient sqs = mock(SqsClient.class);
    private final MediaAssetService mediaAssetService = mock(MediaAssetService.class);

    private TranscribeStatusPollingService service(String queueUrl) {
        return new TranscribeStatusPollingService(sqs, mediaAssetService, new ObjectMapper(), queueUrl, "out-bucket");
    }

    private void receive(Message... messages) {
        when(sqs.receiveMessage(any(ReceiveMessageRequest.class)))
                .thenReturn(ReceiveMessageResponse.builder().messages(messages).build());
    }

    private static Message msg(String receiptHandle, String body) {
        return Message.builder().messageId("m-" + receiptHandle).receiptHandle(receiptHandle).body(body).build();
    }

    private static String event(String job, String status) {
        return "{\"detail\":{\"TranscriptionJobName\":\"" + job + "\",\"TranscriptionJobStatus\":\"" + status + "\"}}";
    }

    @Test
    void poll_blankQueueUrl_doesNothing() {
        service("").pollStatusQueue();
        service(null).pollStatusQueue();

        verifyNoInteractions(sqs, mediaAssetService);
    }

    @Test
    void poll_completedEvent_marksCompletedWithTranscriptUriAndDeletesMessage() {
        receive(msg("rh1", event("job-1", "COMPLETED")));

        service(QUEUE).pollStatusQueue();

        verify(mediaAssetService).applyTranscribeStatusUpdate(
                "job-1", MediaAssetStatus.COMPLETED, "s3://out-bucket/transcripts/job-1.json", null);
        verify(sqs).deleteMessage(DeleteMessageRequest.builder().queueUrl(QUEUE).receiptHandle("rh1").build());
    }

    @Test
    void poll_failedEvent_marksFailedWithReasonAndNoTranscriptUri() {
        receive(msg("rh1", "{\"detail\":{\"TranscriptionJobName\":\"job-2\",\"TranscriptionJobStatus\":\"FAILED\","
                + "\"FailureReason\":\"bad audio\"}}"));

        service(QUEUE).pollStatusQueue();

        verify(mediaAssetService).applyTranscribeStatusUpdate("job-2", MediaAssetStatus.FAILED, null, "bad audio");
        verify(sqs).deleteMessage(any(DeleteMessageRequest.class));
    }

    @Test
    void poll_malformedEvent_isIgnoredButDeleted() {
        receive(msg("rh1", "{\"detail\":{\"TranscriptionJobStatus\":\"COMPLETED\"}}"));

        service(QUEUE).pollStatusQueue();

        verifyNoInteractions(mediaAssetService);
        verify(sqs).deleteMessage(any(DeleteMessageRequest.class));
    }

    @Test
    void poll_processingFailure_leavesMessageInQueue() {
        receive(msg("rh1", event("job-3", "COMPLETED")));
        doThrow(new IllegalStateException("db down")).when(mediaAssetService)
                .applyTranscribeStatusUpdate(eq("job-3"), any(), any(), any());

        service(QUEUE).pollStatusQueue();

        verify(sqs, never()).deleteMessage(any(DeleteMessageRequest.class));
    }

    @Test
    void poll_unparseableBody_leavesMessageInQueue() {
        receive(msg("rh1", "not json"));

        service(QUEUE).pollStatusQueue();

        verify(sqs, never()).deleteMessage(any(DeleteMessageRequest.class));
    }

    @Test
    void poll_oneBadMessageDoesNotBlockTheOthers() {
        receive(msg("bad", "not json"), msg("good", event("ok", "COMPLETED")));

        service(QUEUE).pollStatusQueue();

        verify(mediaAssetService).applyTranscribeStatusUpdate(
                "ok", MediaAssetStatus.COMPLETED, "s3://out-bucket/transcripts/ok.json", null);
        verify(sqs).deleteMessage(DeleteMessageRequest.builder().queueUrl(QUEUE).receiptHandle("good").build());
        verify(sqs, never()).deleteMessage(DeleteMessageRequest.builder().queueUrl(QUEUE).receiptHandle("bad").build());
    }
}
