package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.MediaAsset;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.SendMessageRequest;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class TranscriptHandoffServiceTest {

    private final SqsClient sqs = mock(SqsClient.class);
    private final ChecklistService checklistService = mock(ChecklistService.class);
    private final ObjectMapper objectMapper = new ObjectMapper();

    private final MediaAsset asset = MediaAsset.builder()
            .id(UUID.randomUUID()).transcriptUri("s3://bucket/transcripts/job.json").build();

    private TranscriptHandoffService serviceWith(String queueUrl) {
        return new TranscriptHandoffService(sqs, objectMapper, queueUrl, checklistService);
    }

    @Test
    void publish_sendsPayloadWithAssetTranscriptAndChecklistIds() {
        Checklist checklist = Checklist.builder().id(UUID.randomUUID()).build();
        when(checklistService.getOrCreateForAsset(asset.getId())).thenReturn(checklist);

        serviceWith("https://sqs/queue").publish(asset);

        ArgumentCaptor<SendMessageRequest> captor = ArgumentCaptor.forClass(SendMessageRequest.class);
        verify(sqs).sendMessage(captor.capture());
        assertThat(captor.getValue().queueUrl()).isEqualTo("https://sqs/queue");
        JsonNode body = objectMapper.readTree(captor.getValue().messageBody());
        assertThat(body.get("mediaAssetId").asText()).isEqualTo(asset.getId().toString());
        assertThat(body.get("transcriptUri").asText()).isEqualTo("s3://bucket/transcripts/job.json");
        assertThat(body.get("checklistId").asText()).isEqualTo(checklist.getId().toString());
    }

    @Test
    void publish_blankOrNullQueueUrl_isNoOp() {
        serviceWith("").publish(asset);
        serviceWith("  ").publish(asset);
        serviceWith(null).publish(asset);

        verifyNoInteractions(sqs, checklistService);
    }

    @Test
    void publish_sqsFailure_propagatesSoCallerTransactionRollsBack() {
        when(checklistService.getOrCreateForAsset(any()))
                .thenReturn(Checklist.builder().id(UUID.randomUUID()).build());
        when(sqs.sendMessage(any(SendMessageRequest.class))).thenThrow(new IllegalStateException("sqs down"));

        assertThatThrownBy(() -> serviceWith("https://sqs/queue").publish(asset))
                .isInstanceOf(IllegalStateException.class);
    }
}
