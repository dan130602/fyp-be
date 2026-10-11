package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;
import software.amazon.awssdk.core.ResponseBytes;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.DeleteMessageRequest;
import software.amazon.awssdk.services.sqs.model.Message;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageRequest;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageResponse;
import tools.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.http.HttpMethod.POST;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.jsonPath;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

class ComplianceAnalysisServiceTest {

    private static final String QUEUE = "https://sqs/handoff";
    private static final String TRANSCRIPT = "{\"results\":{\"transcripts\":[{\"transcript\":\"hello\"}]}}";

    private final SqsClient sqs = mock(SqsClient.class);
    private final S3Client s3 = mock(S3Client.class);
    private final ChecklistRepository checklistRepository = mock(ChecklistRepository.class);
    private final ChecklistService checklistService = mock(ChecklistService.class);

    private MockRestServiceServer server;
    private ComplianceAnalysisService service;

    private final UUID assetId = UUID.randomUUID();
    private final UUID checklistId = UUID.randomUUID();

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://ai");
        server = MockRestServiceServer.bindTo(builder).build();
        service = new ComplianceAnalysisService(
                sqs, s3, builder.build(), checklistRepository, checklistService, new ObjectMapper(), QUEUE);
    }

    private Message handoffMessage() {
        String body = "{\"mediaAssetId\":\"%s\",\"checklistId\":\"%s\",\"transcriptUri\":\"s3://tbucket/transcripts/job.json\"}"
                .formatted(assetId, checklistId);
        return Message.builder().messageId("m1").receiptHandle("rh1").body(body).build();
    }

    private void receive(Message... messages) {
        when(sqs.receiveMessage(any(ReceiveMessageRequest.class)))
                .thenReturn(ReceiveMessageResponse.builder().messages(messages).build());
    }

    private void stubChecklist(String checklistJson) {
        when(checklistRepository.findById(checklistId))
                .thenReturn(Optional.of(Checklist.builder().id(checklistId).checklist(checklistJson).build()));
    }

    private void stubTranscript(String json) {
        when(s3.getObjectAsBytes(any(GetObjectRequest.class))).thenReturn(
                ResponseBytes.fromByteArray(GetObjectResponse.builder().build(), json.getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void poll_blankQueueUrl_doesNothing() {
        ComplianceAnalysisService blank = new ComplianceAnalysisService(
                sqs, s3, RestClient.create(), checklistRepository, checklistService, new ObjectMapper(), "");

        blank.pollHandoffQueue();

        verifyNoInteractions(sqs, s3, checklistRepository, checklistService);
    }

    @Test
    void poll_withChecklistItems_postsResultsAndChecklistThenStoresOutputAndDeletes() {
        receive(handoffMessage());
        stubChecklist("[{\"id\":\"q1\",\"question\":\"Greeting?\"}]");
        stubTranscript(TRANSCRIPT);
        server.expect(requestTo("http://ai/runs/wait"))
                .andExpect(method(POST))
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.assistant_id").value("compliance"))
                .andExpect(jsonPath("$.input.results.transcripts[0].transcript").value("hello"))
                .andExpect(jsonPath("$.input.checklist[0].id").value("q1"))
                .andRespond(withSuccess("{\"report\":\"ok\"}", MediaType.APPLICATION_JSON));

        service.pollHandoffQueue();

        server.verify();
        verify(checklistService).saveLanggraphOutput(checklistId, "{\"report\":\"ok\"}");
        verify(sqs).deleteMessage(DeleteMessageRequest.builder().queueUrl(QUEUE).receiptHandle("rh1").build());
    }

    @Test
    void poll_readsTranscriptFromBucketAndKeyInUri() {
        receive(handoffMessage());
        stubChecklist("[]");
        stubTranscript(TRANSCRIPT);
        server.expect(requestTo("http://ai/runs/wait")).andRespond(withSuccess("{}", MediaType.APPLICATION_JSON));

        service.pollHandoffQueue();

        verify(s3).getObjectAsBytes(GetObjectRequest.builder().bucket("tbucket").key("transcripts/job.json").build());
    }

    @Test
    void poll_defaultEmptyObjectChecklist_omitsChecklistFromRequest() {
        receive(handoffMessage());
        stubChecklist("{}");
        stubTranscript(TRANSCRIPT);
        server.expect(requestTo("http://ai/runs/wait"))
                .andExpect(jsonPath("$.input.checklist").doesNotExist())
                .andExpect(jsonPath("$.input.results").exists())
                .andRespond(withSuccess("{}", MediaType.APPLICATION_JSON));

        service.pollHandoffQueue();

        server.verify();
        verify(checklistService).saveLanggraphOutput(checklistId, "{}");
    }

    @Test
    void poll_checklistMissing_leavesMessageInQueue() {
        receive(handoffMessage());
        when(checklistRepository.findById(checklistId)).thenReturn(Optional.empty());

        service.pollHandoffQueue();

        verify(sqs, never()).deleteMessage(any(DeleteMessageRequest.class));
        verify(checklistService, never()).saveLanggraphOutput(any(), any());
    }

    @Test
    void poll_transcriptWithoutResults_leavesMessageInQueue() {
        receive(handoffMessage());
        stubChecklist("[]");
        stubTranscript("{\"other\":1}");

        service.pollHandoffQueue();

        verify(sqs, never()).deleteMessage(any(DeleteMessageRequest.class));
        verify(checklistService, never()).saveLanggraphOutput(any(), any());
    }

    @Test
    void poll_aiServiceError_leavesMessageInQueueAndStoresNothing() {
        receive(handoffMessage());
        stubChecklist("[]");
        stubTranscript(TRANSCRIPT);
        server.expect(requestTo("http://ai/runs/wait")).andRespond(withServerError());

        service.pollHandoffQueue();

        verify(sqs, never()).deleteMessage(any(DeleteMessageRequest.class));
        verify(checklistService, never()).saveLanggraphOutput(any(), any());
    }

    @Test
    void poll_malformedMessageBody_leavesMessageInQueue() {
        receive(Message.builder().messageId("m").receiptHandle("rh").body("{\"checklistId\":\"not-a-uuid\"}").build());

        service.pollHandoffQueue();

        verify(sqs, never()).deleteMessage(any(DeleteMessageRequest.class));
    }
}
