package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ChecklistItemRequest;
import com.fyp.be.media_processing_service.dtos.ChecklistItemResponse;
import com.fyp.be.media_processing_service.dtos.ChecklistReviewResponse;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideRequest;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideResponse;
import com.fyp.be.media_processing_service.exceptions.AnalysisNotReadyException;
import com.fyp.be.media_processing_service.exceptions.InvalidReviewException;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import tools.jackson.databind.ObjectMapper;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ChecklistServiceTest {

    private ChecklistService serviceWith(String json) {
        return new ChecklistService(
                mock(ChecklistRepository.class),
                mock(LanggraphOutputRepository.class),
                new ObjectMapper(),
                json);
    }

    @Test
    void getDefaultChecklist_validArray_returnsItems() {
        List<ChecklistItemResponse> items = serviceWith(
                "[{\"id\":\"q1\",\"question\":\"Greeting?\"},{\"id\":\"q2\",\"question\":\"Disclosure?\"}]")
                .getDefaultChecklist();

        assertThat(items).containsExactly(
                new ChecklistItemResponse("q1", "Greeting?"),
                new ChecklistItemResponse("q2", "Disclosure?"));
    }

    @Test
    void getDefaultChecklist_emptyObject_returnsEmpty() {
        assertThat(serviceWith("{}").getDefaultChecklist()).isEmpty();
    }

    @Test
    void getDefaultChecklist_blank_returnsEmpty() {
        assertThat(serviceWith("").getDefaultChecklist()).isEmpty();
    }

    private final ChecklistRepository checklistRepository = mock(ChecklistRepository.class);
    private final LanggraphOutputRepository outputRepository = mock(LanggraphOutputRepository.class);

    private ChecklistService persistingService() {
        when(checklistRepository.save(any(Checklist.class))).thenAnswer(inv -> inv.getArgument(0));
        return new ChecklistService(checklistRepository, outputRepository, new ObjectMapper(), "[{\"id\":\"d\",\"question\":\"Default?\"}]");
    }

    @Test
    void createForUpload_withItems_storesSubmittedItemsAsSubmitted() {
        UUID assetId = UUID.randomUUID();
        UUID userId = UUID.randomUUID();

        Checklist saved = persistingService().createForUpload(assetId, userId, List.of(new ChecklistItemRequest("q1", "Greeting?")));

        assertThat(saved.getMediaAssetId()).isEqualTo(assetId);
        assertThat(saved.getSubmittedBy()).isEqualTo(userId);
        assertThat(saved.getStatus()).isEqualTo(ChecklistStatus.SUBMITTED);
        assertThat(saved.getChecklist()).contains("\"q1\"").contains("Greeting?");
    }

    @Test
    void createForUpload_nullOrEmptyItems_fallsBackToDefault() {
        ChecklistService service = persistingService();

        assertThat(service.createForUpload(UUID.randomUUID(), UUID.randomUUID(), null).getChecklist())
                .isEqualTo("[{\"id\":\"d\",\"question\":\"Default?\"}]");
        assertThat(service.createForUpload(UUID.randomUUID(), UUID.randomUUID(), List.of()).getChecklist())
                .isEqualTo("[{\"id\":\"d\",\"question\":\"Default?\"}]");
    }

    @Test
    void getOrCreateForAsset_existing_returnsItWithoutSaving() {
        UUID assetId = UUID.randomUUID();
        Checklist existing = Checklist.builder().id(UUID.randomUUID()).mediaAssetId(assetId).build();
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.of(existing));

        assertThat(persistingService().getOrCreateForAsset(assetId)).isSameAs(existing);
        verify(checklistRepository, never()).save(any());
    }

    @Test
    void getOrCreateForAsset_missing_createsPendingReviewFromDefault() {
        UUID assetId = UUID.randomUUID();
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.empty());

        Checklist created = persistingService().getOrCreateForAsset(assetId);

        assertThat(created.getMediaAssetId()).isEqualTo(assetId);
        assertThat(created.getSubmittedBy()).isNull();
        assertThat(created.getStatus()).isEqualTo(ChecklistStatus.PENDING_REVIEW);
        assertThat(created.getChecklist()).isEqualTo("[{\"id\":\"d\",\"question\":\"Default?\"}]");
    }

    @Test
    void saveLanggraphOutput_noExistingRow_insertsNew() {
        UUID checklistId = UUID.randomUUID();
        when(outputRepository.findByChecklistId(checklistId)).thenReturn(Optional.empty());

        persistingService().saveLanggraphOutput(checklistId, "{\"a\":1}");

        ArgumentCaptor<LanggraphOutput> captor = ArgumentCaptor.forClass(LanggraphOutput.class);
        verify(outputRepository).save(captor.capture());
        assertThat(captor.getValue().getChecklistId()).isEqualTo(checklistId);
        assertThat(captor.getValue().getOutput()).isEqualTo("{\"a\":1}");
        assertThat(captor.getValue().getId()).isNotNull();
    }

    @Test
    void saveLanggraphOutput_existingRow_updatesInPlace() {
        UUID checklistId = UUID.randomUUID();
        UUID rowId = UUID.randomUUID();
        LanggraphOutput existing = LanggraphOutput.builder().id(rowId).checklistId(checklistId).output("old").build();
        when(outputRepository.findByChecklistId(checklistId)).thenReturn(Optional.of(existing));

        persistingService().saveLanggraphOutput(checklistId, "new");

        verify(outputRepository).save(existing);
        assertThat(existing.getId()).isEqualTo(rowId);
        assertThat(existing.getOutput()).isEqualTo("new");
        assertThat(existing.getUpdatedAt()).isNotNull();
    }

    private static final String AI_OUTPUT = "{\"final_results\":["
            + "{\"item_id\":\"K01\",\"status\":\"PASS\"},"
            + "{\"item_id\":\"K02\",\"status\":\"FAIL\"}]}";

    private final UUID checklistId = UUID.randomUUID();
    private final UUID reviewerId = UUID.randomUUID();

    private Checklist givenChecklistWithOutput(String output) {
        Checklist checklist = Checklist.builder().id(checklistId).status(ChecklistStatus.PENDING_REVIEW).build();
        when(checklistRepository.findById(checklistId)).thenReturn(Optional.of(checklist));
        when(outputRepository.findByChecklistId(checklistId)).thenReturn(
                output == null ? Optional.empty() : Optional.of(LanggraphOutput.builder().checklistId(checklistId).output(output).build()));
        return checklist;
    }

    private static ReviewOverrideRequest override(String itemId, String decision) {
        return new ReviewOverrideRequest(itemId, decision, "because");
    }

    @Test
    void submitReview_snapshotsAiVerdictAndMarksReviewed() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId, List.of(override("K01", "FAIL")));

        assertThat(response.reviewedBy()).isEqualTo(reviewerId);
        assertThat(response.reviewedAt()).isNotNull();
        assertThat(response.overrides()).containsExactly(new ReviewOverrideResponse("K01", "PASS", "FAIL", "because"));
        assertThat(checklist.getStatus()).isEqualTo(ChecklistStatus.REVIEWED);
        assertThat(checklist.getReviewedBy()).isEqualTo(reviewerId);
        assertThat(checklist.getReviewedAt()).isNotNull();
        assertThat(checklist.getReview()).contains("\"aiVerdict\":\"PASS\"").contains("\"decision\":\"FAIL\"");
        verify(checklistRepository).save(checklist);
    }

    @Test
    void submitReview_emptyOverrides_storesEmptyArray() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId, List.of());

        assertThat(response.overrides()).isEmpty();
        assertThat(checklist.getReview()).isEqualTo("[]");
        assertThat(checklist.getStatus()).isEqualTo(ChecklistStatus.REVIEWED);
    }

    @Test
    void submitReview_unknownItem_isInvalid() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of(override("K99", "FAIL"))))
                .isInstanceOf(InvalidReviewException.class);
        assertThat(checklist.getReview()).isNull();
        verify(checklistRepository, never()).save(any());
    }

    @Test
    void submitReview_duplicateItem_isInvalid() {
        givenChecklistWithOutput(AI_OUTPUT);

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId,
                List.of(override("K01", "FAIL"), override("K01", "PARTIAL"))))
                .isInstanceOf(InvalidReviewException.class);
    }

    @Test
    void submitReview_unknownDecision_isInvalid() {
        givenChecklistWithOutput(AI_OUTPUT);

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of(override("K01", "MAYBE"))))
                .isInstanceOf(InvalidReviewException.class);
    }

    @Test
    void submitReview_decisionEqualToAiVerdict_isInvalid() {
        givenChecklistWithOutput(AI_OUTPUT);

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of(override("K01", "PASS"))))
                .isInstanceOf(InvalidReviewException.class);
    }

    @Test
    void submitReview_noOutput_isNotReady() {
        givenChecklistWithOutput(null);

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of()))
                .isInstanceOf(AnalysisNotReadyException.class);
    }

    @Test
    void submitReview_unreadableOutput_isNotReady() {
        givenChecklistWithOutput("not json {");

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of()))
                .isInstanceOf(AnalysisNotReadyException.class);
    }

    @Test
    void submitReview_errorOutput_isNotReady() {
        givenChecklistWithOutput("{\"__error__\":{\"message\":\"boom\"}}");

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of()))
                .isInstanceOf(AnalysisNotReadyException.class);
        verify(checklistRepository, never()).save(any());
    }

    @Test
    void submitReview_resubmit_overwritesPreviousReview() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);
        ChecklistService service = persistingService();

        service.submitReview(checklistId, reviewerId, List.of(override("K01", "FAIL")));
        ChecklistReviewResponse second = service.submitReview(checklistId, reviewerId, List.of(override("K02", "PASS")));

        assertThat(second.overrides()).containsExactly(new ReviewOverrideResponse("K02", "FAIL", "PASS", "because"));
        assertThat(checklist.getReview()).contains("K02").doesNotContain("K01");
    }

    @Test
    void toReviewResponse_neverReviewed_isNull() {
        assertThat(persistingService().toReviewResponse(Checklist.builder().build())).isNull();
    }
}
