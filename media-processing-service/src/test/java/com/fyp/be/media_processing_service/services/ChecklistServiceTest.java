package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ApiTime;
import com.fyp.be.media_processing_service.dtos.ChecklistDecisionResponse;
import com.fyp.be.media_processing_service.dtos.ChecklistItemRequest;
import com.fyp.be.media_processing_service.dtos.ChecklistItemResponse;
import com.fyp.be.media_processing_service.dtos.ChecklistReviewResponse;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideRequest;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideResponse;
import com.fyp.be.media_processing_service.exceptions.AnalysisNotReadyException;
import com.fyp.be.media_processing_service.exceptions.InvalidReviewException;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.repositories.ChecklistDecisionRepository;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
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
                mock(ChecklistDecisionRepository.class),
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
    private final ChecklistDecisionRepository decisionRepository = mock(ChecklistDecisionRepository.class);

    private ChecklistService persistingService() {
        when(checklistRepository.save(any(Checklist.class))).thenAnswer(inv -> inv.getArgument(0));
        return new ChecklistService(checklistRepository, outputRepository, decisionRepository, new ObjectMapper(), "[{\"id\":\"d\",\"question\":\"Default?\"}]");
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

    /** Rows saved by submitReview, which the repository mock then returns as the log. */
    private List<ChecklistDecision> savedRows() {
        ArgumentCaptor<List<ChecklistDecision>> captor = ArgumentCaptor.forClass(List.class);
        verify(decisionRepository).saveAll(captor.capture());
        return captor.getValue();
    }

    private static ChecklistDecision row(String itemId, String aiVerdict, String decision, String comment, LocalDateTime at) {
        return ChecklistDecision.builder().id(UUID.randomUUID()).checklistItemId(itemId).aiVerdict(aiVerdict)
                .decision(decision).comment(comment).decidedBy(UUID.randomUUID()).decidedAt(at).build();
    }

    private void givenLog(ChecklistDecision... rows) {
        when(decisionRepository.findByChecklistIdOrderByDecidedAtAsc(checklistId)).thenReturn(List.of(rows));
    }

    private static final LocalDateTime T1 = LocalDateTime.of(2026, 10, 24, 9, 0);
    private static final LocalDateTime T2 = LocalDateTime.of(2026, 10, 24, 10, 0);

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
        verify(checklistRepository).save(checklist);
    }

    @Test
    void submitReview_firstSubmit_logsOneRowPerOverride() {
        givenChecklistWithOutput(AI_OUTPUT);

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId,
                List.of(override("K01", "FAIL"), override("K02", "PASS")));

        List<ChecklistDecision> rows = savedRows();
        assertThat(rows).extracting(ChecklistDecision::getChecklistItemId).containsExactly("K01", "K02");
        assertThat(rows).extracting(ChecklistDecision::getAiVerdict).containsExactly("PASS", "FAIL");
        assertThat(rows).extracting(ChecklistDecision::getDecision).containsExactly("FAIL", "PASS");
        assertThat(rows).extracting(ChecklistDecision::getDecidedBy).containsOnly(reviewerId);
        assertThat(rows).extracting(ChecklistDecision::getChecklistId).containsOnly(checklistId);
        assertThat(rows).extracting(ChecklistDecision::getDecidedAt).containsOnly(rows.get(0).getDecidedAt());
        assertThat(response.history()).hasSize(2);
    }

    @Test
    void submitReview_emptyFirstSubmit_logsNothingButMarksReviewed() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId, List.of());

        assertThat(savedRows()).isEmpty();
        assertThat(response.overrides()).isEmpty();
        assertThat(response.history()).isEmpty();
        assertThat(checklist.getStatus()).isEqualTo(ChecklistStatus.REVIEWED);
        assertThat(checklist.getReviewedAt()).isNotNull();
    }

    @Test
    void submitReview_identicalResubmit_logsNothing() {
        givenChecklistWithOutput(AI_OUTPUT);
        givenLog(row("K01", "PASS", "FAIL", "because", T1));

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId, List.of(override("K01", "FAIL")));

        assertThat(savedRows()).isEmpty();
        assertThat(response.overrides()).containsExactly(new ReviewOverrideResponse("K01", "PASS", "FAIL", "because"));
        assertThat(response.history()).hasSize(1);
    }

    @Test
    void submitReview_decisionChange_logsRow() {
        givenChecklistWithOutput(AI_OUTPUT);
        givenLog(row("K01", "PASS", "FAIL", "because", T1));

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId,
                List.of(new ReviewOverrideRequest("K01", "PARTIAL", "because")));

        assertThat(savedRows()).singleElement().satisfies(r -> {
            assertThat(r.getChecklistItemId()).isEqualTo("K01");
            assertThat(r.getDecision()).isEqualTo("PARTIAL");
        });
        assertThat(response.overrides()).containsExactly(new ReviewOverrideResponse("K01", "PASS", "PARTIAL", "because"));
        assertThat(response.history()).hasSize(2);
    }

    @Test
    void submitReview_commentOnlyChange_logsRow() {
        givenChecklistWithOutput(AI_OUTPUT);
        givenLog(row("K01", "PASS", "FAIL", "because", T1));

        persistingService().submitReview(checklistId, reviewerId, List.of(new ReviewOverrideRequest("K01", "FAIL", "updated reason")));

        assertThat(savedRows()).singleElement().satisfies(r -> {
            assertThat(r.getDecision()).isEqualTo("FAIL");
            assertThat(r.getComment()).isEqualTo("updated reason");
        });
    }

    @Test
    void submitReview_removedOverride_logsRevertRow() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);
        givenLog(row("K01", "PASS", "FAIL", "because", T1), row("K02", "FAIL", "PASS", "ok", T1));

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId, List.of(override("K01", "FAIL")));

        assertThat(savedRows()).singleElement().satisfies(r -> {
            assertThat(r.getChecklistItemId()).isEqualTo("K02");
            assertThat(r.getAiVerdict()).isEqualTo("FAIL");
            assertThat(r.getDecision()).isNull();
            assertThat(r.getComment()).isNull();
            assertThat(r.getDecidedBy()).isEqualTo(reviewerId);
        });
        assertThat(response.overrides()).containsExactly(new ReviewOverrideResponse("K01", "PASS", "FAIL", "because"));
        assertThat(checklist.getStatus()).isEqualTo(ChecklistStatus.REVIEWED);
    }

    @Test
    void submitReview_previouslyRevertedItemAbsent_logsNothing() {
        givenChecklistWithOutput(AI_OUTPUT);
        givenLog(row("K01", "PASS", "FAIL", "because", T1), row("K01", "PASS", null, null, T2));

        persistingService().submitReview(checklistId, reviewerId, List.of());

        assertThat(savedRows()).isEmpty();
    }

    @Test
    void submitReview_returnsHistoryOldestFirst() {
        givenChecklistWithOutput(AI_OUTPUT);
        givenLog(row("K01", "PASS", "FAIL", "first", T1), row("K01", "PASS", "PARTIAL", "second", T2));

        ChecklistReviewResponse response = persistingService().submitReview(checklistId, reviewerId, List.of(override("K02", "PASS")));

        assertThat(response.history()).extracting(ChecklistDecisionResponse::comment)
                .containsExactly("first", "second", "because", null);
        assertThat(response.history().get(3).decision()).isNull();
        assertThat(response.history().get(0).decidedAt()).isEqualTo(ApiTime.of(T1));
    }

    @Test
    void currentOverrides_keepsNewestRowPerItemAndDropsReverted() {
        ChecklistDecision k01First = row("K01", "PASS", "FAIL", "a", T1);
        ChecklistDecision k01Revert = row("K01", "PASS", null, null, T2);
        ChecklistDecision k02First = row("K02", "FAIL", "PASS", "b", T1);
        ChecklistDecision k02Second = row("K02", "FAIL", "PARTIAL", "c", T2);

        assertThat(ChecklistService.currentOverrides(List.of(k01First, k02First, k01Revert, k02Second)))
                .containsExactly(k02Second);
        assertThat(ChecklistService.currentOverrides(List.of())).isEmpty();
    }

    @Test
    void submitReview_unknownItem_isInvalid() {
        Checklist checklist = givenChecklistWithOutput(AI_OUTPUT);

        assertThatThrownBy(() -> persistingService().submitReview(checklistId, reviewerId, List.of(override("K99", "FAIL"))))
                .isInstanceOf(InvalidReviewException.class);
        assertThat(checklist.getReviewedAt()).isNull();
        verify(decisionRepository, never()).saveAll(any());
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
    void toReviewResponse_neverReviewed_isNull() {
        assertThat(persistingService().toReviewResponse(Checklist.builder().build())).isNull();
    }

    @Test
    void toReviewResponse_approvedAsIs_hasEmptyOverridesButIsReviewed() {
        Checklist checklist = Checklist.builder().id(checklistId).reviewedBy(reviewerId).reviewedAt(T1).build();
        givenLog();

        ChecklistReviewResponse response = persistingService().toReviewResponse(checklist);

        assertThat(response.reviewedBy()).isEqualTo(reviewerId);
        assertThat(response.reviewedAt()).isEqualTo(ApiTime.of(T1));
        assertThat(response.overrides()).isEmpty();
        assertThat(response.history()).isEmpty();
    }

    @Test
    void toReviewResponse_derivesOverridesFromLogAndKeepsFullHistory() {
        Checklist checklist = Checklist.builder().id(checklistId).reviewedBy(reviewerId).reviewedAt(T2).build();
        givenLog(row("K01", "PASS", "FAIL", "a", T1), row("K02", "FAIL", "PASS", "b", T1), row("K01", "PASS", null, null, T2));

        ChecklistReviewResponse response = persistingService().toReviewResponse(checklist);

        assertThat(response.overrides()).containsExactly(new ReviewOverrideResponse("K02", "FAIL", "PASS", "b"));
        assertThat(response.history()).extracting(ChecklistDecisionResponse::decision).containsExactly("FAIL", "PASS", null);
    }

    @Test
    void checklistDecision_isAlwaysNew() {
        ChecklistDecision row = ChecklistDecision.builder().id(UUID.randomUUID()).build();

        assertThat(row.isNew()).isTrue();
    }
}
