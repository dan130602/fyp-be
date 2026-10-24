package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AnalysisStatus;
import com.fyp.be.media_processing_service.dtos.ApiTime;
import com.fyp.be.media_processing_service.dtos.ChecklistDecisionResponse;
import com.fyp.be.media_processing_service.dtos.ChecklistReviewResponse;
import com.fyp.be.media_processing_service.dtos.MediaAssetAnalysisResponse;
import com.fyp.be.media_processing_service.dtos.MediaAssetResponse;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideRequest;
import com.fyp.be.media_processing_service.dtos.ReviewOverrideResponse;
import com.fyp.be.media_processing_service.dtos.SubmitReviewRequest;
import com.fyp.be.media_processing_service.exceptions.AnalysisNotReadyException;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.Advisor;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistDecision;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.models.ProcessingMode;
import com.fyp.be.media_processing_service.repositories.AdvisorRepository;
import com.fyp.be.media_processing_service.repositories.ChecklistDecisionRepository;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tools.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.never;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class MediaAssetServiceTest {

    @Mock
    private MediaAssetRepository mediaAssetRepository;

    @Mock
    private PresignedUrlService presignedUrlService;

    @Mock
    private TranscriptHandoffService transcriptHandoffService;

    @Mock
    private ChecklistRepository checklistRepository;

    @Mock
    private LanggraphOutputRepository langgraphOutputRepository;

    @Mock
    private ChecklistDecisionRepository checklistDecisionRepository;

    @Mock
    private BatchObjectService batchObjectService;

    @Mock
    private AdvisorRepository advisorRepository;

    private final UUID ownerId = UUID.randomUUID();
    private final UUID assetId = UUID.randomUUID();

    private MediaAssetService service() {
        return new MediaAssetService(mediaAssetRepository, presignedUrlService, transcriptHandoffService,
                checklistRepository, langgraphOutputRepository, batchObjectService, advisorRepository,
                checklistService(), new ObjectMapper());
    }

    private ChecklistService checklistService() {
        return new ChecklistService(checklistRepository, langgraphOutputRepository, checklistDecisionRepository, new ObjectMapper(), "[]");
    }

    private MediaAsset asset(MediaAssetStatus status, String failureReason) {
        return MediaAsset.builder()
                .id(assetId)
                .userId(ownerId)
                .originalFilename("call.mp3")
                .status(status)
                .failureReason(failureReason)
                .build();
    }

    private void givenOutput(String json) {
        UUID checklistId = UUID.randomUUID();
        when(checklistRepository.findByMediaAssetId(assetId))
                .thenReturn(Optional.of(Checklist.builder().id(checklistId).mediaAssetId(assetId).build()));
        when(langgraphOutputRepository.findByChecklistId(checklistId))
                .thenReturn(Optional.of(LanggraphOutput.builder().checklistId(checklistId).output(json).build()));
    }

    @Test
    void listForUserAttachesAdvisorAndLeavesLegacyAssetsWithout() {
        UUID advisorId = UUID.randomUUID();
        MediaAsset withAdvisor = MediaAsset.builder().id(assetId).userId(ownerId).advisorId(advisorId).build();
        MediaAsset legacy = MediaAsset.builder().id(UUID.randomUUID()).userId(ownerId).build();
        when(mediaAssetRepository.findByUserIdOrderByCreatedAtDesc(ownerId)).thenReturn(List.of(withAdvisor, legacy));
        when(advisorRepository.findAllById(List.of(advisorId))).thenReturn(List.of(Advisor.builder()
                .id(advisorId).employeeId("E100001").fullName("Hugh Mungus")
                .email("hugh.mungus@clearcall-bank.example").department("Private Banking").build()));

        List<MediaAssetResponse> result = service().listForUser(ownerId);

        assertThat(result.get(0).advisor().employeeId()).isEqualTo("E100001");
        assertThat(result.get(1).advisor()).isNull();
    }

    @Test
    void missingAssetIsNotFound() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service().getAnalysis(assetId, ownerId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void someoneElsesAssetIsNotFound() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));

        assertThatThrownBy(() -> service().getAnalysis(assetId, UUID.randomUUID()))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void noOutputRowIsPending() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.empty());

        MediaAssetAnalysisResponse result = service().getAnalysis(assetId, ownerId);

        assertThat(result.analysisStatus()).isEqualTo(AnalysisStatus.PENDING);
        assertThat(result.result()).isNull();
    }

    @Test
    void parsedOutputIsReady() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        givenOutput("{\"call_id\":\"c1\",\"segments\":[]}");

        MediaAssetAnalysisResponse result = service().getAnalysis(assetId, ownerId);

        assertThat(result.analysisStatus()).isEqualTo(AnalysisStatus.READY);
        assertThat(result.fileName()).isEqualTo("call.mp3");
        assertThat(result.result().get("call_id").asString()).isEqualTo("c1");
    }

    @Test
    void errorOutputIsFailed() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        givenOutput("{\"__error__\":{\"message\":\"boom\"}}");

        MediaAssetAnalysisResponse result = service().getAnalysis(assetId, ownerId);

        assertThat(result.analysisStatus()).isEqualTo(AnalysisStatus.FAILED);
        assertThat(result.failureReason()).isEqualTo("AI analysis failed");
        assertThat(result.result()).isNull();
    }

    @Test
    void failedAssetIsFailedWithItsReason() {
        when(mediaAssetRepository.findById(assetId))
                .thenReturn(Optional.of(asset(MediaAssetStatus.FAILED, "Transcribe failed")));

        MediaAssetAnalysisResponse result = service().getAnalysis(assetId, ownerId);

        assertThat(result.analysisStatus()).isEqualTo(AnalysisStatus.FAILED);
        assertThat(result.failureReason()).isEqualTo("Transcribe failed");
        assertThat(result.result()).isNull();
    }

    @Test
    void completedBatchAssetDeletesItsBatchObjectAfterHandoff() {
        MediaAsset batch = asset(MediaAssetStatus.QUEUED, null);
        batch.setProcessingMode(ProcessingMode.BATCH);
        batch.setObjectKey(assetId + "/call.mp3");
        when(mediaAssetRepository.findByTranscribeJobName("job-1")).thenReturn(Optional.of(batch));

        service().applyTranscribeStatusUpdate("job-1", MediaAssetStatus.COMPLETED, "s3://out/t.json", null);

        assertThat(batch.getStatus()).isEqualTo(MediaAssetStatus.COMPLETED);
        verify(transcriptHandoffService).publish(batch);
        verify(batchObjectService).delete(assetId + "/call.mp3");
    }

    @Test
    void failedBatchAssetKeepsItsBatchObject() {
        MediaAsset batch = asset(MediaAssetStatus.PROCESSING, null);
        batch.setProcessingMode(ProcessingMode.BATCH);
        when(mediaAssetRepository.findByTranscribeJobName("job-1")).thenReturn(Optional.of(batch));

        service().applyTranscribeStatusUpdate("job-1", MediaAssetStatus.FAILED, null, "bad audio");

        assertThat(batch.getStatus()).isEqualTo(MediaAssetStatus.FAILED);
        verifyNoInteractions(batchObjectService);
    }

    @Test
    void completedImmediateAssetNeverTouchesBatchBucket() {
        MediaAsset immediate = asset(MediaAssetStatus.PENDING, null);
        when(mediaAssetRepository.findByTranscribeJobName("job-1")).thenReturn(Optional.of(immediate));

        service().applyTranscribeStatusUpdate("job-1", MediaAssetStatus.COMPLETED, "s3://out/t.json", null);

        verify(batchObjectService, never()).delete(org.mockito.ArgumentMatchers.any());
    }

    @Test
    void unreadableOutputIsFailed() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        givenOutput("not json {");

        MediaAssetAnalysisResponse result = service().getAnalysis(assetId, ownerId);

        assertThat(result.analysisStatus()).isEqualTo(AnalysisStatus.FAILED);
        assertThat(result.failureReason()).isEqualTo("AI analysis returned an unreadable result");
        assertThat(result.result()).isNull();
    }

    @Test
    void neverReviewedChecklistHasNoReview() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        givenOutput("{\"final_results\":[]}");

        assertThat(service().getAnalysis(assetId, ownerId).review()).isNull();
    }

    @Test
    void analysisIncludesStoredReview() {
        UUID checklistId = UUID.randomUUID();
        LocalDateTime reviewedAt = LocalDateTime.of(2026, 10, 24, 9, 30);
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.of(Checklist.builder()
                .id(checklistId).mediaAssetId(assetId)
                .reviewedBy(ownerId).reviewedAt(reviewedAt).build()));
        when(checklistDecisionRepository.findByChecklistIdOrderByDecidedAtAsc(checklistId)).thenReturn(List.of(
                ChecklistDecision.builder().checklistId(checklistId).checklistItemId("K01").aiVerdict("PASS")
                        .decision("FAIL").comment("missed it").decidedBy(ownerId).decidedAt(reviewedAt).build()));
        when(langgraphOutputRepository.findByChecklistId(checklistId))
                .thenReturn(Optional.of(LanggraphOutput.builder().checklistId(checklistId).output("{\"final_results\":[]}").build()));

        ChecklistReviewResponse review = service().getAnalysis(assetId, ownerId).review();

        assertThat(review.reviewedBy()).isEqualTo(ownerId);
        assertThat(review.reviewedAt()).isEqualTo(ApiTime.of(reviewedAt));
        assertThat(review.overrides()).containsExactly(new ReviewOverrideResponse("K01", "PASS", "FAIL", "missed it"));
        assertThat(review.history()).containsExactly(new ChecklistDecisionResponse(
                "K01", "PASS", "FAIL", "missed it", ownerId, ApiTime.of(reviewedAt)));
    }

    @Test
    void submitReviewOnSomeoneElsesAssetIsNotFound() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));

        assertThatThrownBy(() -> service().submitReview(assetId, UUID.randomUUID(), new SubmitReviewRequest(List.of())))
                .isInstanceOf(ResourceNotFoundException.class);
        verify(checklistRepository, never()).save(any());
    }

    @Test
    void submitReviewOnMissingAssetIsNotFound() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service().submitReview(assetId, ownerId, new SubmitReviewRequest(List.of())))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void submitReviewWithoutChecklistIsNotReady() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service().submitReview(assetId, ownerId, new SubmitReviewRequest(List.of())))
                .isInstanceOf(AnalysisNotReadyException.class);
    }

    @Test
    void submitReviewDelegatesToChecklistService() {
        UUID checklistId = UUID.randomUUID();
        Checklist checklist = Checklist.builder().id(checklistId).mediaAssetId(assetId).build();
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.of(checklist));
        when(checklistRepository.findById(checklistId)).thenReturn(Optional.of(checklist));
        when(langgraphOutputRepository.findByChecklistId(checklistId)).thenReturn(Optional.of(LanggraphOutput.builder()
                .checklistId(checklistId).output("{\"final_results\":[{\"item_id\":\"K01\",\"status\":\"PASS\"}]}").build()));

        ChecklistReviewResponse response = service().submitReview(assetId, ownerId,
                new SubmitReviewRequest(List.of(new ReviewOverrideRequest("K01", "FAIL", "missed it"))));

        assertThat(response.reviewedBy()).isEqualTo(ownerId);
        assertThat(response.overrides()).containsExactly(new ReviewOverrideResponse("K01", "PASS", "FAIL", "missed it"));
        assertThat(checklist.getStatus()).isEqualTo(ChecklistStatus.REVIEWED);
        verify(checklistRepository).save(checklist);
    }
}
