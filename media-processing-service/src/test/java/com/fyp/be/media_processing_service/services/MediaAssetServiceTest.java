package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AnalysisStatus;
import com.fyp.be.media_processing_service.dtos.MediaAssetAnalysisResponse;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tools.jackson.databind.ObjectMapper;

import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
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

    private final UUID ownerId = UUID.randomUUID();
    private final UUID assetId = UUID.randomUUID();

    private MediaAssetService service() {
        return new MediaAssetService(mediaAssetRepository, presignedUrlService, transcriptHandoffService,
                checklistRepository, langgraphOutputRepository, new ObjectMapper());
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
    void unreadableOutputIsFailed() {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(MediaAssetStatus.COMPLETED, null)));
        givenOutput("not json {");

        MediaAssetAnalysisResponse result = service().getAnalysis(assetId, ownerId);

        assertThat(result.analysisStatus()).isEqualTo(AnalysisStatus.FAILED);
        assertThat(result.failureReason()).isEqualTo("AI analysis returned an unreadable result");
        assertThat(result.result()).isNull();
    }
}
