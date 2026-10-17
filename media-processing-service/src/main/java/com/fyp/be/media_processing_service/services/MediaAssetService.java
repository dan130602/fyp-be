package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AnalysisStatus;
import com.fyp.be.media_processing_service.dtos.ApiTime;
import com.fyp.be.media_processing_service.dtos.MediaAssetAnalysisResponse;
import com.fyp.be.media_processing_service.dtos.MediaAssetResponse;
import com.fyp.be.media_processing_service.dtos.PlaybackUrlResponse;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.models.ProcessingMode;
import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Slf4j
@Service
public class MediaAssetService {

    private final MediaAssetRepository mediaAssetRepository;
    private final PresignedUrlService presignedUrlService;
    private final TranscriptHandoffService transcriptHandoffService;
    private final ChecklistRepository checklistRepository;
    private final LanggraphOutputRepository langgraphOutputRepository;
    private final BatchObjectService batchObjectService;
    private final ObjectMapper objectMapper;

    public MediaAssetService(
            MediaAssetRepository mediaAssetRepository,
            PresignedUrlService presignedUrlService,
            TranscriptHandoffService transcriptHandoffService,
            ChecklistRepository checklistRepository,
            LanggraphOutputRepository langgraphOutputRepository,
            BatchObjectService batchObjectService,
            ObjectMapper objectMapper
    ) {
        this.mediaAssetRepository = mediaAssetRepository;
        this.presignedUrlService = presignedUrlService;
        this.transcriptHandoffService = transcriptHandoffService;
        this.checklistRepository = checklistRepository;
        this.langgraphOutputRepository = langgraphOutputRepository;
        this.batchObjectService = batchObjectService;
        this.objectMapper = objectMapper;
    }

    public List<MediaAssetResponse> listForUser(UUID userId) {
        return mediaAssetRepository.findByUserIdOrderByCreatedAtDesc(userId).stream()
                .map(this::toResponse)
                .toList();
    }

    // Same 404 for missing and not-yours - a 403 would confirm the id exists.
    public PlaybackUrlResponse getPlaybackUrl(UUID assetId, UUID userId) {
        MediaAsset asset = mediaAssetRepository.findById(assetId)
                .orElseThrow(() -> new ResourceNotFoundException("Media asset not found: " + assetId));

        if (!asset.getUserId().equals(userId)) {
            throw new ResourceNotFoundException("Media asset not found: " + assetId);
        }

        String url = presignedUrlService.generatePlaybackUrl(asset.getObjectKey());
        long expiresIn = Duration.ofMinutes(presignedUrlService.getPlaybackExpiryMinutes()).toSeconds();
        return new PlaybackUrlResponse(url, expiresIn);
    }

    // Same 404 for missing and not-yours, as getPlaybackUrl.
    public MediaAssetAnalysisResponse getAnalysis(UUID assetId, UUID userId) {
        MediaAsset asset = mediaAssetRepository.findById(assetId)
                .orElseThrow(() -> new ResourceNotFoundException("Media asset not found: " + assetId));

        if (!asset.getUserId().equals(userId)) {
            throw new ResourceNotFoundException("Media asset not found: " + assetId);
        }

        if (asset.getStatus() == MediaAssetStatus.FAILED) {
            return analysisResponse(asset, AnalysisStatus.FAILED, asset.getFailureReason(), null);
        }

        Optional<LanggraphOutput> output = checklistRepository.findByMediaAssetId(assetId)
                .map(Checklist::getId)
                .flatMap(langgraphOutputRepository::findByChecklistId);
        if (output.isEmpty()) {
            return analysisResponse(asset, AnalysisStatus.PENDING, null, null);
        }

        JsonNode result;
        try {
            result = objectMapper.readTree(output.get().getOutput());
        } catch (JacksonException ex) {
            log.error("Stored AI output for media asset {} is not valid JSON", assetId, ex);
            return analysisResponse(asset, AnalysisStatus.FAILED, "AI analysis returned an unreadable result", null);
        }
        if (result.has("__error__")) {
            return analysisResponse(asset, AnalysisStatus.FAILED, "AI analysis failed", null);
        }
        return analysisResponse(asset, AnalysisStatus.READY, null, result);
    }

    private MediaAssetAnalysisResponse analysisResponse(
            MediaAsset asset, AnalysisStatus status, String failureReason, JsonNode result) {
        return new MediaAssetAnalysisResponse(
                asset.getId(),
                asset.getOriginalFilename(),
                asset.getStatus(),
                status,
                failureReason,
                result
        );
    }

    /** Owner or ADMIN/SUPER_ADMIN only - not found (not forbidden) for anyone else, so existence isn't leaked. */
    public MediaAsset getForUser(UUID id, User currentUser) {
        MediaAsset asset = mediaAssetRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Media asset not found: " + id));

        boolean isOwner = asset.getUserId().equals(currentUser.getId());
        boolean isAdmin = currentUser.getRole() == Role.ADMIN || currentUser.getRole() == Role.SUPER_ADMIN;
        if (!isOwner && !isAdmin) {
            throw new ResourceNotFoundException("Media asset not found: " + id);
        }
        return asset;
    }

    /** Called by TranscribeStatusPollingService for each SQS message it successfully parses.
     * On COMPLETED, also hands the transcript + checklist off to the next service - a failure
     * there rolls back this whole update (see TranscriptHandoffService), so the incoming status
     * message gets redelivered and retried rather than silently losing the handoff. */
    @Transactional
    public void applyTranscribeStatusUpdate(String jobName, MediaAssetStatus newStatus, String transcriptUri, String failureReason) {
        mediaAssetRepository.findByTranscribeJobName(jobName).ifPresentOrElse(asset -> {
            asset.setStatus(newStatus);
            asset.setTranscriptUri(transcriptUri);
            asset.setFailureReason(failureReason);
            asset.setUpdatedAt(LocalDateTime.now());
            mediaAssetRepository.save(asset);
            if (newStatus == MediaAssetStatus.COMPLETED) {
                transcriptHandoffService.publish(asset);
                // Last, so a failed handoff (which rolls this update back for redelivery) never
                // leaves a batch object deleted while its row is not completed.
                if (asset.getProcessingMode() == ProcessingMode.BATCH) {
                    batchObjectService.delete(asset.getObjectKey());
                }
            }
        }, () -> log.warn("No MediaAsset found for transcribe job '{}': skipping", jobName));
    }

    private MediaAssetResponse toResponse(MediaAsset asset) {
        return new MediaAssetResponse(
                asset.getId(),
                asset.getOriginalFilename(),
                asset.getContentType(),
                asset.getFileSizeBytes(),
                asset.getStatus(),
                asset.getProcessingMode(),
                ApiTime.of(asset.getCreatedAt())
        );
    }
}
