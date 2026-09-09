package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.UUID;

@Slf4j
@Service
public class MediaAssetService {

    private final MediaAssetRepository mediaAssetRepository;

    public MediaAssetService(MediaAssetRepository mediaAssetRepository) {
        this.mediaAssetRepository = mediaAssetRepository;
    }

    /** Owner or ADMIN only - not found (not forbidden) for anyone else, so existence isn't leaked. */
    public MediaAsset getForUser(UUID id, User currentUser) {
        MediaAsset asset = mediaAssetRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Media asset not found: " + id));

        boolean isOwner = asset.getUserId().equals(currentUser.getId());
        boolean isAdmin = currentUser.getRole() == Role.ADMIN;
        if (!isOwner && !isAdmin) {
            throw new ResourceNotFoundException("Media asset not found: " + id);
        }
        return asset;
    }

    /** Called by TranscribeStatusPollingService for each SQS message it successfully parses. */
    @Transactional
    public void applyTranscribeStatusUpdate(String jobName, MediaAssetStatus newStatus, String transcriptUri, String failureReason) {
        mediaAssetRepository.findByTranscribeJobName(jobName).ifPresentOrElse(asset -> {
            asset.setStatus(newStatus);
            asset.setTranscriptUri(transcriptUri);
            asset.setFailureReason(failureReason);
            asset.setUpdatedAt(LocalDateTime.now());
            mediaAssetRepository.save(asset);
        }, () -> log.warn("No MediaAsset found for transcribe job '{}': skipping", jobName));
    }
}
