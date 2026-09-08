package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.MediaAssetResponse;
import com.fyp.be.media_processing_service.dtos.PlaybackUrlResponse;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;
import java.util.UUID;

@Service
public class MediaAssetService {

    private final MediaAssetRepository mediaAssetRepository;
    private final PresignedUrlService presignedUrlService;

    public MediaAssetService(MediaAssetRepository mediaAssetRepository, PresignedUrlService presignedUrlService) {
        this.mediaAssetRepository = mediaAssetRepository;
        this.presignedUrlService = presignedUrlService;
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

    private MediaAssetResponse toResponse(MediaAsset asset) {
        return new MediaAssetResponse(
                asset.getId(),
                asset.getOriginalFilename(),
                asset.getContentType(),
                asset.getFileSizeBytes(),
                asset.getStatus(),
                asset.getCreatedAt()
        );
    }
}
