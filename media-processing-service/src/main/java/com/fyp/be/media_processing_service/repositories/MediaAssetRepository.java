package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.MediaAsset;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;
import java.util.List;
import java.util.UUID;

public interface MediaAssetRepository extends JpaRepository<MediaAsset, UUID> {
    List<MediaAsset> findByUserIdOrderByCreatedAtDesc(UUID userId);
    Optional<MediaAsset> findByTranscribeJobName(String transcribeJobName);
}
