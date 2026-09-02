package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.MediaAsset;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;

public interface MediaAssetRepository extends JpaRepository<MediaAsset, UUID> {
}
