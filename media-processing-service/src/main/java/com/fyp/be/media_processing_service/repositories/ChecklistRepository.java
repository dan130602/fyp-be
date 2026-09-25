package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.Checklist;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;
import java.util.UUID;

public interface ChecklistRepository extends JpaRepository<Checklist, UUID> {
    Optional<Checklist> findByMediaAssetId(UUID mediaAssetId);
}
