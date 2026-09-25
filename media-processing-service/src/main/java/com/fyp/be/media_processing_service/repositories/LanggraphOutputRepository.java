package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.LanggraphOutput;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;
import java.util.UUID;

public interface LanggraphOutputRepository extends JpaRepository<LanggraphOutput, UUID> {
    Optional<LanggraphOutput> findByChecklistId(UUID checklistId);
}
