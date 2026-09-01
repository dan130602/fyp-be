package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.Audio;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.UUID;

public interface AudioRepository extends JpaRepository<Audio, UUID> {
}
