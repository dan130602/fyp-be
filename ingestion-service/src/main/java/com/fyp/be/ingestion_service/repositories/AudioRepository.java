package com.fyp.be.ingestion_service.repositories;

import com.fyp.be.ingestion_service.models.Audio;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.UUID;

public interface AudioRepository extends JpaRepository<Audio, UUID> {
}
