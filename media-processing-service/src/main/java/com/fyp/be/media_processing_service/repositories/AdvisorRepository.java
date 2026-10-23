package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.Advisor;
import com.fyp.be.media_processing_service.models.AdvisorStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface AdvisorRepository extends JpaRepository<Advisor, UUID> {
    List<Advisor> findAllByStatusOrderByFullNameAsc(AdvisorStatus status);
}
