package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AdvisorResponse;
import com.fyp.be.media_processing_service.exceptions.InvalidFileException;
import com.fyp.be.media_processing_service.models.Advisor;
import com.fyp.be.media_processing_service.models.AdvisorStatus;
import com.fyp.be.media_processing_service.repositories.AdvisorRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
public class AdvisorService {

    private final AdvisorRepository advisorRepository;

    public AdvisorService(AdvisorRepository advisorRepository) {
        this.advisorRepository = advisorRepository;
    }

    /** Active advisors only - inactive ones stay in the table for historical uploads but can't be picked. */
    public List<AdvisorResponse> listActive() {
        return advisorRepository.findAllByStatusOrderByFullNameAsc(AdvisorStatus.ACTIVE).stream()
                .map(AdvisorResponse::from)
                .toList();
    }

    /** Rejects (400) an upload attributed to a missing, unknown or inactive advisor. */
    public Advisor requireActive(UUID advisorId) {
        if (advisorId == null) {
            throw new InvalidFileException("An advisor must be selected for every uploaded file");
        }
        return advisorRepository.findById(advisorId)
                .filter(advisor -> advisor.getStatus() == AdvisorStatus.ACTIVE)
                .orElseThrow(() -> new InvalidFileException(
                        "Advisor '%s' does not exist or is inactive".formatted(advisorId)));
    }
}
