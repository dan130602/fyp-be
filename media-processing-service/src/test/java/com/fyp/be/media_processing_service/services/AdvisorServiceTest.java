package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.AdvisorResponse;
import com.fyp.be.media_processing_service.exceptions.InvalidFileException;
import com.fyp.be.media_processing_service.models.Advisor;
import com.fyp.be.media_processing_service.models.AdvisorStatus;
import com.fyp.be.media_processing_service.repositories.AdvisorRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AdvisorServiceTest {

    @Mock
    private AdvisorRepository advisorRepository;

    private final UUID advisorId = UUID.randomUUID();

    private AdvisorService service() {
        return new AdvisorService(advisorRepository);
    }

    private Advisor advisor(AdvisorStatus status) {
        return Advisor.builder()
                .id(advisorId)
                .employeeId("E100002")
                .fullName("Phua Chiu Cheng")
                .email("phua.chiu.cheng@clearcall-bank.example")
                .department("Private Banking")
                .status(status)
                .build();
    }

    @Test
    void listActiveMapsActiveAdvisors() {
        when(advisorRepository.findAllByStatusOrderByFullNameAsc(AdvisorStatus.ACTIVE))
                .thenReturn(List.of(advisor(AdvisorStatus.ACTIVE)));

        List<AdvisorResponse> result = service().listActive();

        assertThat(result).containsExactly(new AdvisorResponse(
                advisorId, "E100002", "Phua Chiu Cheng", "phua.chiu.cheng@clearcall-bank.example", "Private Banking"));
    }

    @Test
    void requireActiveReturnsActiveAdvisor() {
        when(advisorRepository.findById(advisorId)).thenReturn(Optional.of(advisor(AdvisorStatus.ACTIVE)));

        assertThat(service().requireActive(advisorId).getEmployeeId()).isEqualTo("E100002");
    }

    @Test
    void requireActiveRejectsInactiveAdvisor() {
        when(advisorRepository.findById(advisorId)).thenReturn(Optional.of(advisor(AdvisorStatus.INACTIVE)));

        assertThatThrownBy(() -> service().requireActive(advisorId)).isInstanceOf(InvalidFileException.class);
    }

    @Test
    void requireActiveRejectsUnknownAdvisor() {
        when(advisorRepository.findById(advisorId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service().requireActive(advisorId)).isInstanceOf(InvalidFileException.class);
    }

    @Test
    void requireActiveRejectsMissingAdvisor() {
        assertThatThrownBy(() -> service().requireActive(null)).isInstanceOf(InvalidFileException.class);
    }
}
