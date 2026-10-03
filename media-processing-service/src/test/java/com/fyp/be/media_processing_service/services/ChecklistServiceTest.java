package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ChecklistItemResponse;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class ChecklistServiceTest {

    private ChecklistService serviceWith(String json) {
        return new ChecklistService(
                mock(ChecklistRepository.class),
                mock(LanggraphOutputRepository.class),
                new ObjectMapper(),
                json);
    }

    @Test
    void getDefaultChecklist_validArray_returnsItems() {
        List<ChecklistItemResponse> items = serviceWith(
                "[{\"id\":\"q1\",\"question\":\"Greeting?\"},{\"id\":\"q2\",\"question\":\"Disclosure?\"}]")
                .getDefaultChecklist();

        assertThat(items).containsExactly(
                new ChecklistItemResponse("q1", "Greeting?"),
                new ChecklistItemResponse("q2", "Disclosure?"));
    }

    @Test
    void getDefaultChecklist_emptyObject_returnsEmpty() {
        assertThat(serviceWith("{}").getDefaultChecklist()).isEmpty();
    }

    @Test
    void getDefaultChecklist_blank_returnsEmpty() {
        assertThat(serviceWith("").getDefaultChecklist()).isEmpty();
    }
}
