package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.ChecklistItemRequest;
import com.fyp.be.media_processing_service.dtos.ChecklistItemResponse;
import com.fyp.be.media_processing_service.models.Checklist;
import com.fyp.be.media_processing_service.models.ChecklistStatus;
import com.fyp.be.media_processing_service.models.LanggraphOutput;
import com.fyp.be.media_processing_service.repositories.ChecklistRepository;
import com.fyp.be.media_processing_service.repositories.LanggraphOutputRepository;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import tools.jackson.databind.ObjectMapper;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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

    private final ChecklistRepository checklistRepository = mock(ChecklistRepository.class);
    private final LanggraphOutputRepository outputRepository = mock(LanggraphOutputRepository.class);

    private ChecklistService persistingService() {
        when(checklistRepository.save(any(Checklist.class))).thenAnswer(inv -> inv.getArgument(0));
        return new ChecklistService(checklistRepository, outputRepository, new ObjectMapper(), "[{\"id\":\"d\",\"question\":\"Default?\"}]");
    }

    @Test
    void createForUpload_withItems_storesSubmittedItemsAsSubmitted() {
        UUID assetId = UUID.randomUUID();
        UUID userId = UUID.randomUUID();

        Checklist saved = persistingService().createForUpload(assetId, userId, List.of(new ChecklistItemRequest("q1", "Greeting?")));

        assertThat(saved.getMediaAssetId()).isEqualTo(assetId);
        assertThat(saved.getSubmittedBy()).isEqualTo(userId);
        assertThat(saved.getStatus()).isEqualTo(ChecklistStatus.SUBMITTED);
        assertThat(saved.getChecklist()).contains("\"q1\"").contains("Greeting?");
    }

    @Test
    void createForUpload_nullOrEmptyItems_fallsBackToDefault() {
        ChecklistService service = persistingService();

        assertThat(service.createForUpload(UUID.randomUUID(), UUID.randomUUID(), null).getChecklist())
                .isEqualTo("[{\"id\":\"d\",\"question\":\"Default?\"}]");
        assertThat(service.createForUpload(UUID.randomUUID(), UUID.randomUUID(), List.of()).getChecklist())
                .isEqualTo("[{\"id\":\"d\",\"question\":\"Default?\"}]");
    }

    @Test
    void getOrCreateForAsset_existing_returnsItWithoutSaving() {
        UUID assetId = UUID.randomUUID();
        Checklist existing = Checklist.builder().id(UUID.randomUUID()).mediaAssetId(assetId).build();
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.of(existing));

        assertThat(persistingService().getOrCreateForAsset(assetId)).isSameAs(existing);
        verify(checklistRepository, never()).save(any());
    }

    @Test
    void getOrCreateForAsset_missing_createsPendingReviewFromDefault() {
        UUID assetId = UUID.randomUUID();
        when(checklistRepository.findByMediaAssetId(assetId)).thenReturn(Optional.empty());

        Checklist created = persistingService().getOrCreateForAsset(assetId);

        assertThat(created.getMediaAssetId()).isEqualTo(assetId);
        assertThat(created.getSubmittedBy()).isNull();
        assertThat(created.getStatus()).isEqualTo(ChecklistStatus.PENDING_REVIEW);
        assertThat(created.getChecklist()).isEqualTo("[{\"id\":\"d\",\"question\":\"Default?\"}]");
    }

    @Test
    void saveLanggraphOutput_noExistingRow_insertsNew() {
        UUID checklistId = UUID.randomUUID();
        when(outputRepository.findByChecklistId(checklistId)).thenReturn(Optional.empty());

        persistingService().saveLanggraphOutput(checklistId, "{\"a\":1}");

        ArgumentCaptor<LanggraphOutput> captor = ArgumentCaptor.forClass(LanggraphOutput.class);
        verify(outputRepository).save(captor.capture());
        assertThat(captor.getValue().getChecklistId()).isEqualTo(checklistId);
        assertThat(captor.getValue().getOutput()).isEqualTo("{\"a\":1}");
        assertThat(captor.getValue().getId()).isNotNull();
    }

    @Test
    void saveLanggraphOutput_existingRow_updatesInPlace() {
        UUID checklistId = UUID.randomUUID();
        UUID rowId = UUID.randomUUID();
        LanggraphOutput existing = LanggraphOutput.builder().id(rowId).checklistId(checklistId).output("old").build();
        when(outputRepository.findByChecklistId(checklistId)).thenReturn(Optional.of(existing));

        persistingService().saveLanggraphOutput(checklistId, "new");

        verify(outputRepository).save(existing);
        assertThat(existing.getId()).isEqualTo(rowId);
        assertThat(existing.getOutput()).isEqualTo("new");
        assertThat(existing.getUpdatedAt()).isNotNull();
    }
}
