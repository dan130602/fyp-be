package com.fyp.be.media_processing_service.controllers;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import com.fyp.be.media_processing_service.dtos.ChecklistItemResponse;
import com.fyp.be.media_processing_service.services.ChecklistService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Tag(name = "Checklists", description = "Compliance checklists applied to calls")
@Slf4j
@RestController
@RequestMapping("/api/checklists")
public class ChecklistController {

    private final ChecklistService checklistService;

    public ChecklistController(ChecklistService checklistService) {
        this.checklistService = checklistService;
    }

    @Operation(summary = "Get the default checklist",
            description = "Taken from CHECKLIST_JSON. Empty when it is unset or not a JSON array, in which case fyp-ai uses its built-in checklist.")
    @GetMapping("/default")
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<List<ChecklistItemResponse>> getDefault() {
        return ResponseEntity.ok(checklistService.getDefaultChecklist());
    }
}
