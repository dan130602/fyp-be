package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import com.fyp.be.media_processing_service.dtos.ChecklistReviewResponse;
import com.fyp.be.media_processing_service.dtos.MediaAssetAnalysisResponse;
import com.fyp.be.media_processing_service.dtos.MediaAssetResponse;
import com.fyp.be.media_processing_service.dtos.PlaybackUrlResponse;
import com.fyp.be.media_processing_service.dtos.SubmitReviewRequest;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.services.CurrentUserService;
import com.fyp.be.media_processing_service.services.MediaAssetService;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@Tag(name = "Media assets", description = "Uploaded recordings, their AI analysis and officer review")
@Slf4j
@RestController
@RequestMapping("/api/media-assets")
public class MediaAssetController {

    private final MediaAssetService mediaAssetService;
    private final CurrentUserService currentUserService;

    public MediaAssetController(MediaAssetService mediaAssetService, CurrentUserService currentUserService) {
        this.mediaAssetService = mediaAssetService;
        this.currentUserService = currentUserService;
    }

    @Operation(summary = "List my uploads",
            description = "The caller's own uploads, newest first. Admins also see only their own here.")
    @GetMapping
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<List<MediaAssetResponse>> list(Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.ok(mediaAssetService.listForUser(currentUser.getId()));
    }

    @Operation(summary = "Get a playback URL",
            description = "Presigned S3 GET on the main upload bucket. Uploader only; anyone else gets 404.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "404", description = "Asset not found, or not yours", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @GetMapping("/{id}/playback-url")
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<PlaybackUrlResponse> playbackUrl(@PathVariable UUID id, Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.ok(mediaAssetService.getPlaybackUrl(id, currentUser.getId()));
    }

    @Operation(summary = "Get AI analysis and review",
            description = "Works for any asset because the org-wide dashboard links here. analysisStatus is PENDING while transcription or AI is running, READY once output is stored, FAILED otherwise. result is fyp-ai's raw output; review is null until someone has reviewed.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "404", description = "Asset not found", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @GetMapping("/{id}/analysis")
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'MANAGER', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<MediaAssetAnalysisResponse> analysis(@PathVariable UUID id) {
        return ResponseEntity.ok(mediaAssetService.getAnalysis(id));
    }

    @Operation(summary = "Submit a review",
            description = "Uploader only. Send only the items you disagree with: an empty overrides list accepts the AI verdicts. Each submission replaces the previous one; unchanged overrides are not logged again and dropped ones are logged as a revert to the AI verdict.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "400", description = "Unknown or duplicate item, invalid decision, or decision equal to the AI verdict", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "404", description = "Asset not found, or not yours", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "409", description = "AI analysis is not ready for review", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @PutMapping("/{id}/review")
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<ChecklistReviewResponse> submitReview(
            @PathVariable UUID id, @Valid @RequestBody SubmitReviewRequest request, Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.ok(mediaAssetService.submitReview(id, currentUser.getId(), request));
    }
}
