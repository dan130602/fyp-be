package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.MediaAssetAnalysisResponse;
import com.fyp.be.media_processing_service.dtos.MediaAssetResponse;
import com.fyp.be.media_processing_service.dtos.PlaybackUrlResponse;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.services.CurrentUserService;
import com.fyp.be.media_processing_service.services.MediaAssetService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

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

    @GetMapping
    @PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<List<MediaAssetResponse>> list(Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.ok(mediaAssetService.listForUser(currentUser.getId()));
    }

    @GetMapping("/{id}/playback-url")
    @PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<PlaybackUrlResponse> playbackUrl(@PathVariable UUID id, Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.ok(mediaAssetService.getPlaybackUrl(id, currentUser.getId()));
    }

    @GetMapping("/{id}/analysis")
    @PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<MediaAssetAnalysisResponse> analysis(@PathVariable UUID id, Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.ok(mediaAssetService.getAnalysis(id, currentUser.getId()));
    }
}
