package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.MediaAssetStatusResponse;
import com.fyp.be.media_processing_service.dtos.PresignedUploadRequest;
import com.fyp.be.media_processing_service.dtos.PresignedUploadResponse;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.services.CurrentUserService;
import com.fyp.be.media_processing_service.services.MediaAssetService;
import com.fyp.be.media_processing_service.services.PresignedUrlService;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.UUID;

@Slf4j
@RestController
@RequestMapping("/api/uploads")
public class UploadController {

    private final PresignedUrlService presignedUrlService;
    private final MediaAssetService mediaAssetService;
    private final CurrentUserService currentUserService;

    public UploadController(
            PresignedUrlService presignedUrlService,
            MediaAssetService mediaAssetService,
            CurrentUserService currentUserService
    ) {
        this.presignedUrlService = presignedUrlService;
        this.mediaAssetService = mediaAssetService;
        this.currentUserService = currentUserService;
    }

    @PostMapping("/presigned-urls")
    @PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<PresignedUploadResponse> presignedUrls(
            @Valid @RequestBody PresignedUploadRequest request,
            Authentication authentication
    ) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        PresignedUploadResponse response = presignedUrlService.generatePresignedUploads(currentUser.getId(), request.files());
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{id}")
    @PreAuthorize("hasAnyRole('ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<MediaAssetStatusResponse> getStatus(@PathVariable UUID id, Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        MediaAsset asset = mediaAssetService.getForUser(id, currentUser);
        return ResponseEntity.ok(MediaAssetStatusResponse.from(asset));
    }
}
