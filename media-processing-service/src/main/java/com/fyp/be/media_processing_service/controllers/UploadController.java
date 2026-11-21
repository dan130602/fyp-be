package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
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

@Tag(name = "Uploads", description = "Two-step upload: get presigned S3 URLs here, then PUT the file bytes to each URL")
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

    @Operation(summary = "Request presigned upload URLs",
            description = "The whole request is validated before anything is written, so one bad file means no URLs are issued. The returned id is the media asset id used by every other endpoint; PUT the bytes with the same Content-Type you declared before expiresIn runs out.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "400", description = "Content type, extension, size or file count not allowed; advisor unknown or inactive; or batch mode without a batch bucket", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "500", description = "Failed to generate upload URL", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @PostMapping("/presigned-urls")
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<PresignedUploadResponse> presignedUrls(
            @Valid @RequestBody PresignedUploadRequest request,
            Authentication authentication
    ) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        PresignedUploadResponse response = presignedUrlService.generatePresignedUploads(
                currentUser.getId(), request.files(), request.processingMode());
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get upload status",
            description = "Readable by the uploader or any ADMIN/SUPER_ADMIN; anyone else gets 404. status covers transcription only; use the analysis endpoint to see whether the AI step has finished.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "404", description = "Upload not found, or not yours", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @GetMapping("/{id}")
    @PreAuthorize("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')")
    public ResponseEntity<MediaAssetStatusResponse> getStatus(@PathVariable UUID id, Authentication authentication) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        MediaAsset asset = mediaAssetService.getForUser(id, currentUser);
        return ResponseEntity.ok(MediaAssetStatusResponse.from(asset));
    }
}
