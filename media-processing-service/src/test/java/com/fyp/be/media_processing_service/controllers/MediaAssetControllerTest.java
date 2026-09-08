package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.models.MediaAssetStatus;
import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PresignedGetObjectRequest;

import java.net.URI;
import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class MediaAssetControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private UserRepository userRepository;

    @MockitoBean
    private MediaAssetRepository mediaAssetRepository;

    @MockitoBean
    private S3Presigner s3Presigner;

    private final UUID ownerId = UUID.randomUUID();
    private final UUID assetId = UUID.randomUUID();

    @BeforeEach
    void stubPresigner() throws Exception {
        PresignedGetObjectRequest presignedRequest = mock(PresignedGetObjectRequest.class);
        when(presignedRequest.url()).thenReturn(URI.create("https://bucket.s3.amazonaws.com/uploads/mock").toURL());
        when(s3Presigner.presignGetObject(any(software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest.class)))
                .thenReturn(presignedRequest);
    }

    private MediaAsset asset(UUID userId) {
        return MediaAsset.builder()
                .id(assetId)
                .userId(userId)
                .objectKey("uploads/%s/call.mp3".formatted(userId))
                .originalFilename("call.mp3")
                .contentType("audio/mpeg")
                .fileSizeBytes(1024L)
                .status(MediaAssetStatus.PENDING)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
    }

    private MockHttpServletRequestBuilder authorizedGet(String url, String sub, Role role) {
        User user = User.builder()
                .id(ownerId)
                .cognitoSub(sub)
                .email("user@example.com")
                .name("Test User")
                .role(role)
                .status(UserStatus.ACTIVE)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        when(userRepository.findByCognitoSub(sub)).thenReturn(Optional.of(user));

        return get(url)
                .with(SecurityMockMvcRequestPostProcessors.jwt()
                        .jwt(jwt -> jwt.subject(sub))
                        .authorities(new SimpleGrantedAuthority("ROLE_" + role.name())));
    }

    @Test
    void ownerReceivesAPlaybackUrl() throws Exception {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(ownerId)));

        mockMvc.perform(authorizedGet("/api/media-assets/" + assetId + "/playback-url", "sub-owner", Role.COMPLIANCE_OFFICER))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.url").exists())
                .andExpect(jsonPath("$.expiresIn").exists());
    }

    @Test
    void anotherUsersAssetReturnsNotFound() throws Exception {
        UUID someoneElseId = UUID.randomUUID();
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.of(asset(someoneElseId)));

        mockMvc.perform(authorizedGet("/api/media-assets/" + assetId + "/playback-url", "sub-owner", Role.COMPLIANCE_OFFICER))
                .andExpect(status().isNotFound());
    }

    @Test
    void missingAssetReturnsNotFound() throws Exception {
        when(mediaAssetRepository.findById(assetId)).thenReturn(Optional.empty());

        mockMvc.perform(authorizedGet("/api/media-assets/" + assetId + "/playback-url", "sub-owner", Role.COMPLIANCE_OFFICER))
                .andExpect(status().isNotFound());
    }

    @Test
    void unauthenticatedRequestIsRejected() throws Exception {
        mockMvc.perform(get("/api/media-assets/" + assetId + "/playback-url"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void managerRoleIsForbidden() throws Exception {
        mockMvc.perform(authorizedGet("/api/media-assets/" + assetId + "/playback-url", "sub-manager", Role.MANAGER))
                .andExpect(status().isForbidden());
    }

    @Test
    void listReturnsAssetsForCurrentUserWithoutObjectKey() throws Exception {
        when(mediaAssetRepository.findByUserIdOrderByCreatedAtDesc(ownerId)).thenReturn(java.util.List.of(asset(ownerId)));

        mockMvc.perform(authorizedGet("/api/media-assets", "sub-owner", Role.COMPLIANCE_OFFICER))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(1))
                .andExpect(jsonPath("$[0].id").value(assetId.toString()))
                .andExpect(jsonPath("$[0].objectKey").doesNotExist());
    }

    @Test
    void listUnauthenticatedRequestIsRejected() throws Exception {
        mockMvc.perform(get("/api/media-assets"))
                .andExpect(status().isUnauthorized());
    }
}
