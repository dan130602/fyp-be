package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.FileUploadRequest;
import com.fyp.be.media_processing_service.dtos.PresignedUploadRequest;
import com.fyp.be.media_processing_service.models.MediaAsset;
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
import org.springframework.http.MediaType;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PresignedPutObjectRequest;
import tools.jackson.databind.json.JsonMapper;

import java.net.URI;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class UploadControllerTest {

    private static final String UPLOAD_URL = "/api/uploads/presigned-urls";

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private JsonMapper jsonMapper;

    @MockitoBean
    private UserRepository userRepository;

    @MockitoBean
    private S3Presigner s3Presigner;

    @MockitoBean
    private MediaAssetRepository mediaAssetRepository;

    @BeforeEach
    void stubPresigner() throws Exception {
        PresignedPutObjectRequest presignedRequest = mock(PresignedPutObjectRequest.class);
        when(presignedRequest.url()).thenReturn(URI.create("https://bucket.s3.amazonaws.com/uploads/mock").toURL());
        when(s3Presigner.presignPutObject(any(software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest.class)))
                .thenReturn(presignedRequest);

        when(mediaAssetRepository.save(any(MediaAsset.class))).thenAnswer(invocation -> {
            MediaAsset asset = invocation.getArgument(0);
            if (asset.getId() == null) {
                asset.setId(UUID.randomUUID());
            }
            return asset;
        });
    }

    private MockHttpServletRequestBuilder authorizedRequest(String sub, Role role, String body) {
        User user = User.builder()
                .id(UUID.randomUUID())
                .cognitoSub(sub)
                .email("user@example.com")
                .name("Test User")
                .role(role)
                .status(UserStatus.ACTIVE)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        when(userRepository.findByCognitoSub(sub)).thenReturn(Optional.of(user));

        return post(UPLOAD_URL)
                .with(SecurityMockMvcRequestPostProcessors.jwt()
                        .jwt(jwt -> jwt.subject(sub))
                        .authorities(new SimpleGrantedAuthority("ROLE_" + role.name())))
                .contentType(MediaType.APPLICATION_JSON)
                .content(body);
    }

    private String requestJson(FileUploadRequest... files) {
        return jsonMapper.writeValueAsString(new PresignedUploadRequest(List.of(files)));
    }

    @Test
    void authorisedAdminCanRequestPresignedUrls() throws Exception {
        String body = requestJson(new FileUploadRequest("call-001.mp3", "audio/mpeg", 1024L));

        mockMvc.perform(authorizedRequest("sub-admin", Role.ADMIN, body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.uploads.length()").value(1))
                .andExpect(jsonPath("$.uploads[0].fileName").value("call-001.mp3"))
                .andExpect(jsonPath("$.uploads[0].objectKey").exists())
                .andExpect(jsonPath("$.uploads[0].presignedUrl").exists());
    }

    @Test
    void unauthenticatedRequestIsRejected() throws Exception {
        String body = requestJson(new FileUploadRequest("call-001.mp3", "audio/mpeg", 1024L));

        mockMvc.perform(post(UPLOAD_URL).contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void managerRoleIsForbidden() throws Exception {
        String body = requestJson(new FileUploadRequest("call-001.mp3", "audio/mpeg", 1024L));

        mockMvc.perform(authorizedRequest("sub-manager", Role.MANAGER, body))
                .andExpect(status().isForbidden());
    }

    @Test
    void invalidContentTypeIsRejected() throws Exception {
        String body = requestJson(new FileUploadRequest("malware.exe", "application/octet-stream", 1024L));

        mockMvc.perform(authorizedRequest("sub-compliance", Role.COMPLIANCE_OFFICER, body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void invalidExtensionIsRejected() throws Exception {
        String body = requestJson(new FileUploadRequest("call-001.exe", "audio/mpeg", 1024L));

        mockMvc.perform(authorizedRequest("sub-compliance", Role.COMPLIANCE_OFFICER, body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void oversizedFileIsRejected() throws Exception {
        String body = requestJson(new FileUploadRequest("call-001.mp3", "audio/mpeg", 1_000_000_000L));

        mockMvc.perform(authorizedRequest("sub-compliance", Role.COMPLIANCE_OFFICER, body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void multipleValidFilesReturnMultipleUrls() throws Exception {
        String body = requestJson(
                new FileUploadRequest("call-001.mp3", "audio/mpeg", 1024L),
                new FileUploadRequest("call-002.wav", "audio/wav", 2048L)
        );

        mockMvc.perform(authorizedRequest("sub-compliance", Role.COMPLIANCE_OFFICER, body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.uploads.length()").value(2));
    }
}
