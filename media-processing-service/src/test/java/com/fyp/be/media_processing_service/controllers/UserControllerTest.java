package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import com.fyp.be.media_processing_service.services.CognitoAdminService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class UserControllerTest {

    private static final String CREATE_USER_URL = "/api/users";

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private UserRepository userRepository;

    @MockitoBean
    private CognitoAdminService cognitoAdminService;

    private static final String REQUEST_JSON = """
            {"email":"officer@example.com","name":"Jane Doe","role":"COMPLIANCE_OFFICER"}""";

    @Test
    void adminCanCreateAComplianceOfficer() throws Exception {
        when(cognitoAdminService.createUser(eq("officer@example.com"), eq("Jane Doe")))
                .thenReturn("new-cognito-sub");
        when(userRepository.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        mockMvc.perform(post(CREATE_USER_URL)
                        .with(SecurityMockMvcRequestPostProcessors.jwt()
                                .jwt(jwt -> jwt.subject("admin-sub"))
                                .authorities(new SimpleGrantedAuthority("ROLE_ADMIN")))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(REQUEST_JSON))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.email").value("officer@example.com"))
                .andExpect(jsonPath("$.role").value("COMPLIANCE_OFFICER"));
    }

    @Test
    void nonAdminCannotCreateUsers() throws Exception {
        mockMvc.perform(post(CREATE_USER_URL)
                        .with(SecurityMockMvcRequestPostProcessors.jwt()
                                .jwt(jwt -> jwt.subject("officer-sub"))
                                .authorities(new SimpleGrantedAuthority("ROLE_COMPLIANCE_OFFICER")))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(REQUEST_JSON))
                .andExpect(status().isForbidden());
    }

    @Test
    void unauthenticatedRequestIsRejected() throws Exception {
        mockMvc.perform(post(CREATE_USER_URL)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(REQUEST_JSON))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void firstAdminCanBeCreatedWithoutAuthWhenNoUsersExist() throws Exception {
        when(userRepository.count()).thenReturn(0L);
        when(cognitoAdminService.createUser(eq("owner@example.com"), eq("Owner")))
                .thenReturn("owner-cognito-sub");
        when(userRepository.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        mockMvc.perform(post("/api/users/first-admin")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"email":"owner@example.com","name":"Owner"}"""))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.role").value("ADMIN"));
    }

    @Test
    void firstAdminEndpointRefusesOnceAUserAlreadyExists() throws Exception {
        when(userRepository.count()).thenReturn(1L);

        mockMvc.perform(post("/api/users/first-admin")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {"email":"owner@example.com","name":"Owner"}"""))
                .andExpect(status().isForbidden());
    }
}
