package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.services.ProductionReviewReader;
import com.fyp.be.media_processing_service.services.MediaAssetService;
import com.fyp.be.media_processing_service.services.UserService;
import com.fyp.be.media_processing_service.services.CurrentUserService;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.test.context.web.WebAppConfiguration;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.context.WebApplicationContext;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.beans.factory.annotation.Autowired;
import tools.jackson.databind.ObjectMapper;
import static org.mockito.Mockito.*;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.user;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.csrf;
import static org.springframework.security.test.web.servlet.setup.SecurityMockMvcConfigurers.springSecurity;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/** Isolated HTTP/security context: no application configuration, database or cloud clients. */
@ExtendWith(SpringExtension.class) @WebAppConfiguration
@ContextConfiguration(classes = ObservabilityControllerTest.Config.class)
class ObservabilityControllerTest {
    @Autowired WebApplicationContext context;
    @Autowired ProductionReviewReader reader;
    @Autowired MediaAssetService media;
    @Autowired UserService users;
    @Autowired CurrentUserService currentUser;
    MockMvc mvc;
    @BeforeEach void setup() {
        reset(reader, media, users, currentUser);
        when(reader.read(anyInt(), any())).thenReturn(new ProductionReviewReader.Snapshot(List.of(), List.of(), List.of()));
        mvc = MockMvcBuilders.webAppContextSetup(context).apply(springSecurity()).build();
    }
    @Test void anonymous_cannot_read_production_metrics() throws Exception {
        mvc.perform(get("/api/observability")).andExpect(status().isUnauthorized());
        verifyNoInteractions(reader);
    }
    @Test void officer_cannot_read_administrator_metrics() throws Exception {
        mvc.perform(get("/api/observability").with(user("officer").roles("COMPLIANCE_OFFICER")))
                .andExpect(status().isForbidden());
        verifyNoInteractions(reader);
    }
    @Test void administrator_gets_honest_empty_rates() throws Exception {
        mvc.perform(get("/api/observability").with(user("admin").roles("ADMIN")))
                .andExpect(status().isOk()).andExpect(jsonPath("$.counts.calls").value(0))
                .andExpect(jsonPath("$.metrics.reviewedBreachRecall.percentage").isEmpty())
                .andExpect(jsonPath("$.source").value("production"));
    }
    @Test void rejects_unbounded_query_windows() throws Exception {
        mvc.perform(get("/api/observability?days=365").with(user("admin").roles("ADMIN")))
                .andExpect(status().isBadRequest());
        verifyNoInteractions(reader);
    }
    @Test void backend_super_admin_role_can_read_metrics() throws Exception {
        mvc.perform(get("/api/observability").with(user("super-admin").roles("SUPER_ADMIN")))
                .andExpect(status().isOk());
    }
    @Test void compliance_manager_can_read_aggregate_reporting() throws Exception {
        mvc.perform(get("/api/observability").with(user("manager").roles("MANAGER")))
                .andExpect(status().isOk());
    }
    @Test void developer_can_read_production_metrics() throws Exception {
        mvc.perform(get("/api/observability").with(user("developer").roles("DEVELOPER")))
                .andExpect(status().isOk()).andExpect(jsonPath("$.source").value("production"));
        verify(reader).read(eq(7), any());
    }
    @Test void developer_cannot_submit_an_officer_review() throws Exception {
        mvc.perform(put("/api/media-assets/00000000-0000-0000-0000-000000000001/review")
                .with(user("developer").roles("DEVELOPER")).with(csrf())
                .contentType("application/json").content("{\"overrides\":[]}"))
                .andExpect(status().isForbidden());
        verifyNoInteractions(media, currentUser, reader);
    }
    @Test void developer_cannot_change_account_roles() throws Exception {
        mvc.perform(patch("/api/users/00000000-0000-0000-0000-000000000001/role")
                .with(user("developer").roles("DEVELOPER")).with(csrf())
                .contentType("application/json").content("{\"role\":\"ADMIN\"}"))
                .andExpect(status().isForbidden());
        verifyNoInteractions(users, currentUser, reader);
    }
    @Configuration @EnableWebMvc @EnableWebSecurity @EnableMethodSecurity
    static class Config {
        @Bean ProductionReviewReader reader() { return mock(ProductionReviewReader.class); }
        @Bean ObservabilityController controller(ProductionReviewReader reader) {
            return new ObservabilityController(reader, new ObjectMapper());
        }
        @Bean MediaAssetService media() { return mock(MediaAssetService.class); }
        @Bean UserService users() { return mock(UserService.class); }
        @Bean CurrentUserService currentUser() { return mock(CurrentUserService.class); }
        @Bean MediaAssetController mediaController(MediaAssetService media, CurrentUserService currentUser) {
            return new MediaAssetController(media, currentUser);
        }
        @Bean UserController userController(CurrentUserService currentUser, UserService users) {
            return new UserController(currentUser, users);
        }
        @Bean SecurityFilterChain security(HttpSecurity http) throws Exception {
            return http.authorizeHttpRequests(r -> r.anyRequest().authenticated())
                    .httpBasic(b -> {}).build();
        }
    }
}
