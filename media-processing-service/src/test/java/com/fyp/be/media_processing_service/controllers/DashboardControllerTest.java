package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.services.DashboardService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
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

import static org.mockito.Mockito.*;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.user;
import static org.springframework.security.test.web.servlet.setup.SecurityMockMvcConfigurers.springSecurity;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** Who may read the org-wide dashboard figures. Isolated context: no database or cloud clients. */
@ExtendWith(SpringExtension.class) @WebAppConfiguration
@ContextConfiguration(classes = DashboardControllerTest.Config.class)
class DashboardControllerTest {
    @Autowired WebApplicationContext context;
    @Autowired DashboardService dashboard;
    MockMvc mvc;

    @BeforeEach void setup() {
        reset(dashboard);
        mvc = MockMvcBuilders.webAppContextSetup(context).apply(springSecurity()).build();
    }

    @ParameterizedTest
    @ValueSource(strings = {"SUPER_ADMIN", "ADMIN", "MANAGER", "COMPLIANCE_OFFICER"})
    void dashboard_roles_can_read_figures(String role) throws Exception {
        mvc.perform(get("/api/dashboard/compliance-overview").with(user("u").roles(role)))
                .andExpect(status().isOk());
        verify(dashboard).getComplianceOverview(null, null, null);
    }

    @Test void developer_cannot_read_figures() throws Exception {
        mvc.perform(get("/api/dashboard/compliance-overview").with(user("dev").roles("DEVELOPER")))
                .andExpect(status().isForbidden());
        verifyNoInteractions(dashboard);
    }

    @Test void anonymous_cannot_read_figures() throws Exception {
        mvc.perform(get("/api/dashboard/compliance-overview")).andExpect(status().isUnauthorized());
        verifyNoInteractions(dashboard);
    }

    @Configuration @EnableWebMvc @EnableWebSecurity @EnableMethodSecurity
    static class Config {
        @Bean DashboardService dashboard() { return mock(DashboardService.class); }
        @Bean DashboardController controller(DashboardService dashboard) { return new DashboardController(dashboard); }
        @Bean SecurityFilterChain security(HttpSecurity http) throws Exception {
            return http.authorizeHttpRequests(r -> r.anyRequest().authenticated())
                    .httpBasic(b -> {}).build();
        }
    }
}
