package com.fyp.be.media_processing_service.configs;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class OpenApiConfigTest {

    @Test void has_any_role_lists_every_role_in_order() {
        assertThat(OpenApiConfig.parseRoles("hasAnyRole('SUPER_ADMIN', 'ADMIN', 'COMPLIANCE_OFFICER')"))
                .containsExactly("SUPER_ADMIN", "ADMIN", "COMPLIANCE_OFFICER");
    }

    @Test void has_role_yields_a_single_role() {
        assertThat(OpenApiConfig.parseRoles("hasRole('MANAGER')")).containsExactly("MANAGER");
    }

    @Test void missing_or_unrelated_expression_yields_no_roles() {
        assertThat(OpenApiConfig.parseRoles(null)).isEqualTo(List.of());
        assertThat(OpenApiConfig.parseRoles("isAuthenticated()")).isEmpty();
    }
}
