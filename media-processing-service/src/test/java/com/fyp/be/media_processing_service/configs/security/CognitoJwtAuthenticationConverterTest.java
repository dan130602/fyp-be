package com.fyp.be.media_processing_service.configs.security;

import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.services.UserProvisioningService;
import org.junit.jupiter.api.Test;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.oauth2.jwt.Jwt;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class CognitoJwtAuthenticationConverterTest {
    private final UserProvisioningService users = mock(UserProvisioningService.class);
    private final CognitoJwtAuthenticationConverter converter = new CognitoJwtAuthenticationConverter(users);

    private Jwt token(String group) {
        return Jwt.withTokenValue("test-token").header("alg", "RS256").subject("test-sub")
                .claim("email", "test@example.com").claim("name", "Test")
                .claim("cognito:username", "test-user")
                .claim("cognito:groups", List.of(group)).build();
    }

    @Test void stored_developer_role_produces_only_developer_authority() {
        when(users.resolveOrProvision("test-sub", "test@example.com", "Test", "test-user"))
                .thenReturn(User.builder().role(Role.DEVELOPER).build());
        var authentication = converter.convert(token("SUPER_ADMIN"));
        assertThat(authentication.getAuthorities()).extracting(GrantedAuthority::getAuthority)
                .containsExactly("ROLE_DEVELOPER");
    }

    @Test void cognito_group_alone_cannot_grant_the_database_developer_role() {
        when(users.resolveOrProvision("test-sub", "test@example.com", "Test", "test-user"))
                .thenReturn(User.builder().role(Role.COMPLIANCE_OFFICER).build());
        var authentication = converter.convert(token("DEVELOPER"));
        assertThat(authentication.getAuthorities()).extracting(GrantedAuthority::getAuthority)
                .containsExactly("ROLE_COMPLIANCE_OFFICER");
    }
}
