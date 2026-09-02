package com.fyp.be.media_processing_service.configs.security;

import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.services.UserProvisioningService;
import org.springframework.core.convert.converter.Converter;
import org.springframework.security.authentication.AbstractAuthenticationToken;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Cognito authenticates the request; authorization is entirely driven by the
 * internal User.role looked up (and auto-provisioned if needed) here - the
 * JWT itself never grants roles, so @PreAuthorize checks reflect our own RBAC.
 */
@Component
public class CognitoJwtAuthenticationConverter implements Converter<Jwt, AbstractAuthenticationToken> {

    private final UserProvisioningService userProvisioningService;

    public CognitoJwtAuthenticationConverter(UserProvisioningService userProvisioningService) {
        this.userProvisioningService = userProvisioningService;
    }

    @Override
    public AbstractAuthenticationToken convert(Jwt jwt) {
        // Access tokens carry "username"; ID tokens carry "cognito:username" instead.
        String cognitoUsername = jwt.getClaimAsString("username");
        if (cognitoUsername == null) {
            cognitoUsername = jwt.getClaimAsString("cognito:username");
        }

        User user = userProvisioningService.resolveOrProvision(
                jwt.getSubject(),
                jwt.getClaimAsString("email"),
                jwt.getClaimAsString("name"),
                cognitoUsername
        );

        List<GrantedAuthority> authorities = List.of(new SimpleGrantedAuthority("ROLE_" + user.getRole().name()));
        return new JwtAuthenticationToken(jwt, authorities, jwt.getSubject());
    }
}
