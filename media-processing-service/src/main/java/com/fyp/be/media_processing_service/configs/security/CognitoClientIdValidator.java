package com.fyp.be.media_processing_service.configs.security;

import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.OAuth2TokenValidator;
import org.springframework.security.oauth2.core.OAuth2TokenValidatorResult;
import org.springframework.security.oauth2.jwt.Jwt;

/**
 * Cognito access tokens carry the app client id in "client_id", not "aud" -
 * only ID tokens use "aud". Accept either token type but always pin it to
 * our configured app client so a token issued for a different client is rejected.
 */
public class CognitoClientIdValidator implements OAuth2TokenValidator<Jwt> {

    private static final OAuth2Error INVALID_CLIENT = new OAuth2Error(
            "invalid_token", "The token was not issued for this application client", null);

    private final String expectedClientId;

    public CognitoClientIdValidator(String expectedClientId) {
        this.expectedClientId = expectedClientId;
    }

    @Override
    public OAuth2TokenValidatorResult validate(Jwt jwt) {
        String tokenUse = jwt.getClaimAsString("token_use");
        boolean matches;
        if ("access".equals(tokenUse)) {
            matches = expectedClientId.equals(jwt.getClaimAsString("client_id"));
        } else if ("id".equals(tokenUse)) {
            matches = jwt.getAudience() != null && jwt.getAudience().contains(expectedClientId);
        } else {
            matches = false;
        }
        return matches ? OAuth2TokenValidatorResult.success() : OAuth2TokenValidatorResult.failure(INVALID_CLIENT);
    }
}
