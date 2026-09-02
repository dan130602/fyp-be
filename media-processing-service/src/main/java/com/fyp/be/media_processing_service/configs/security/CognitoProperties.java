package com.fyp.be.media_processing_service.configs.security;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.aws.cognito")
public record CognitoProperties(
        String region,
        String userPoolId,
        String appClientId,
        String issuerUri
) {}
