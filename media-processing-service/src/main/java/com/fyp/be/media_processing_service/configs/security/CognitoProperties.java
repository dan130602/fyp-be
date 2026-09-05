package com.fyp.be.media_processing_service.configs.security;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.List;

@ConfigurationProperties(prefix = "app.aws.cognito")
public record CognitoProperties(
        String region,
        String userPoolId,
        List<String> appClientId,
        String issuerUri
) {}
