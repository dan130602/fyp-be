package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.exceptions.CognitoAdminException;
import com.fyp.be.media_processing_service.exceptions.UserAlreadyExistsException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.cognitoidentityprovider.CognitoIdentityProviderClient;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminCreateUserRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminCreateUserResponse;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminGetUserRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminGetUserResponse;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminSetUserPasswordRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AttributeType;
import software.amazon.awssdk.services.cognitoidentityprovider.model.MessageActionType;
import software.amazon.awssdk.services.cognitoidentityprovider.model.UsernameExistsException;

import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * All Cognito admin-API calls (using our own AWS credentials, not any caller's token):
 * creating logins for admin-provisioned users, and looking up attributes for a caller
 * whose access token is missing them (access tokens carry no "email"/"name" claim).
 */
@Slf4j
@Service
public class CognitoAdminService {

    private final CognitoIdentityProviderClient cognitoClient;
    private final String userPoolId;
    private final String defaultUserPassword;

    public CognitoAdminService(
            CognitoIdentityProviderClient cognitoClient,
            @Value("${app.aws.cognito.user-pool-id}") String userPoolId,
            @Value("${app.security.default-user-password}") String defaultUserPassword
    ) {
        this.cognitoClient = cognitoClient;
        this.userPoolId = userPoolId;
        this.defaultUserPassword = defaultUserPassword;
    }

    /**
     * Returns the new user's Cognito sub. Sets defaultUserPassword as a permanent password
     * immediately (school-project convenience: log in right away, no invite-email/temp-password
     * dance) - suppresses the invite email since it's no longer needed for that.
     */
    public String createUser(String email, String name) {
        String username = (name != null && !name.isBlank()) ? name : UUID.randomUUID().toString();

        AdminCreateUserRequest createRequest = AdminCreateUserRequest.builder()
                .userPoolId(userPoolId)
                .username(username)
                .userAttributes(
                        AttributeType.builder().name("email").value(email).build(),
                        AttributeType.builder().name("email_verified").value("true").build(),
                        AttributeType.builder().name("name").value(name == null ? email : name).build()
                )
                .messageAction(MessageActionType.SUPPRESS)
                .build();

        try {
            AdminCreateUserResponse response = cognitoClient.adminCreateUser(createRequest);
            String sub = extractSub(response);

            cognitoClient.adminSetUserPassword(AdminSetUserPasswordRequest.builder()
                    .userPoolId(userPoolId)
                    .username(username)
                    .password(defaultUserPassword)
                    .permanent(true)
                    .build());

            return sub;
        } catch (UsernameExistsException ex) {
            throw new UserAlreadyExistsException("A Cognito user with email '%s' already exists".formatted(email));
        } catch (Exception ex) {
            throw new CognitoAdminException("Failed to create Cognito user for email '%s'".formatted(email), ex);
        }
    }

    private String extractSub(AdminCreateUserResponse response) {
        return response.user().attributes().stream()
                .filter(attribute -> "sub".equals(attribute.name()))
                .map(AttributeType::value)
                .findFirst()
                .orElseThrow(() -> new CognitoAdminException("Cognito did not return a sub for the new user", null));
    }

    /**
     * Returns the user's Cognito standard attributes (e.g. "email", "name"), or empty on
     * any failure. Must be looked up by Cognito's own username (the token's "username"
     * claim), NOT the sub - AdminGetUser's Username parameter does not accept the sub here.
     */
    public Map<String, String> fetchAttributes(String cognitoUsername) {
        try {
            AdminGetUserResponse response = cognitoClient.adminGetUser(AdminGetUserRequest.builder()
                    .userPoolId(userPoolId)
                    .username(cognitoUsername)
                    .build());
            return response.userAttributes().stream()
                    .collect(Collectors.toMap(AttributeType::name, AttributeType::value, (a, b) -> a));
        } catch (Exception ex) {
            log.warn("Failed to fetch Cognito attributes via AdminGetUser for username={}", cognitoUsername, ex);
            return Map.of();
        }
    }
}
