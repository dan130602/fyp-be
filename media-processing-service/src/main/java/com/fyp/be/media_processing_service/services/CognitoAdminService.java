package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.exceptions.CognitoAdminException;
import com.fyp.be.media_processing_service.exceptions.UserAlreadyExistsException;
import com.fyp.be.media_processing_service.models.Role;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.cognitoidentityprovider.CognitoIdentityProviderClient;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminAddUserToGroupRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminCreateUserRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminCreateUserResponse;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminDisableUserRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminEnableUserRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminGetUserRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminGetUserResponse;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminRemoveUserFromGroupRequest;
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
     * dance) - suppresses the invite email since it's no longer needed for that. Also puts the
     * user into the Cognito group matching role, so cognito:groups shows up in their tokens.
     */
    public String createUser(String email, String name, Role role) {
        // Opaque username: Cognito requires it unique, and email is an alias so it cannot be used here.
        String username = UUID.randomUUID().toString();

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

            addUserToGroup(username, role);

            return sub;
        } catch (UsernameExistsException ex) {
            throw new UserAlreadyExistsException("A user with email '%s' already exists".formatted(email));
        } catch (Exception ex) {
            throw new CognitoAdminException("Failed to create Cognito user for email '%s'".formatted(email), ex);
        }
    }

    /**
     * Adds a user (identified by their Cognito username, or any configured alias
     * attribute such as email) to the group matching role.
     */
    public void addUserToGroup(String usernameOrAlias, Role role) {
        try {
            cognitoClient.adminAddUserToGroup(AdminAddUserToGroupRequest.builder()
                    .userPoolId(userPoolId)
                    .username(usernameOrAlias)
                    .groupName(role.name())
                    .build());
        } catch (Exception ex) {
            throw new CognitoAdminException(
                    "Failed to add Cognito user '%s' to group '%s'".formatted(usernameOrAlias, role.name()), ex);
        }
    }

    /**
     * Moves a user to the group matching newRole, removing them from every other
     * role group first - a user should only ever be in one role group at a time.
     */
    public void setUserGroup(String usernameOrAlias, Role newRole) {
        try {
            for (Role role : Role.values()) {
                if (role != newRole) {
                    cognitoClient.adminRemoveUserFromGroup(AdminRemoveUserFromGroupRequest.builder()
                            .userPoolId(userPoolId)
                            .username(usernameOrAlias)
                            .groupName(role.name())
                            .build());
                }
            }
        } catch (Exception ex) {
            throw new CognitoAdminException(
                    "Failed to clear stale Cognito role groups for user '%s'".formatted(usernameOrAlias), ex);
        }
        addUserToGroup(usernameOrAlias, newRole);
    }

    /** Disables a user's Cognito login, keyed on email (the alias attribute), so they can no longer get fresh tokens. */
    public void disableUser(String email) {
        try {
            cognitoClient.adminDisableUser(AdminDisableUserRequest.builder()
                    .userPoolId(userPoolId)
                    .username(email)
                    .build());
        } catch (Exception ex) {
            throw new CognitoAdminException("Failed to disable Cognito user '%s'".formatted(email), ex);
        }
    }

    /** Re-enables a user's Cognito login, keyed on email (the alias attribute). */
    public void enableUser(String email) {
        try {
            cognitoClient.adminEnableUser(AdminEnableUserRequest.builder()
                    .userPoolId(userPoolId)
                    .username(email)
                    .build());
        } catch (Exception ex) {
            throw new CognitoAdminException("Failed to enable Cognito user '%s'".formatted(email), ex);
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
