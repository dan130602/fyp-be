package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.DisabledException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * New Cognito identities are provisioned here with a fixed, non-elevated
 * default role - the role is never taken from the token or any client input,
 * so a caller cannot self-assign an elevated role on first login.
 *
 * The one exception is app.security.bootstrap-admin-emails: an operator-controlled
 * allowlist (env var, never client input) solving the RBAC bootstrap problem - with
 * zero users in the table, nothing in the app itself could ever create the first one.
 * Grants SUPER_ADMIN specifically (not ADMIN) - since only a SUPER_ADMIN can ever grant
 * ADMIN or SUPER_ADMIN through the app (see UserService), this is the only door in.
 * Only applies at first provisioning, only for pre-configured emails.
 */
@Slf4j
@Service
public class UserProvisioningService {

    static final Role DEFAULT_ROLE = Role.COMPLIANCE_OFFICER;

    private final UserRepository userRepository;
    private final CognitoAdminService cognitoAdminService;
    private final Set<String> bootstrapAdminEmails;

    public UserProvisioningService(
            UserRepository userRepository,
            CognitoAdminService cognitoAdminService,
            @Value("${app.security.bootstrap-admin-emails:}") List<String> bootstrapAdminEmails
    ) {
        this.userRepository = userRepository;
        this.cognitoAdminService = cognitoAdminService;
        this.bootstrapAdminEmails = bootstrapAdminEmails.stream()
                .map(email -> email.toLowerCase(Locale.ROOT).trim())
                .filter(email -> !email.isEmpty())
                .collect(Collectors.toUnmodifiableSet());
    }

    /**
     * cognitoUsername (the token's "username" claim, NOT the sub) is only ever used as a
     * fallback AdminGetUser lookup when the token's own "email" claim is missing - true
     * for Cognito access tokens, only ID tokens carry it - and only on first-time
     * provisioning. Never called once the user already exists.
     */
    @Transactional
    public User resolveOrProvision(String cognitoSub, String email, String name, String cognitoUsername) {
        User user = userRepository.findByCognitoSub(cognitoSub)
                .orElseGet(() -> provision(cognitoSub, email, name, cognitoUsername));

        if (user.getStatus() == UserStatus.DISABLED) {
            throw new DisabledException("User account is disabled");
        }
        return user;
    }

    private User provision(String cognitoSub, String email, String name, String cognitoUsername) {
        if (email == null && cognitoUsername != null) {
            Map<String, String> attributes = cognitoAdminService.fetchAttributes(cognitoUsername);
            email = attributes.get("email");
            name = name != null ? name : attributes.get("name");
        }

        Role role = isBootstrapAdmin(email) ? Role.SUPER_ADMIN : DEFAULT_ROLE;
        log.info("Provisioning new internal user for cognitoSub={} with role={}", cognitoSub, role);

        if (email != null) {
            cognitoAdminService.addUserToGroup(email, role);
        } else {
            log.warn("No email resolved for cognitoSub={}; skipping Cognito group assignment", cognitoSub);
        }

        LocalDateTime now = LocalDateTime.now();
        User user = User.builder()
                .id(UUID.randomUUID())
                .cognitoSub(cognitoSub)
                .email(email)
                .name(name)
                .role(role)
                .status(UserStatus.ACTIVE)
                .createdAt(now)
                .updatedAt(now)
                .build();
        return userRepository.save(user);
    }

    private boolean isBootstrapAdmin(String email) {
        return email != null && bootstrapAdminEmails.contains(email.toLowerCase(Locale.ROOT).trim());
    }
}
