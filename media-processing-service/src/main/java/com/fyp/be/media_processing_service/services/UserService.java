package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.CreateUserRequest;
import com.fyp.be.media_processing_service.dtos.UserResponseDto;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.exceptions.SelfModificationException;
import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Set;
import java.util.UUID;

@Service
@Slf4j
public class UserService {

    /** Roles only a SUPER_ADMIN may grant - via createUser or updateRole. */
    private static final Set<Role> ELEVATED_ROLES = Set.of(Role.ADMIN, Role.SUPER_ADMIN);

    private final UserRepository userRepository;
    private final CognitoAdminService cognitoAdminService;

    public UserService(UserRepository userRepository, CognitoAdminService cognitoAdminService) {
        this.userRepository = userRepository;
        this.cognitoAdminService = cognitoAdminService;
    }

    public UserResponseDto toResponseDto(User user) {
        return new UserResponseDto(user.getId(), user.getEmail(), user.getName(), user.getRole(), user.getStatus(), user.getCreatedAt());
    }

    public List<UserResponseDto> listUsers() {
        return userRepository.findAllByOrderByCreatedAtDesc().stream()
                .map(this::toResponseDto)
                .toList();
    }

    public UserResponseDto updateRole(UUID userId, Role newRole, User currentUser) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User not found: " + userId));

        if (user.getId().equals(currentUser.getId())) {
            throw new SelfModificationException("You cannot change your own role");
        }
        requireSuperAdminToGrant(newRole, currentUser);
        if (newRole != user.getRole() && isLastActiveInProtectedRole(user)) {
            throw new SelfModificationException(
                    "Cannot change the role of the last remaining active " + user.getRole());
        }

        user.setRole(newRole);
        user.setUpdatedAt(LocalDateTime.now());
        User saved = userRepository.save(user);
        cognitoAdminService.setUserGroup(user.getEmail(), newRole);
        return toResponseDto(saved);
    }

    public UserResponseDto updateStatus(UUID userId, UserStatus newStatus, User currentUser) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User not found: " + userId));

        if (user.getId().equals(currentUser.getId())) {
            throw new SelfModificationException("You cannot change your own status");
        }
        if (newStatus == UserStatus.DISABLED && isLastActiveInProtectedRole(user)) {
            throw new SelfModificationException(
                    "Cannot disable the last remaining active " + user.getRole());
        }

        // Cognito first: for a security-relevant operation, failing before the DB write is the safer order.
        if (user.getEmail() == null) {
            log.warn("No email on file for user={}; skipping Cognito enable/disable call", user.getId());
        } else if (newStatus == UserStatus.DISABLED) {
            cognitoAdminService.disableUser(user.getEmail());
        } else {
            cognitoAdminService.enableUser(user.getEmail());
        }

        user.setStatus(newStatus);
        user.setUpdatedAt(LocalDateTime.now());
        return toResponseDto(userRepository.save(user));
    }

    /** Covers ADMIN and SUPER_ADMIN - losing the last active holder of either would leave
     * no one able to ever grant that role again (see requireSuperAdminToGrant). */
    private boolean isLastActiveInProtectedRole(User user) {
        Role role = user.getRole();
        return ELEVATED_ROLES.contains(role)
                && user.getStatus() == UserStatus.ACTIVE
                && userRepository.countByRoleAndStatus(role, UserStatus.ACTIVE) <= 1;
    }

    /** ADMIN can grant COMPLIANCE_OFFICER/MANAGER freely, but only a SUPER_ADMIN can grant
     * ADMIN or SUPER_ADMIN itself - otherwise any ADMIN could mint themselves a peer. */
    private void requireSuperAdminToGrant(Role targetRole, User currentUser) {
        if (ELEVATED_ROLES.contains(targetRole) && currentUser.getRole() != Role.SUPER_ADMIN) {
            throw new AccessDeniedException("Only SUPER_ADMIN can grant the '%s' role".formatted(targetRole));
        }
    }

    /** Creates both the Cognito login and the internal user record - only ever called from an
     * ADMIN/SUPER_ADMIN-only endpoint; granting ADMIN or SUPER_ADMIN itself requires the caller
     * to already be SUPER_ADMIN (see requireSuperAdminToGrant). */
    public UserResponseDto createUser(CreateUserRequest request, User currentUser) {
        requireSuperAdminToGrant(request.role(), currentUser);
        return provisionUser(request);
    }

    /** Unauthenticated on purpose, but only while the system has zero users - see UserController.
     * Grants SUPER_ADMIN (not ADMIN) - since only a SUPER_ADMIN can grant ADMIN/SUPER_ADMIN
     * through the app, granting a plain ADMIN here would leave no one able to ever create another
     * admin afterward. No currentUser to check here (unauthenticated), so this bypasses
     * requireSuperAdminToGrant entirely - the "zero users" check above is the actual gate. */
    public UserResponseDto createFirstAdmin(String email, String name) {
        if (userRepository.count() > 0) {
            throw new AccessDeniedException("Setup already completed");
        }
        return provisionUser(new CreateUserRequest(email, name, Role.SUPER_ADMIN));
    }

    private UserResponseDto provisionUser(CreateUserRequest request) {
        String cognitoSub = cognitoAdminService.createUser(request.email(), request.name(), request.role());

        LocalDateTime now = LocalDateTime.now();
        User user = User.builder()
                .id(UUID.randomUUID())
                .cognitoSub(cognitoSub)
                .email(request.email())
                .name(request.name())
                .role(request.role())
                .status(UserStatus.ACTIVE)
                .createdAt(now)
                .updatedAt(now)
                .build();
        log.info("Creating new user: {}", user);
        return toResponseDto(userRepository.save(user));
    }
}
