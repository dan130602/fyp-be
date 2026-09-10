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
import java.util.UUID;

@Service
@Slf4j
public class UserService {

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
        if (newRole != Role.ADMIN && isLastActiveAdmin(user)) {
            throw new SelfModificationException("Cannot change the role of the last remaining active admin");
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
        if (newStatus == UserStatus.DISABLED && isLastActiveAdmin(user)) {
            throw new SelfModificationException("Cannot disable the last remaining active admin");
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

    private boolean isLastActiveAdmin(User user) {
        return user.getRole() == Role.ADMIN
                && user.getStatus() == UserStatus.ACTIVE
                && userRepository.countByRoleAndStatus(Role.ADMIN, UserStatus.ACTIVE) <= 1;
    }

    /** Creates both the Cognito login and the internal user record - only ever called from an ADMIN-only endpoint. */
    public UserResponseDto createUser(CreateUserRequest request) {
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

    /** Unauthenticated on purpose, but only while the system has zero users - see UserController. */
    public UserResponseDto createFirstAdmin(String email, String name) {
        if (userRepository.count() > 0) {
            throw new AccessDeniedException("Setup already completed");
        }
        return createUser(new CreateUserRequest(email, name, Role.ADMIN));
    }
}
