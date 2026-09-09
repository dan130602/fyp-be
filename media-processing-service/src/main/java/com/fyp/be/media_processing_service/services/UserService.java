package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.CreateUserRequest;
import com.fyp.be.media_processing_service.dtos.UserResponseDto;
import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
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
        return new UserResponseDto(user.getId(), user.getEmail(), user.getName(), user.getRole(), user.getStatus());
    }

    public UserResponseDto updateRole(UUID userId, Role newRole) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User not found: " + userId));
        user.setRole(newRole);
        user.setUpdatedAt(LocalDateTime.now());
        User saved = userRepository.save(user);
        cognitoAdminService.setUserGroup(user.getEmail(), newRole);
        return toResponseDto(saved);
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
