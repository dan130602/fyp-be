package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.CreateUserRequest;
import com.fyp.be.media_processing_service.dtos.FirstAdminRequest;
import com.fyp.be.media_processing_service.dtos.UpdateUserRoleDto;
import com.fyp.be.media_processing_service.dtos.UpdateUserStatusDto;
import com.fyp.be.media_processing_service.dtos.UserResponseDto;
import com.fyp.be.media_processing_service.services.CurrentUserService;
import com.fyp.be.media_processing_service.services.UserService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/users")
public class UserController {

    private final CurrentUserService currentUserService;
    private final UserService userService;

    public UserController(CurrentUserService currentUserService, UserService userService) {
        this.currentUserService = currentUserService;
        this.userService = userService;
    }

    @GetMapping("/me")
    public ResponseEntity<UserResponseDto> me(Authentication authentication) {
        return ResponseEntity.ok(userService.toResponseDto(currentUserService.getCurrentUser(authentication)));
    }

    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<UserResponseDto>> listUsers() {
        return ResponseEntity.ok(userService.listUsers());
    }

    // Deliberately unauthenticated - see SecurityConfig's permitAll for this path - but
    // UserService#createFirstAdmin refuses (403) the instant any user row exists, so this
    // is only ever a live door during a genuinely fresh deployment, never afterward.
    @PostMapping("/first-admin")
    public ResponseEntity<UserResponseDto> createFirstAdmin(@Valid @RequestBody FirstAdminRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(userService.createFirstAdmin(request.email(), request.name()));
    }

    @PostMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<UserResponseDto> createUser(@Valid @RequestBody CreateUserRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(userService.createUser(request));
    }

    @PatchMapping("/{userId}/role")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<UserResponseDto> updateRole(
            @PathVariable UUID userId,
            @Valid @RequestBody UpdateUserRoleDto request,
            Authentication authentication
    ) {
        return ResponseEntity.ok(userService.updateRole(userId, request.role(), currentUserService.getCurrentUser(authentication)));
    }

    @PatchMapping("/{userId}/status")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<UserResponseDto> updateStatus(
            @PathVariable UUID userId,
            @Valid @RequestBody UpdateUserStatusDto request,
            Authentication authentication
    ) {
        return ResponseEntity.ok(userService.updateStatus(userId, request.status(), currentUserService.getCurrentUser(authentication)));
    }
}
