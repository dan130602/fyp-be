package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.annotations.security.SecurityRequirements;
import com.fyp.be.media_processing_service.dtos.CreateUserRequest;
import com.fyp.be.media_processing_service.dtos.FirstAdminRequest;
import com.fyp.be.media_processing_service.dtos.UpdateUserRoleDto;
import com.fyp.be.media_processing_service.dtos.UpdateUserStatusDto;
import com.fyp.be.media_processing_service.dtos.UserResponseDto;
import com.fyp.be.media_processing_service.models.User;
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

@Tag(name = "Users", description = "Service-side user accounts and roles")
@RestController
@RequestMapping("/api/users")
public class UserController {

    private final CurrentUserService currentUserService;
    private final UserService userService;

    public UserController(CurrentUserService currentUserService, UserService userService) {
        this.currentUserService = currentUserService;
        this.userService = userService;
    }

    @Operation(summary = "Get the current user",
            description = "On a caller's first request the user row is created first.")
    @GetMapping("/me")
    public ResponseEntity<UserResponseDto> me(Authentication authentication) {
        return ResponseEntity.ok(userService.toResponseDto(currentUserService.getCurrentUser(authentication)));
    }

    @Operation(summary = "List users", description = "All users, newest first.")
    @GetMapping
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN')")
    public ResponseEntity<List<UserResponseDto>> listUsers() {
        return ResponseEntity.ok(userService.listUsers());
    }

    // Deliberately unauthenticated - see SecurityConfig's permitAll for this path - but
    // UserService#createFirstAdmin refuses (403) the instant any user row exists, so this
    // is only ever a live door during a genuinely fresh deployment, never afterward.
    @Operation(summary = "Create the first admin",
            description = "Public. Creates the first account as SUPER_ADMIN in both Cognito and the database. Refused with 403 as soon as any user exists.")
    @ApiResponse(responseCode = "201", description = "Created")
    @ApiResponse(responseCode = "403", description = "Setup already completed", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "500", description = "Cognito call failed", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @SecurityRequirements
    @PostMapping("/first-admin")
    public ResponseEntity<UserResponseDto> createFirstAdmin(@Valid @RequestBody FirstAdminRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(userService.createFirstAdmin(request.email(), request.name()));
    }

    @Operation(summary = "Create a user",
            description = "Creates a Cognito login and a matching user row. The password is set permanently to DEFAULT_USER_PASSWORD and no invite email is sent. Only SUPER_ADMIN can create ADMIN or SUPER_ADMIN.")
    @ApiResponse(responseCode = "201", description = "Created")
    @ApiResponse(responseCode = "403", description = "Caller may not create that role", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "409", description = "User already exists", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "500", description = "Cognito call failed", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @PostMapping
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN')")
    public ResponseEntity<UserResponseDto> createUser(
            @Valid @RequestBody CreateUserRequest request,
            Authentication authentication
    ) {
        User currentUser = currentUserService.getCurrentUser(authentication);
        return ResponseEntity.status(HttpStatus.CREATED).body(userService.createUser(request, currentUser));
    }

    @Operation(summary = "Change a user's role",
            description = "Also moves the user to the matching Cognito group. Only SUPER_ADMIN can grant ADMIN or SUPER_ADMIN; you cannot change your own role or leave the system without an active ADMIN or SUPER_ADMIN.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "403", description = "Caller may not grant that role", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "404", description = "User not found", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "409", description = "Self-change, or it would leave no active ADMIN or SUPER_ADMIN", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @PatchMapping("/{userId}/role")
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN')")
    public ResponseEntity<UserResponseDto> updateRole(
            @PathVariable UUID userId,
            @Valid @RequestBody UpdateUserRoleDto request,
            Authentication authentication
    ) {
        return ResponseEntity.ok(userService.updateRole(userId, request.role(), currentUserService.getCurrentUser(authentication)));
    }

    @Operation(summary = "Enable or disable a user",
            description = "The Cognito login is updated before the database. You cannot change your own status or disable the last active ADMIN or SUPER_ADMIN.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "404", description = "User not found", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @ApiResponse(responseCode = "409", description = "Self-change, or it would disable the last active ADMIN or SUPER_ADMIN", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @PatchMapping("/{userId}/status")
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN')")
    public ResponseEntity<UserResponseDto> updateStatus(
            @PathVariable UUID userId,
            @Valid @RequestBody UpdateUserStatusDto request,
            Authentication authentication
    ) {
        return ResponseEntity.ok(userService.updateStatus(userId, request.status(), currentUserService.getCurrentUser(authentication)));
    }
}
