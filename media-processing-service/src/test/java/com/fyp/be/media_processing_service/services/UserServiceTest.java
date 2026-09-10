package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.dtos.CreateUserRequest;
import com.fyp.be.media_processing_service.dtos.UserResponseDto;
import com.fyp.be.media_processing_service.exceptions.SelfModificationException;
import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private CognitoAdminService cognitoAdminService;

    private User user(UUID id, Role role, UserStatus status, String email) {
        return User.builder()
                .id(id)
                .cognitoSub("sub-" + id)
                .email(email)
                .name("Name " + id)
                .role(role)
                .status(status)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
    }

    @Test
    void listUsersReturnsRowsMappedInRepositoryOrder() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        User newer = user(UUID.randomUUID(), Role.ADMIN, UserStatus.ACTIVE, "newer@example.com");
        User older = user(UUID.randomUUID(), Role.COMPLIANCE_OFFICER, UserStatus.ACTIVE, "older@example.com");
        when(userRepository.findAllByOrderByCreatedAtDesc()).thenReturn(List.of(newer, older));

        List<UserResponseDto> result = service.listUsers();

        assertThat(result).extracting(UserResponseDto::id)
                .containsExactly(newer.getId(), older.getId());
    }

    @Test
    void createUserPassesTheRequestedRoleStraightThrough() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        CreateUserRequest request = new CreateUserRequest("new@example.com", "New User", Role.MANAGER);
        when(cognitoAdminService.createUser("new@example.com", "New User", Role.MANAGER)).thenReturn("new-sub");
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));

        UserResponseDto result = service.createUser(request);

        assertThat(result.role()).isEqualTo(Role.MANAGER);
    }

    @Test
    void selfRoleChangeIsRefused() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        UUID id = UUID.randomUUID();
        User self = user(id, Role.ADMIN, UserStatus.ACTIVE, "self@example.com");
        when(userRepository.findById(id)).thenReturn(Optional.of(self));

        assertThatThrownBy(() -> service.updateRole(id, Role.COMPLIANCE_OFFICER, self))
                .isInstanceOf(SelfModificationException.class);
        verify(userRepository, never()).save(any(User.class));
    }

    @Test
    void selfStatusChangeIsRefused() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        UUID id = UUID.randomUUID();
        User self = user(id, Role.ADMIN, UserStatus.ACTIVE, "self@example.com");
        when(userRepository.findById(id)).thenReturn(Optional.of(self));

        assertThatThrownBy(() -> service.updateStatus(id, UserStatus.DISABLED, self))
                .isInstanceOf(SelfModificationException.class);
        verify(userRepository, never()).save(any(User.class));
    }

    @Test
    void disablingTheLastActiveAdminIsRefused() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        User caller = user(UUID.randomUUID(), Role.ADMIN, UserStatus.ACTIVE, "caller@example.com");
        User lastAdmin = user(UUID.randomUUID(), Role.ADMIN, UserStatus.ACTIVE, "last-admin@example.com");
        when(userRepository.findById(lastAdmin.getId())).thenReturn(Optional.of(lastAdmin));
        when(userRepository.countByRoleAndStatus(Role.ADMIN, UserStatus.ACTIVE)).thenReturn(1L);

        assertThatThrownBy(() -> service.updateStatus(lastAdmin.getId(), UserStatus.DISABLED, caller))
                .isInstanceOf(SelfModificationException.class);
        verify(userRepository, never()).save(any(User.class));
        verify(cognitoAdminService, never()).disableUser(any());
    }

    @Test
    void disablingANonLastAdminCallsCognitoBeforeSaving() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        User caller = user(UUID.randomUUID(), Role.ADMIN, UserStatus.ACTIVE, "caller@example.com");
        User target = user(UUID.randomUUID(), Role.ADMIN, UserStatus.ACTIVE, "target@example.com");
        when(userRepository.findById(target.getId())).thenReturn(Optional.of(target));
        when(userRepository.countByRoleAndStatus(Role.ADMIN, UserStatus.ACTIVE)).thenReturn(2L);
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));

        UserResponseDto result = service.updateStatus(target.getId(), UserStatus.DISABLED, caller);

        assertThat(result.status()).isEqualTo(UserStatus.DISABLED);
        InOrder ordered = inOrder(cognitoAdminService, userRepository);
        ordered.verify(cognitoAdminService).disableUser("target@example.com");
        ordered.verify(userRepository).save(target);
    }

    @Test
    void nullEmailUserSkipsTheCognitoCallRatherThanThrowing() {
        UserService service = new UserService(userRepository, cognitoAdminService);
        User caller = user(UUID.randomUUID(), Role.ADMIN, UserStatus.ACTIVE, "caller@example.com");
        User target = user(UUID.randomUUID(), Role.COMPLIANCE_OFFICER, UserStatus.ACTIVE, null);
        when(userRepository.findById(target.getId())).thenReturn(Optional.of(target));
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));

        UserResponseDto result = service.updateStatus(target.getId(), UserStatus.DISABLED, caller);

        assertThat(result.status()).isEqualTo(UserStatus.DISABLED);
        verify(cognitoAdminService, never()).disableUser(any());
        verify(cognitoAdminService, never()).enableUser(any());
    }
}
