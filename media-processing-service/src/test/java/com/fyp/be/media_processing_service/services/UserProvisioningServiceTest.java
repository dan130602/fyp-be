package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.authentication.DisabledException;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserProvisioningServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private CognitoAdminService cognitoAdminService;

    @Test
    void newCognitoSubIsProvisionedWithDefaultNonElevatedRole() {
        UserProvisioningService service = new UserProvisioningService(userRepository, cognitoAdminService, List.of());
        when(userRepository.findByCognitoSub("new-sub")).thenReturn(Optional.empty());
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));

        User result = service.resolveOrProvision("new-sub", "new@example.com", "New User", "newuser");

        ArgumentCaptor<User> captor = ArgumentCaptor.forClass(User.class);
        verify(userRepository).save(captor.capture());
        User saved = captor.getValue();

        assertThat(saved.getCognitoSub()).isEqualTo("new-sub");
        assertThat(saved.getRole()).isEqualTo(Role.COMPLIANCE_OFFICER);
        assertThat(saved.getRole()).isNotEqualTo(Role.ADMIN);
        assertThat(saved.getStatus()).isEqualTo(UserStatus.ACTIVE);
        assertThat(result.getCognitoSub()).isEqualTo("new-sub");
    }

    @Test
    void existingCognitoSubMapsToTheSameInternalUserWithoutReprovisioning() {
        UserProvisioningService service = new UserProvisioningService(userRepository, cognitoAdminService, List.of());
        User existing = User.builder()
                .id(UUID.randomUUID())
                .cognitoSub("existing-sub")
                .email("existing@example.com")
                .role(Role.ADMIN)
                .status(UserStatus.ACTIVE)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        when(userRepository.findByCognitoSub("existing-sub")).thenReturn(Optional.of(existing));

        User result = service.resolveOrProvision("existing-sub", "existing@example.com", "Existing User", "existinguser");

        assertThat(result).isSameAs(existing);
        verify(userRepository, never()).save(any(User.class));
    }

    @Test
    void bootstrapAdminEmailGetsAdminOnFirstProvisionOnly() {
        UserProvisioningService service = new UserProvisioningService(userRepository, cognitoAdminService, List.of("Boss@Example.com"));
        when(userRepository.findByCognitoSub("boss-sub")).thenReturn(Optional.empty());
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));

        User result = service.resolveOrProvision("boss-sub", "boss@example.com", "Boss", "boss");

        assertThat(result.getRole()).isEqualTo(Role.SUPER_ADMIN);
    }

    @Test
    void emailNotOnBootstrapListStillGetsDefaultRole() {
        UserProvisioningService service = new UserProvisioningService(userRepository, cognitoAdminService, List.of("boss@example.com"));
        when(userRepository.findByCognitoSub("other-sub")).thenReturn(Optional.empty());
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));

        User result = service.resolveOrProvision("other-sub", "someone-else@example.com", "Someone Else", "someoneelse");

        assertThat(result.getRole()).isEqualTo(Role.COMPLIANCE_OFFICER);
    }

    @Test
    void missingEmailClaimFallsBackToCognitoAdminLookupByUsernameAndStillMatchesBootstrapList() {
        UserProvisioningService service = new UserProvisioningService(userRepository, cognitoAdminService, List.of("boss@example.com"));
        when(userRepository.findByCognitoSub("access-token-sub")).thenReturn(Optional.empty());
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));
        when(cognitoAdminService.fetchAttributes("boss-username"))
                .thenReturn(Map.of("email", "boss@example.com", "name", "Boss"));

        User result = service.resolveOrProvision("access-token-sub", null, null, "boss-username");

        assertThat(result.getEmail()).isEqualTo("boss@example.com");
        assertThat(result.getRole()).isEqualTo(Role.SUPER_ADMIN);
    }

    @Test
    void disabledUserIsRejected() {
        UserProvisioningService service = new UserProvisioningService(userRepository, cognitoAdminService, List.of());
        User disabled = User.builder()
                .id(UUID.randomUUID())
                .cognitoSub("disabled-sub")
                .role(Role.COMPLIANCE_OFFICER)
                .status(UserStatus.DISABLED)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        when(userRepository.findByCognitoSub("disabled-sub")).thenReturn(Optional.of(disabled));

        assertThatThrownBy(() -> service.resolveOrProvision("disabled-sub", "d@example.com", "Disabled User", "disableduser"))
                .isInstanceOf(DisabledException.class);
    }
}
