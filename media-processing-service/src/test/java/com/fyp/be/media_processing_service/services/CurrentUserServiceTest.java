package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.security.authentication.TestingAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.jwt.Jwt;

import java.time.Instant;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CurrentUserServiceTest {

    private final UserRepository userRepository = mock(UserRepository.class);
    private final CurrentUserService service = new CurrentUserService(userRepository);

    private static Authentication jwtAuth(String sub) {
        Jwt jwt = Jwt.withTokenValue("token").header("alg", "none").subject(sub)
                .issuedAt(Instant.now()).expiresAt(Instant.now().plusSeconds(60)).build();
        return new TestingAuthenticationToken(jwt, null);
    }

    @Test
    void getCurrentUser_knownSub_returnsUser() {
        User user = User.builder().cognitoSub("sub-1").build();
        when(userRepository.findByCognitoSub("sub-1")).thenReturn(Optional.of(user));

        assertThat(service.getCurrentUser(jwtAuth("sub-1"))).isSameAs(user);
    }

    @Test
    void getCurrentUser_unknownSub_throwsNotFound() {
        when(userRepository.findByCognitoSub("sub-2")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.getCurrentUser(jwtAuth("sub-2")))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void getCurrentUser_nullAuthentication_throwsNotFound() {
        assertThatThrownBy(() -> service.getCurrentUser(null))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void getCurrentUser_nonJwtPrincipal_throwsNotFound() {
        assertThatThrownBy(() -> service.getCurrentUser(new TestingAuthenticationToken("someone", null)))
                .isInstanceOf(ResourceNotFoundException.class);
    }
}
