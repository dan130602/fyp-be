package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.exceptions.ResourceNotFoundException;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.repositories.UserRepository;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.stereotype.Service;

@Service
public class CurrentUserService {

    private final UserRepository userRepository;

    public CurrentUserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User getCurrentUser(Authentication authentication) {
        String cognitoSub = extractSub(authentication);
        return userRepository.findByCognitoSub(cognitoSub)
                .orElseThrow(() -> new ResourceNotFoundException("No internal user found for the authenticated identity"));
    }

    private String extractSub(Authentication authentication) {
        if (authentication == null || !(authentication.getPrincipal() instanceof Jwt jwt)) {
            throw new ResourceNotFoundException("No authenticated Cognito identity present on this request");
        }
        return jwt.getSubject();
    }
}
