package com.fyp.be.media_processing_service.repositories;

import com.fyp.be.media_processing_service.models.Role;
import com.fyp.be.media_processing_service.models.User;
import com.fyp.be.media_processing_service.models.UserStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface UserRepository extends JpaRepository<User, UUID> {
    Optional<User> findByCognitoSub(String cognitoSub);

    List<User> findAllByOrderByCreatedAtDesc();

    long countByRoleAndStatus(Role role, UserStatus status);
}
