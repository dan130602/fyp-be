package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.models.Role;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import software.amazon.awssdk.services.cognitoidentityprovider.CognitoIdentityProviderClient;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminAddUserToGroupRequest;
import software.amazon.awssdk.services.cognitoidentityprovider.model.AdminRemoveUserFromGroupRequest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class CognitoAdminServiceTest {
    private final CognitoIdentityProviderClient client = mock(CognitoIdentityProviderClient.class);
    private final CognitoAdminService service = new CognitoAdminService(client, "test-pool", "unused-test-value");

    @Test void developer_is_added_to_the_matching_user_pool_group() {
        service.addUserToGroup("test-user", Role.DEVELOPER);
        var request = ArgumentCaptor.forClass(AdminAddUserToGroupRequest.class);
        verify(client).adminAddUserToGroup(request.capture());
        assertThat(request.getValue().userPoolId()).isEqualTo("test-pool");
        assertThat(request.getValue().username()).isEqualTo("test-user");
        assertThat(request.getValue().groupName()).isEqualTo("DEVELOPER");
    }

    @Test void assigning_developer_removes_previous_role_groups() {
        service.setUserGroup("test-user", Role.DEVELOPER);
        var removed = ArgumentCaptor.forClass(AdminRemoveUserFromGroupRequest.class);
        verify(client, times(4)).adminRemoveUserFromGroup(removed.capture());
        assertThat(removed.getAllValues()).extracting(AdminRemoveUserFromGroupRequest::groupName)
                .containsExactlyInAnyOrder("SUPER_ADMIN", "ADMIN", "COMPLIANCE_OFFICER", "MANAGER");
        verify(client).adminAddUserToGroup(argThat((AdminAddUserToGroupRequest request) ->
                request.groupName().equals("DEVELOPER")));
    }

    @Test void changing_away_from_developer_removes_the_developer_group() {
        service.setUserGroup("test-user", Role.COMPLIANCE_OFFICER);
        verify(client).adminRemoveUserFromGroup(argThat((AdminRemoveUserFromGroupRequest request) ->
                request.groupName().equals("DEVELOPER")));
        verify(client).adminAddUserToGroup(argThat((AdminAddUserToGroupRequest request) ->
                request.groupName().equals("COMPLIANCE_OFFICER")));
    }
}
