package com.fyp.be.media_processing_service.configs;

import com.fyp.be.media_processing_service.dtos.ErrorResponse;
import io.swagger.v3.core.converter.ModelConverters;
import io.swagger.v3.oas.annotations.security.SecurityRequirements;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.Operation;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.media.Content;
import io.swagger.v3.oas.models.media.MediaType;
import io.swagger.v3.oas.models.media.Schema;
import io.swagger.v3.oas.models.responses.ApiResponse;
import io.swagger.v3.oas.models.responses.ApiResponses;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import org.springdoc.core.customizers.OperationCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.security.access.prepost.PreAuthorize;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Configuration
public class OpenApiConfig {

    static final String BEARER_SCHEME = "bearerAuth";

    private static final Pattern ROLE_EXPRESSION = Pattern.compile("has(?:Any)?Role\\s*\\(([^)]*)\\)");
    private static final Pattern QUOTED = Pattern.compile("'([^']+)'");

    @Bean
    public OpenAPI openApi() {
        return new OpenAPI()
                .info(new Info()
                        .title("ClearCall media-processing-service API")
                        .version("0.0.1")
                        .description("""
                                Uploads call recordings, tracks transcription and AI analysis, and records \
                                compliance-officer reviews. Roles come from the service's own user table, \
                                not from the token. Workflow rules, ownership and the dashboard definitions \
                                are written up in docs/API.md."""))
                .components(new Components()
                        .schemas(ModelConverters.getInstance().read(ErrorResponse.class))
                        .addSecuritySchemes(BEARER_SCHEME, new SecurityScheme()
                        .type(SecurityScheme.Type.HTTP)
                        .scheme("bearer")
                        .bearerFormat("JWT")
                        .description("Cognito access or ID token")))
                .addSecurityItem(new SecurityRequirement().addList(BEARER_SCHEME));
    }

    /**
     * Role lists live only in {@code @PreAuthorize}; this copies them into the docs so the two cannot drift.
     */
    @Bean
    public OperationCustomizer roleDocumentingCustomizer() {
        return (operation, handlerMethod) -> {
            Method method = handlerMethod.getMethod();
            boolean isPublic = AnnotatedElementUtils.hasAnnotation(method, SecurityRequirements.class);
            if (isPublic) {
                return operation;
            }
            PreAuthorize preAuthorize = AnnotatedElementUtils.findMergedAnnotation(method, PreAuthorize.class);
            if (preAuthorize == null) {
                preAuthorize = AnnotatedElementUtils.findMergedAnnotation(handlerMethod.getBeanType(), PreAuthorize.class);
            }
            List<String> roles = preAuthorize == null ? List.of() : parseRoles(preAuthorize.value());
            String roleLine = roles.isEmpty() ? "any authenticated user" : String.join(", ", roles);
            String existing = operation.getDescription() == null ? "" : operation.getDescription() + "\n\n";
            operation.setDescription(existing + "**Roles:** " + roleLine);

            addErrorResponse(operation, "401", "Missing, invalid or expired token, or the account is disabled");
            addErrorResponse(operation, "403", "Your role is not allowed");
            return operation;
        };
    }

    /** Pulls the role names out of {@code hasRole('A')} / {@code hasAnyRole('A', 'B')}; empty if neither appears. */
    static List<String> parseRoles(String expression) {
        List<String> roles = new ArrayList<>();
        if (expression == null) {
            return roles;
        }
        Matcher call = ROLE_EXPRESSION.matcher(expression);
        while (call.find()) {
            Matcher name = QUOTED.matcher(call.group(1));
            while (name.find()) {
                roles.add(name.group(1));
            }
        }
        return roles;
    }

    private static void addErrorResponse(Operation operation, String code, String description) {
        ApiResponses responses = operation.getResponses();
        if (responses == null) {
            responses = new ApiResponses();
            operation.setResponses(responses);
        }
        if (responses.containsKey(code)) {
            return;
        }
        Schema<?> schema = new Schema<>().$ref("#/components/schemas/ErrorResponse");
        responses.addApiResponse(code, new ApiResponse()
                .description(description)
                .content(new Content().addMediaType(
                        org.springframework.http.MediaType.APPLICATION_JSON_VALUE,
                        new MediaType().schema(schema))));
    }
}
