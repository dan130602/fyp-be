package com.fyp.be.media_processing_service.configs;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;

/**
 * HTTP client for the fyp-ai LangGraph server. Locally that's http://localhost:2024; on ECS it's
 * fyp-ai's Cloud Map DNS name (AI_BASE_URL, set by infra). The read timeout is long on purpose -
 * /runs/wait blocks until the whole compliance graph finishes, one LLM call per checklist item.
 */
@Configuration
public class AiConfig {

    @Bean
    public RestClient aiRestClient(
            @Value("${app.ai.base-url}") String baseUrl,
            @Value("${app.ai.read-timeout-seconds}") long readTimeoutSeconds
    ) {
        HttpClient httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(Duration.ofSeconds(readTimeoutSeconds));

        return RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(requestFactory)
                .build();
    }
}
