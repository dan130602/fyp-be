package com.fyp.be.media_processing_service;

import com.fyp.be.media_processing_service.configs.UploadProperties;
import com.fyp.be.media_processing_service.configs.security.CognitoProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableConfigurationProperties({CognitoProperties.class, UploadProperties.class})
@EnableScheduling
public class MediaProcessingServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(MediaProcessingServiceApplication.class, args);
	}

}
