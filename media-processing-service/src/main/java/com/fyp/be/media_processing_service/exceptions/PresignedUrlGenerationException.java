package com.fyp.be.media_processing_service.exceptions;

public class PresignedUrlGenerationException extends RuntimeException {
    public PresignedUrlGenerationException(String message, Throwable cause) {
        super(message, cause);
    }
}
