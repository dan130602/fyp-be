package com.fyp.be.media_processing_service.exceptions;

public class CognitoAdminException extends RuntimeException {
    public CognitoAdminException(String message, Throwable cause) {
        super(message, cause);
    }
}
