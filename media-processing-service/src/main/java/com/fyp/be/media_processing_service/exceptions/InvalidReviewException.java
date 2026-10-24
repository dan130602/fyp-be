package com.fyp.be.media_processing_service.exceptions;

public class InvalidReviewException extends RuntimeException {
    public InvalidReviewException(String message) {
        super(message);
    }
}
