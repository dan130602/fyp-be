package com.fyp.be.media_processing_service.exceptions;

public class InvalidFileException extends RuntimeException {
    public InvalidFileException(String message) {
        super(message);
    }
}
