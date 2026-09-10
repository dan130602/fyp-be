package com.fyp.be.media_processing_service.exceptions;

/** Thrown when an admin attempts to change their own role/status, or disable the last active admin. */
public class SelfModificationException extends RuntimeException {
    public SelfModificationException(String message) {
        super(message);
    }
}
