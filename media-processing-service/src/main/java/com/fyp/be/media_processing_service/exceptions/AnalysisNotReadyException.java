package com.fyp.be.media_processing_service.exceptions;

public class AnalysisNotReadyException extends RuntimeException {
    public AnalysisNotReadyException(String message) {
        super(message);
    }
}
