package com.fyp.be.media_processing_service.dtos;

public record ReviewOverrideResponse(
        String checklistItemId,
        String aiVerdict,
        String decision,
        String comment
) {}
