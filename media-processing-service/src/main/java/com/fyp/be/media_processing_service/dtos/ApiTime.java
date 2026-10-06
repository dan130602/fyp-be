package com.fyp.be.media_processing_service.dtos;

import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;

/** Entities store zone-less LocalDateTime.now(); responses carry the offset so clients don't read UTC as local. */
public final class ApiTime {

    private ApiTime() {}

    public static OffsetDateTime of(LocalDateTime value) {
        return value == null ? null : value.atZone(ZoneId.systemDefault()).toOffsetDateTime();
    }
}
