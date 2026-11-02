package com.fyp.be.media_processing_service.repositories;

import java.time.LocalDateTime;
import java.util.UUID;

/** One call as the dashboard needs it. analysis holds only the output fields it reads, never the transcript. */
public interface DashboardCallRow {
    UUID getMediaAssetId();
    String getFileName();
    String getMediaStatus();
    LocalDateTime getCreatedAt();
    UUID getAdvisorId();
    String getEmployeeId();
    String getFullName();
    String getEmail();
    String getDepartment();
    String getAdvisorStatus();
    UUID getChecklistId();
    String getChecklistStatus();
    /** When the AI output was first stored; null while the call has no output. */
    LocalDateTime getAnalysedAt();
    String getAnalysis();
}
