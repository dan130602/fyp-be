package com.fyp.be.media_processing_service.models;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * One row per S3 object a presigned upload URL was issued for - records which
 * user requested it. Written at presign time, so PENDING only means "a URL was
 * issued", not that the client actually finished the S3 PUT.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "t_media_asset")
public class MediaAsset {

    @Id
    @Column(name = "id", nullable = false, updatable = false)
    private UUID id;

    @Column(name = "user_id", nullable = false, updatable = false)
    private UUID userId;

    @Column(name = "object_key", nullable = false, updatable = false, columnDefinition = "TEXT")
    private String objectKey;

    @Column(name = "original_filename", length = 255)
    private String originalFilename;

    @Column(name = "content_type", length = 100)
    private String contentType;

    @Column(name = "file_size_bytes")
    private Long fileSizeBytes;

    @Enumerated(EnumType.STRING)
    @Builder.Default
    @Column(name = "status", nullable = false, length = 30)
    private MediaAssetStatus status = MediaAssetStatus.PENDING;

    @Enumerated(EnumType.STRING)
    @Builder.Default
    @Column(name = "processing_mode", nullable = false, updatable = false, length = 20)
    private ProcessingMode processingMode = ProcessingMode.IMMEDIATE;

    /** Nullable only for rows uploaded before advisor selection existed. */
    @Column(name = "advisor_id", updatable = false)
    private UUID advisorId;

    @Column(name = "transcribe_job_name", length = 255)
    private String transcribeJobName;

    @Column(name = "transcript_uri", columnDefinition = "TEXT")
    private String transcriptUri;

    @Column(name = "failure_reason", columnDefinition = "TEXT")
    private String failureReason;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;
}
