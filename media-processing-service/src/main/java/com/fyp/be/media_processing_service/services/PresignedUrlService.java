package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.configs.UploadProperties;
import com.fyp.be.media_processing_service.dtos.FileUploadRequest;
import com.fyp.be.media_processing_service.dtos.FileUploadResponse;
import com.fyp.be.media_processing_service.dtos.PresignedUploadResponse;
import com.fyp.be.media_processing_service.exceptions.InvalidFileException;
import com.fyp.be.media_processing_service.exceptions.PresignedUrlGenerationException;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PresignedPutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Object keys are always generated here from the authenticated user's internal
 * id, never accepted from the client, so a caller cannot write to another
 * user's prefix or an arbitrary key in the bucket.
 *
 * Validation lives here too (not a separate class) since PresignedUrlService is
 * its only caller - content type and extension are checked independently, so a
 * file is only accepted if BOTH are on the allow-list.
 */
@Slf4j
@Service
public class PresignedUrlService {

    private static final Pattern UNSAFE_FILENAME_CHARS = Pattern.compile("[^a-zA-Z0-9._-]");

    private final S3Presigner s3Presigner;
    private final MediaAssetRepository mediaAssetRepository;
    private final UploadProperties uploadProperties;
    private final List<String> allowedContentTypes;
    private final List<String> allowedExtensions;
    private final String bucketName;
    private final long expiryMinutes;

    public PresignedUrlService(
            S3Presigner s3Presigner,
            MediaAssetRepository mediaAssetRepository,
            UploadProperties uploadProperties,
            @Value("${app.aws.s3.bucket}") String bucketName,
            @Value("${app.aws.s3.presigned-url-expiry-minutes}") long expiryMinutes
    ) {
        this.s3Presigner = s3Presigner;
        this.mediaAssetRepository = mediaAssetRepository;
        this.uploadProperties = uploadProperties;
        this.allowedContentTypes = normalize(uploadProperties.allowedContentTypes());
        this.allowedExtensions = normalize(uploadProperties.allowedExtensions());
        this.bucketName = bucketName;
        this.expiryMinutes = expiryMinutes;
    }

    private static List<String> normalize(List<String> values) {
        return values.stream().map(v -> v.toLowerCase(Locale.ROOT)).toList();
    }

    public PresignedUploadResponse generatePresignedUploads(UUID userId, List<FileUploadRequest> files) {
        validateBatch(files);

        List<FileUploadResponse> uploads = files.stream()
                .map(file -> generateForFile(userId, file))
                .toList();

        return new PresignedUploadResponse(uploads);
    }

    private void validateBatch(List<FileUploadRequest> files) {
        if (files.isEmpty()) {
            throw new InvalidFileException("At least one file must be requested");
        }
        if (files.size() > uploadProperties.maxFilesPerRequest()) {
            throw new InvalidFileException(
                    "Too many files requested: max is " + uploadProperties.maxFilesPerRequest());
        }
        files.forEach(this::validateFile);
    }

    private void validateFile(FileUploadRequest file) {
        String contentType = file.contentType() == null ? "" : file.contentType().toLowerCase(Locale.ROOT);
        if (!allowedContentTypes.contains(contentType)) {
            throw new InvalidFileException(
                    "Content type '%s' is not allowed for file '%s'".formatted(file.contentType(), file.fileName()));
        }

        String extension = extractExtension(file.fileName());
        if (extension.isEmpty() || !allowedExtensions.contains(extension)) {
            throw new InvalidFileException(
                    "File extension is not allowed for file '%s'".formatted(file.fileName()));
        }

        if (file.fileSize() == null || file.fileSize() <= 0) {
            throw new InvalidFileException("File size must be greater than zero for file '%s'".formatted(file.fileName()));
        }
        if (file.fileSize() > uploadProperties.maxFileSizeBytes()) {
            throw new InvalidFileException(
                    "File '%s' exceeds the maximum allowed size of %d bytes".formatted(file.fileName(), uploadProperties.maxFileSizeBytes()));
        }
    }

    private String extractExtension(String fileName) {
        if (fileName == null) {
            return "";
        }
        int dotIndex = fileName.lastIndexOf('.');
        if (dotIndex < 0 || dotIndex == fileName.length() - 1) {
            return "";
        }
        return fileName.substring(dotIndex + 1).toLowerCase(Locale.ROOT);
    }

    private FileUploadResponse generateForFile(UUID userId, FileUploadRequest file) {
        String objectKey = buildObjectKey(userId, file.fileName());

        try {
            PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                    .bucket(bucketName)
                    .key(objectKey)
                    .contentType(file.contentType())
                    .build();

            PutObjectPresignRequest presignRequest = PutObjectPresignRequest.builder()
                    .signatureDuration(Duration.ofMinutes(expiryMinutes))
                    .putObjectRequest(putObjectRequest)
                    .build();

            PresignedPutObjectRequest presigned = s3Presigner.presignPutObject(presignRequest);

            MediaAsset asset = recordMediaAsset(userId, objectKey, file);

            return new FileUploadResponse(
                    asset.getId(),
                    file.fileName(),
                    objectKey,
                    presigned.url().toString(),
                    Duration.ofMinutes(expiryMinutes).toSeconds()
            );
        } catch (Exception ex) {
            throw new PresignedUrlGenerationException("Failed to generate presigned URL for file '%s'".formatted(file.fileName()), ex);
        }
    }

    private MediaAsset recordMediaAsset(UUID userId, String objectKey, FileUploadRequest file) {
        LocalDateTime now = LocalDateTime.now();
        return mediaAssetRepository.save(MediaAsset.builder()
                .id(UUID.randomUUID())
                .userId(userId)
                .objectKey(objectKey)
                .originalFilename(file.fileName())
                .contentType(file.contentType())
                .fileSizeBytes(file.fileSize())
                .transcribeJobName(jobNameForObjectKey(objectKey))
                .createdAt(now)
                .updatedAt(now)
                .build());
    }

    /**
     * Must produce byte-identical output to the trigger Lambda's _job_name_for_key
     * (lambda/transcribe-trigger/app/handler.py) for the same object key - this is the only
     * correlation between a MediaAsset row and the Transcribe job it eventually spawns, with
     * neither side ever calling the other directly.
     */
    private static String jobNameForObjectKey(String objectKey) {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 not available", ex);
        }
        byte[] hash = digest.digest(objectKey.getBytes(StandardCharsets.UTF_8));
        return "transcribe-" + HexFormat.of().formatHex(hash);
    }

    private String buildObjectKey(UUID userId, String fileName) {
        return "uploads/%s/%s-%s".formatted(userId, UUID.randomUUID(), sanitizeFilename(fileName));
    }

    private String sanitizeFilename(String fileName) {
        String baseName = fileName == null ? "file" : fileName;
        int lastSlash = Math.max(baseName.lastIndexOf('/'), baseName.lastIndexOf('\\'));
        if (lastSlash >= 0) {
            baseName = baseName.substring(lastSlash + 1);
        }

        String sanitized = UNSAFE_FILENAME_CHARS.matcher(baseName).replaceAll("_");
        sanitized = sanitized.replaceAll("^[._]+", "");
        if (sanitized.isEmpty()) {
            sanitized = "file";
        }
        return sanitized.length() > 150 ? sanitized.substring(sanitized.length() - 150) : sanitized;
    }
}
