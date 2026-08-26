package com.fyp.be.ingestion_service.services;

import com.fyp.be.ingestion_service.dtos.AudioResponse;
import com.fyp.be.ingestion_service.models.Audio;
import com.fyp.be.ingestion_service.repositories.AudioRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.UUID;

@Service
@Slf4j
public class AudioService {
    private final AudioRepository audioRepository;
    private final S3Client s3Client;

    @Value("${app.aws.s3.bucket:ingestion-audio}")
    private String bucketName;

    public AudioService(AudioRepository audioRepository, S3Client s3Client) {
        this.audioRepository = audioRepository;
        this.s3Client = s3Client;
    }

    public AudioResponse saveAudio(MultipartFile file) throws IOException {
        UUID audioId = UUID.randomUUID();
        String originalFilename = file.getOriginalFilename();
        String objectKey = "audio/%s/%s".formatted(
                audioId,
                originalFilename == null ? "audio" : originalFilename
        );

        PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                .bucket(bucketName)
                .key(objectKey)
                .contentType(file.getContentType())
                .contentLength(file.getSize())
                .build();

        try (var inputStream = file.getInputStream()) {
            s3Client.putObject(putObjectRequest, RequestBody.fromInputStream(inputStream, file.getSize()));
        }

        Audio saved = audioRepository.save(Audio.builder()
                .audioId(audioId)
                .blobUrl("s3://" + bucketName + "/" + objectKey)
                .originalFilename(originalFilename)
                .contentType(file.getContentType())
                .fileSizeBytes(file.getSize())
                .processingStatus("UPLOADED")
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build());

        return new AudioResponse(
                saved.getAudioId(),
                saved.getBlobUrl(),
                saved.getOriginalFilename(),
                saved.getContentType(),
                saved.getFileSizeBytes(),
                saved.getProcessingStatus(),
                saved.getCreatedAt(),
                saved.getUpdatedAt()
        );
    }
}
