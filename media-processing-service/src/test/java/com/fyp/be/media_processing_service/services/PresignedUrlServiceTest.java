package com.fyp.be.media_processing_service.services;

import com.fyp.be.media_processing_service.configs.UploadProperties;
import com.fyp.be.media_processing_service.dtos.FileUploadRequest;
import com.fyp.be.media_processing_service.dtos.FileUploadResponse;
import com.fyp.be.media_processing_service.dtos.PresignedUploadResponse;
import com.fyp.be.media_processing_service.exceptions.InvalidFileException;
import com.fyp.be.media_processing_service.exceptions.PresignedUrlGenerationException;
import com.fyp.be.media_processing_service.models.MediaAsset;
import com.fyp.be.media_processing_service.repositories.MediaAssetRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;
import software.amazon.awssdk.services.s3.presigner.model.PresignedGetObjectRequest;
import software.amazon.awssdk.services.s3.presigner.model.PresignedPutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PresignedUrlServiceTest {

    private static final UUID USER_ID = UUID.randomUUID();
    private static final String BUCKET = "test-bucket";

    private S3Presigner presigner;
    private MediaAssetRepository mediaAssetRepository;
    private ChecklistService checklistService;
    private PresignedUrlService service;

    @BeforeEach
    void setUp() throws Exception {
        presigner = mock(S3Presigner.class);
        mediaAssetRepository = mock(MediaAssetRepository.class);
        checklistService = mock(ChecklistService.class);

        PresignedPutObjectRequest presignedPut = mock(PresignedPutObjectRequest.class);
        when(presignedPut.url()).thenReturn(URI.create("https://s3.example.com/put").toURL());
        when(presigner.presignPutObject(any(PutObjectPresignRequest.class))).thenReturn(presignedPut);

        when(mediaAssetRepository.save(any(MediaAsset.class))).thenAnswer(inv -> inv.getArgument(0));

        UploadProperties props = new UploadProperties(
                2, 1_000L, List.of("Audio/MP3", "audio/wav"), List.of("MP3", "wav"));
        service = new PresignedUrlService(presigner, mediaAssetRepository, checklistService, props, BUCKET, 15, 30);
    }

    private static FileUploadRequest file(String name, String type, Long size) {
        return new FileUploadRequest(name, type, size, null);
    }

    private String keyFor(String fileName) {
        return service.generatePresignedUploads(USER_ID, List.of(file(fileName, "audio/mp3", 10L)))
                .uploads().get(0).objectKey();
    }

    @Test
    void generatePresignedUploads_validFile_buildsUserScopedKeyAndPersistsAsset() {
        PresignedUploadResponse response =
                service.generatePresignedUploads(USER_ID, List.of(file("call.mp3", "audio/mp3", 500L)));

        assertThat(response.uploads()).hasSize(1);
        FileUploadResponse upload = response.uploads().get(0);
        assertThat(upload.objectKey()).startsWith("uploads/" + USER_ID + "/").endsWith("-call.mp3");
        assertThat(upload.presignedUrl()).isEqualTo("https://s3.example.com/put");
        assertThat(upload.expiresIn()).isEqualTo(Duration.ofMinutes(15).toSeconds());
        assertThat(upload.fileName()).isEqualTo("call.mp3");

        ArgumentCaptor<MediaAsset> captor = ArgumentCaptor.forClass(MediaAsset.class);
        verify(mediaAssetRepository).save(captor.capture());
        MediaAsset saved = captor.getValue();
        assertThat(saved.getUserId()).isEqualTo(USER_ID);
        assertThat(saved.getObjectKey()).isEqualTo(upload.objectKey());
        assertThat(saved.getFileSizeBytes()).isEqualTo(500L);
        assertThat(saved.getTranscribeJobName()).matches("transcribe-" + USER_ID + "-[0-9a-f]{16}");
        assertThat(upload.id()).isEqualTo(saved.getId());
        verify(checklistService).createForUpload(saved.getId(), USER_ID, null);
    }

    @Test
    void generatePresignedUploads_sameInputTwice_producesDifferentKeys() {
        assertThat(keyFor("call.mp3")).isNotEqualTo(keyFor("call.mp3"));
    }

    @Test
    void generatePresignedUploads_pathTraversalFilename_isSanitized() {
        String key = keyFor("../../etc/pa ss.mp3");
        String prefix = "uploads/" + USER_ID + "/";

        assertThat(key).startsWith(prefix).endsWith("-pa_ss.mp3");
        assertThat(key.substring(prefix.length())).doesNotContain("/").doesNotContain("..");
    }

    @Test
    void generatePresignedUploads_backslashPath_keepsOnlyBaseName() {
        assertThat(keyFor("C:\\temp\\call.mp3")).endsWith("-call.mp3");
    }

    @Test
    void generatePresignedUploads_leadingDotsStripped() {
        assertThat(keyFor("..hidden.mp3")).endsWith("-hidden.mp3");
    }

    @Test
    void generatePresignedUploads_veryLongFilename_isTruncatedKeepingExtension() {
        String key = keyFor("a".repeat(300) + ".mp3");
        String afterPrefix = key.substring(("uploads/" + USER_ID + "/").length());
        // "<36-char uuid>-<sanitized name>"
        String sanitized = afterPrefix.substring(37);

        assertThat(sanitized).hasSize(150).endsWith(".mp3");
    }

    @Test
    void generatePresignedUploads_caseInsensitiveAllowList_accepted() {
        assertThat(service.generatePresignedUploads(USER_ID, List.of(file("CALL.MP3", "AUDIO/MP3", 10L))).uploads())
                .hasSize(1);
    }

    @Test
    void generatePresignedUploads_emptyBatch_rejected() {
        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of()))
                .isInstanceOf(InvalidFileException.class)
                .hasMessageContaining("At least one file");
    }

    @Test
    void generatePresignedUploads_tooManyFiles_rejected() {
        FileUploadRequest f = file("a.mp3", "audio/mp3", 10L);

        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(f, f, f)))
                .isInstanceOf(InvalidFileException.class)
                .hasMessageContaining("max is 2");
    }

    @Test
    void generatePresignedUploads_disallowedContentType_rejected() {
        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file("a.mp3", "video/mp4", 10L))))
                .isInstanceOf(InvalidFileException.class)
                .hasMessageContaining("Content type");
    }

    @Test
    void generatePresignedUploads_nullContentType_rejected() {
        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file("a.mp3", null, 10L))))
                .isInstanceOf(InvalidFileException.class);
    }

    @Test
    void generatePresignedUploads_disallowedExtension_rejected() {
        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file("a.exe", "audio/mp3", 10L))))
                .isInstanceOf(InvalidFileException.class)
                .hasMessageContaining("extension");
    }

    @Test
    void generatePresignedUploads_noExtensionTrailingDotOrNullName_rejected() {
        for (String name : new String[]{"noext", "a.", null}) {
            assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file(name, "audio/mp3", 10L))))
                    .isInstanceOf(InvalidFileException.class);
        }
    }

    @Test
    void generatePresignedUploads_nullZeroOrNegativeSize_rejected() {
        for (Long size : new Long[]{null, 0L, -1L}) {
            assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file("a.mp3", "audio/mp3", size))))
                    .isInstanceOf(InvalidFileException.class)
                    .hasMessageContaining("greater than zero");
        }
    }

    @Test
    void generatePresignedUploads_oversizedFile_rejected() {
        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file("a.mp3", "audio/mp3", 1_001L))))
                .isInstanceOf(InvalidFileException.class)
                .hasMessageContaining("maximum allowed size");
    }

    @Test
    void generatePresignedUploads_sizeExactlyAtLimit_accepted() {
        assertThat(service.generatePresignedUploads(USER_ID, List.of(file("a.mp3", "audio/mp3", 1_000L))).uploads())
                .hasSize(1);
    }

    @Test
    void generatePresignedUploads_oneInvalidFileInBatch_nothingPresignedOrSaved() {
        List<FileUploadRequest> batch = List.of(file("a.mp3", "audio/mp3", 10L), file("b.exe", "audio/mp3", 10L));

        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, batch))
                .isInstanceOf(InvalidFileException.class);

        verify(presigner, never()).presignPutObject(any(PutObjectPresignRequest.class));
        verify(mediaAssetRepository, never()).save(any());
    }

    @Test
    void generatePresignedUploads_presignerFails_wrapsInPresignedUrlGenerationException() {
        when(presigner.presignPutObject(any(PutObjectPresignRequest.class))).thenThrow(new RuntimeException("boom"));

        assertThatThrownBy(() -> service.generatePresignedUploads(USER_ID, List.of(file("a.mp3", "audio/mp3", 10L))))
                .isInstanceOf(PresignedUrlGenerationException.class)
                .hasMessageContaining("a.mp3")
                .hasCauseInstanceOf(RuntimeException.class);
    }

    @Test
    void generatePlaybackUrl_usesBucketKeyAndPlaybackExpiry() throws Exception {
        PresignedGetObjectRequest presignedGet = mock(PresignedGetObjectRequest.class);
        when(presignedGet.url()).thenReturn(URI.create("https://s3.example.com/get").toURL());
        when(presigner.presignGetObject(any(GetObjectPresignRequest.class))).thenReturn(presignedGet);

        String url = service.generatePlaybackUrl("uploads/x/y.mp3");

        assertThat(url).isEqualTo("https://s3.example.com/get");
        ArgumentCaptor<GetObjectPresignRequest> captor = ArgumentCaptor.forClass(GetObjectPresignRequest.class);
        verify(presigner).presignGetObject(captor.capture());
        assertThat(captor.getValue().signatureDuration()).isEqualTo(Duration.ofMinutes(30));
        assertThat(captor.getValue().getObjectRequest().bucket()).isEqualTo(BUCKET);
        assertThat(captor.getValue().getObjectRequest().key()).isEqualTo("uploads/x/y.mp3");
        assertThat(service.getPlaybackExpiryMinutes()).isEqualTo(30);
    }
}
