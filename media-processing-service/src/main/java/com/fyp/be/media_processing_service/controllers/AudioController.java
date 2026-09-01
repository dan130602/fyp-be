package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.AudioResponse;
import com.fyp.be.media_processing_service.services.AudioService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
@RequestMapping("/api/audio")
public class AudioController {
    private final AudioService audioService;

    public AudioController(AudioService audioService) {
        this.audioService = audioService;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<AudioResponse> upload(@RequestPart("file") MultipartFile file) throws IOException {
        return ResponseEntity.ok(audioService.saveAudio(file));
    }
}
