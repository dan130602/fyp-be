package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.services.ProductionMetrics;
import com.fyp.be.media_processing_service.services.ProductionReviewReader;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import tools.jackson.databind.ObjectMapper;
import java.time.LocalDateTime;
import java.util.*;

@RestController @RequestMapping("/api/observability")
public class ObservabilityController {
    private final ProductionReviewReader reader;
    private final ProductionMetrics metrics;
    public ObservabilityController(ProductionReviewReader reader, ObjectMapper mapper) {
        this.reader = reader;
        metrics = new ProductionMetrics(mapper);
    }
    @GetMapping @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN', 'MANAGER', 'DEVELOPER')")
    public Map<String, Object> overview(@RequestParam(defaultValue = "7") int days) {
        if (days < 1 || days > 90) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "days must be between 1 and 90");
        var now = LocalDateTime.now();
        var data = reader.read(days, now);
        return metrics.summarize(data.outputs(), data.checklists(), data.history(), days, now);
    }
}
