package com.fyp.be.media_processing_service.controllers;

import com.fyp.be.media_processing_service.dtos.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.annotations.Parameter;
import com.fyp.be.media_processing_service.services.ProductionMetrics;
import com.fyp.be.media_processing_service.services.ProductionReviewReader;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import tools.jackson.databind.ObjectMapper;
import java.time.LocalDateTime;
import java.util.*;

@Tag(name = "Observability", description = "Production AI-quality metrics for the developer dashboard")
@RestController @RequestMapping("/api/observability")
public class ObservabilityController {
    private final ProductionReviewReader reader;
    private final ProductionMetrics metrics;
    public ObservabilityController(ProductionReviewReader reader, ObjectMapper mapper) {
        this.reader = reader;
        metrics = new ProductionMetrics(mapper);
    }
    @Operation(summary = "Production AI-quality metrics",
            description = "Officer review decisions are the reference answer, not independent ground truth. The response is a free-form object; its shape is documented in docs/API.md#observability.")
    @ApiResponse(responseCode = "200", description = "OK")
    @ApiResponse(responseCode = "400", description = "days is outside 1-90", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    @GetMapping @PreAuthorize("hasAnyRole('ADMIN', 'SUPER_ADMIN', 'MANAGER', 'DEVELOPER')")
    public Map<String, Object> overview(
            @Parameter(description = "Window size in days, 1-90.") @RequestParam(defaultValue = "7") int days) {
        if (days < 1 || days > 90) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "days must be between 1 and 90");
        var now = LocalDateTime.now();
        var data = reader.read(days, now);
        return metrics.summarize(data.outputs(), data.checklists(), data.history(), days, now);
    }
}
