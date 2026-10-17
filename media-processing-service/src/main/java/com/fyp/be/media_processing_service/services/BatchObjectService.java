package com.fyp.be.media_processing_service.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;

/**
 * Removes a batch-bucket object once its Transcribe job has completed (Transcribe reads the
 * object in place, so it can't be deleted any earlier). Failures are logged, not propagated: the
 * asset is already COMPLETED and the bucket's lifecycle rule expires leftovers as a safety net,
 * so a delete hiccup must not roll back the status update or trigger a duplicate handoff.
 */
@Slf4j
@Service
public class BatchObjectService {

    private final S3Client s3Client;
    private final String batchBucketName;

    public BatchObjectService(S3Client s3Client, @Value("${app.aws.s3.batch-bucket:}") String batchBucketName) {
        this.s3Client = s3Client;
        this.batchBucketName = batchBucketName;
    }

    public void delete(String objectKey) {
        if (batchBucketName.isBlank()) {
            log.warn("S3_BATCH_BUCKET is not set; not deleting batch object '{}'", objectKey);
            return;
        }
        try {
            s3Client.deleteObject(DeleteObjectRequest.builder().bucket(batchBucketName).key(objectKey).build());
        } catch (Exception ex) {
            log.warn("Failed to delete batch object s3://{}/{}; the lifecycle rule will expire it", batchBucketName, objectKey, ex);
        }
    }
}
