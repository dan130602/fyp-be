output "enqueuer_function_name" {
  description = "Invoke this manually to run the nightly enqueue on demand"
  value       = aws_lambda_function.enqueuer.function_name
}

output "worker_function_name" {
  value = aws_lambda_function.worker.function_name
}

output "schedule_name" {
  value = aws_scheduler_schedule.nightly.name
}
