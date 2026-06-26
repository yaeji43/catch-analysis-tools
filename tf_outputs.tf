output "alb_dns_name" {
  description = "Public DNS name of the application load balancer"
  value       = aws_lb.main.dns_name
}

output "service_base_url" {
  description = "Base HTTP URL for the Fargate service"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL to tag and push container images"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

output "efs_file_system_id" {
  description = "EFS file system ID used to persist astrometry index files"
  value       = aws_efs_file_system.astrometry_data.id
}

output "cat_images_bucket_name" {
  description = "Public S3 bucket used for generated CAT route images"
  value       = aws_s3_bucket.cat_images.bucket
}

output "cat_images_cloudfront_domain_name" {
  description = "CloudFront domain name serving generated CAT route images"
  value       = aws_cloudfront_distribution.cat_images.domain_name
}

output "cat_images_base_url" {
  description = "HTTPS base URL for generated CAT route images"
  value       = local.cat_images_base_url
}

output "cat_cache_bucket_name" {
  description = "Private S3 bucket used for CAT route JSON result cache"
  value       = aws_s3_bucket.cat_cache.bucket
}

output "cat_cache_prefix" {
  description = "S3 key prefix used for CAT route JSON result cache"
  value       = var.CAT_CACHE_PREFIX
}
