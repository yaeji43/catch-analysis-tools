variable "AWS_REGION" {
  description = "AWS region"
  type        = string
}

variable "PROJECT_PREFIX" {
  description = "Unique, descriptive prefix applied to named resources. Must end with a hyphen if you plan to append suffixes."
  type        = string
  default     = "sbn-cat-"
}

variable "VPC_CIDR" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.42.0.0/16"
}

variable "PUBLIC_SUBNET_CIDRS" {
  description = "Public subnet CIDR blocks used for ALB and Fargate tasks"
  type        = list(string)
  default     = ["10.42.1.0/24", "10.42.2.0/24"]

  validation {
    condition     = length(var.PUBLIC_SUBNET_CIDRS) >= 2
    error_message = "PUBLIC_SUBNET_CIDRS must contain at least two subnet CIDRs for the ALB."
  }
}

variable "ECR_REPOSITORY_NAME" {
  description = "ECR repository suffix appended to PROJECT_PREFIX"
  type        = string
  default     = "app"
}

variable "DOCKER_IMAGE_TAG" {
  description = "Container image tag deployed to ECS from ECR"
  type        = string
}

variable "ECR_IMAGE_SCAN_ON_PUSH" {
  description = "Enable vulnerability scanning on image push in ECR"
  type        = bool
  default     = true
}

variable "ECR_KEEP_IMAGE_COUNT" {
  description = "How many recent images ECR lifecycle policy should retain"
  type        = number
  default     = 20
}

variable "CONTAINER_PORT" {
  description = "Container port exposed by the Flask app"
  type        = number
  default     = 8000
}

variable "HEALTHCHECK_PATH" {
  description = "ALB health check path expected to return HTTP 200"
  type        = string
  default     = "/healthz"
}

variable "ECS_TASK_CPU" {
  description = "Fargate task CPU units"
  type        = number
  default     = 512
}

variable "ECS_TASK_MEMORY" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "ECS_DESIRED_COUNT" {
  description = "Desired number of running ECS tasks"
  type        = number
  default     = 1
}

variable "ECS_HEALTHCHECK_GRACE_PERIOD_SECONDS" {
  description = "Grace period to allow initial container startup tasks such as data downloads before ALB health checks count against the service"
  type        = number
  default     = 1800
}

variable "ECS_LOGS_RETENTION_DAYS" {
  description = "CloudWatch Logs retention in days for ECS container logs"
  type        = number
  default     = 14
}

variable "EFS_PERFORMANCE_MODE" {
  description = "Performance mode for the astrometry EFS file system"
  type        = string
  default     = "generalPurpose"
}

variable "EFS_THROUGHPUT_MODE" {
  description = "Throughput mode for the astrometry EFS file system"
  type        = string
  default     = "bursting"
}

variable "CAT_ARCHITECTURE" {
  description = "CPU architecture for ECS task runtime platform (x86_64 or arm64)"
  type        = string
  default     = "x86_64"

  validation {
    condition     = contains(["x86_64", "arm64"], lower(var.CAT_ARCHITECTURE))
    error_message = "CAT_ARCHITECTURE must be either x86_64 or arm64."
  }
}

variable "CAT_IMAGES_BUCKET_NAME" {
  description = "Public S3 bucket name for generated CAT route images. Must be globally unique and lowercase."
  type        = string
  default     = "sbn-cat-images"
}

variable "S3_BUCKET_NAME" {
  description = "Compatibility alias for CAT_IMAGES_BUCKET_NAME. Prefer CAT_IMAGES_BUCKET_NAME for new configuration."
  type        = string
  default     = null
}

variable "CAT_IMAGES_PREFIX" {
  description = "S3 key prefix used for generated CAT route images"
  type        = string
  default     = "generated-images/"
}

variable "CAT_CACHE_BUCKET_NAME" {
  description = "Private S3 bucket name for CAT route JSON result cache. Leave null to derive from PROJECT_PREFIX."
  type        = string
  default     = null
}

variable "CAT_CACHE_PREFIX" {
  description = "S3 key prefix used for CAT route JSON result cache"
  type        = string
  default     = "cache/"
}

variable "CAT_CACHE_RETENTION_DAYS" {
  description = "Number of days to retain CAT route JSON cache objects"
  type        = number
  default     = 30
}
