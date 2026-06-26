terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.AWS_REGION

  # Prevent Terraform from loading shared config files so that env vars are the single source of truth.
  shared_config_files      = ["${path.module}/.no_aws_config"]
  shared_credentials_files = ["${path.module}/.no_aws_credentials"]
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name_prefix            = var.PROJECT_PREFIX
  normalized_name_prefix = replace(lower(var.PROJECT_PREFIX), "_", "-")
  ecr_repo_name          = "${local.name_prefix}${var.ECR_REPOSITORY_NAME}"
  cluster_name           = "${local.name_prefix}cluster"
  service_name           = "${local.name_prefix}service"
  task_family            = "${local.name_prefix}task"
  container_name         = "${local.name_prefix}app"
  log_group_name         = "/aws/ecs/${local.name_prefix}app"
  alb_name               = trimsuffix(substr("${local.normalized_name_prefix}alb", 0, 32), "-")
  target_group_name      = trimsuffix(substr("${local.normalized_name_prefix}tg", 0, 32), "-")
  efs_name               = trimsuffix(substr("${local.normalized_name_prefix}efs", 0, 32), "-")
  execution_role_name    = substr("${local.name_prefix}ecs-task-execution-role", 0, 64)
  task_role_name         = substr("${local.name_prefix}ecs-task-role", 0, 64)
  cat_images_origin_id   = "${local.normalized_name_prefix}cat-images"
  cat_images_base_url    = "https://${aws_cloudfront_distribution.cat_images.domain_name}"
  cat_images_bucket_name = coalesce(var.S3_BUCKET_NAME, var.CAT_IMAGES_BUCKET_NAME)
  cat_cache_bucket_name  = coalesce(var.CAT_CACHE_BUCKET_NAME, "${local.normalized_name_prefix}cache")
}

resource "aws_ecr_repository" "app" {
  name                 = local.ecr_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = var.ECR_IMAGE_SCAN_ON_PUSH
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only recent images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.ECR_KEEP_IMAGE_COUNT
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_vpc" "main" {
  cidr_block           = var.VPC_CIDR
  enable_dns_support   = true
  enable_dns_hostnames = true
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

resource "aws_subnet" "public" {
  count                   = length(var.PUBLIC_SUBNET_CIDRS)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.PUBLIC_SUBNET_CIDRS[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index % length(data.aws_availability_zones.available.names)]
  map_public_ip_on_launch = true
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = length(var.PUBLIC_SUBNET_CIDRS)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}alb-sg"
  description = "Allow inbound HTTP traffic from the internet"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${local.name_prefix}ecs-tasks-sg"
  description = "Allow inbound app traffic from the ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "App traffic from ALB"
    from_port       = var.CONTAINER_PORT
    to_port         = var.CONTAINER_PORT
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "efs" {
  name        = "${local.name_prefix}efs-sg"
  description = "Allow NFS traffic from ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "NFS from ECS tasks"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = local.log_group_name
  retention_in_days = var.ECS_LOGS_RETENTION_DAYS
}

resource "aws_s3_bucket" "cat_images" {
  bucket = local.cat_images_bucket_name

  tags = {
    Name = local.cat_images_bucket_name
  }
}

resource "aws_s3_bucket_ownership_controls" "cat_images" {
  bucket = aws_s3_bucket.cat_images.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "cat_images" {
  bucket = aws_s3_bucket.cat_images.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cat_images" {
  bucket = aws_s3_bucket.cat_images.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_cors_configuration" "cat_images" {
  bucket = aws_s3_bucket.cat_images.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
}

resource "aws_s3_bucket_policy" "cat_images_public_read" {
  bucket = aws_s3_bucket.cat_images.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGeneratedImages"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.cat_images.arn}/${var.CAT_IMAGES_PREFIX}*"
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.cat_images]
}

resource "aws_cloudfront_distribution" "cat_images" {
  enabled         = true
  is_ipv6_enabled = true
  price_class     = "PriceClass_100"

  origin {
    domain_name = aws_s3_bucket.cat_images.bucket_regional_domain_name
    origin_id   = local.cat_images_origin_id
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = local.cat_images_origin_id
    viewer_protocol_policy = "redirect-to-https"
    compress               = true

    forwarded_values {
      query_string = false

      cookies {
        forward = "none"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_s3_bucket" "cat_cache" {
  bucket = local.cat_cache_bucket_name

  tags = {
    Name = local.cat_cache_bucket_name
  }
}

resource "aws_s3_bucket_public_access_block" "cat_cache" {
  bucket = aws_s3_bucket.cat_cache.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cat_cache" {
  bucket = aws_s3_bucket.cat_cache.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cat_cache" {
  bucket = aws_s3_bucket.cat_cache.id

  rule {
    id     = "expire-cat-route-cache"
    status = "Enabled"

    filter {
      prefix = var.CAT_CACHE_PREFIX
    }

    expiration {
      days = var.CAT_CACHE_RETENTION_DAYS
    }
  }
}

resource "aws_efs_file_system" "astrometry_data" {
  creation_token   = "${local.name_prefix}astrometry-data"
  performance_mode = var.EFS_PERFORMANCE_MODE
  throughput_mode  = var.EFS_THROUGHPUT_MODE
  encrypted        = true

  tags = {
    Name = local.efs_name
  }
}

resource "aws_efs_mount_target" "astrometry_data" {
  count           = length(aws_subnet.public)
  file_system_id  = aws_efs_file_system.astrometry_data.id
  subnet_id       = aws_subnet.public[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_iam_role" "ecs_task_execution" {
  name = local.execution_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_default" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = local.task_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_cat_images" {
  name = "${local.name_prefix}cat-images"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.cat_images.arn}/${var.CAT_IMAGES_PREFIX}*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_cat_cache" {
  name = "${local.name_prefix}cat-cache"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.cat_cache.arn}/${var.CAT_CACHE_PREFIX}*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.cat_cache.arn
        Condition = {
          StringLike = {
            "s3:prefix" = [
              var.CAT_CACHE_PREFIX,
              "${var.CAT_CACHE_PREFIX}*"
            ]
          }
        }
      }
    ]
  })
}

resource "aws_ecs_cluster" "main" {
  name = local.cluster_name
}

resource "aws_ecs_task_definition" "app" {
  family                   = local.task_family
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.ECS_TASK_CPU)
  memory                   = tostring(var.ECS_TASK_MEMORY)
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  runtime_platform {
    cpu_architecture        = lower(var.CAT_ARCHITECTURE) == "arm64" ? "ARM64" : "X86_64"
    operating_system_family = "LINUX"
  }

  volume {
    name = "astrometry-data"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.astrometry_data.id
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    {
      name      = local.container_name
      image     = "${aws_ecr_repository.app.repository_url}:${var.DOCKER_IMAGE_TAG}"
      essential = true
      environment = [
        {
          name  = "ASTROMETRY_INDEX_DIR"
          value = "/root/.astrometry/data"
        },
        {
          name  = "DOCKER_IMAGE_TAG"
          value = var.DOCKER_IMAGE_TAG
        },
        {
          name  = "AWS_REGION"
          value = var.AWS_REGION
        },
        {
          name  = "CAT_IMAGES_BUCKET"
          value = aws_s3_bucket.cat_images.bucket
        },
        {
          name  = "CAT_IMAGES_PREFIX"
          value = var.CAT_IMAGES_PREFIX
        },
        {
          name  = "CAT_IMAGES_BASE_URL"
          value = local.cat_images_base_url
        },
        {
          name  = "CAT_CACHE_BUCKET"
          value = aws_s3_bucket.cat_cache.bucket
        },
        {
          name  = "CAT_CACHE_PREFIX"
          value = var.CAT_CACHE_PREFIX
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "astrometry-data"
          containerPath = "/root/.astrometry/data"
          readOnly      = false
        }
      ]
      portMappings = [
        {
          containerPort = var.CONTAINER_PORT
          hostPort      = var.CONTAINER_PORT
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = var.AWS_REGION
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  depends_on = [
    aws_iam_role_policy_attachment.ecs_task_execution_default,
    aws_iam_role_policy.ecs_task_cat_images,
    aws_iam_role_policy.ecs_task_cat_cache,
    aws_efs_mount_target.astrometry_data,
  ]
}

resource "aws_lb" "main" {
  name               = local.alb_name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "app" {
  name        = local.target_group_name
  port        = var.CONTAINER_PORT
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    enabled  = true
    path     = var.HEALTHCHECK_PATH
    matcher  = "200"
    protocol = "HTTP"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_ecs_service" "app" {
  name            = local.service_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.ECS_DESIRED_COUNT
  launch_type     = "FARGATE"
  platform_version = "1.4.0"
  health_check_grace_period_seconds = var.ECS_HEALTHCHECK_GRACE_PERIOD_SECONDS

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = local.container_name
    container_port   = var.CONTAINER_PORT
  }

  depends_on = [aws_lb_listener.http]
}
