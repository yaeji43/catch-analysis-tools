import io
import os
from uuid import uuid4


def _clean_prefix(prefix):
    if not prefix:
        return ""
    return prefix.strip("/")


def _public_url(base_url, key):
    return f"{base_url.rstrip('/')}/{key}"


def _get_bucket_name():
    return (
        os.environ.get("CAT_IMAGES_BUCKET")
        or os.environ.get("TF_VAR_CAT_IMAGES_BUCKET_NAME")
        or os.environ.get("TF_VAR_S3_BUCKET_NAME")
    )


def _get_aws_client_kwargs(region):
    kwargs = {}
    region = (
        region
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("TF_VAR_AWS_REGION")
        or "us-east-1"
    )
    if region:
        kwargs["region_name"] = region

    access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get(
        "TF_VAR_AWS_ACCESS_KEY_ID"
    )
    secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get(
        "TF_VAR_AWS_SECRET_ACCESS_KEY"
    )
    session_token = os.environ.get("AWS_SESSION_TOKEN") or os.environ.get(
        "TF_VAR_AWS_SESSION_TOKEN"
    )

    if access_key_id and secret_access_key:
        kwargs["aws_access_key_id"] = access_key_id
        kwargs["aws_secret_access_key"] = secret_access_key
        if session_token:
            kwargs["aws_session_token"] = session_token

    return kwargs


def upload_figure_png(fig, route_name, image_name):
    """Upload a Matplotlib figure to the public CAT images bucket."""
    import boto3

    bucket = _get_bucket_name()
    if not bucket:
        raise RuntimeError(
            "CAT_IMAGES_BUCKET is not set. Set CAT_IMAGES_BUCKET for runtime, "
            "or TF_VAR_CAT_IMAGES_BUCKET_NAME / TF_VAR_S3_BUCKET_NAME for local Docker."
        )

    prefix = _clean_prefix(os.environ.get("CAT_IMAGES_PREFIX", "generated-images/"))
    base_url = os.environ.get("CAT_IMAGES_BASE_URL")
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("TF_VAR_AWS_REGION")
        or "us-east-1"
    )

    key_parts = [
        part for part in [prefix, route_name, f"{uuid4().hex}-{image_name}"] if part
    ]
    key = "/".join(key_parts)

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
    image_buffer.seek(0)

    boto3.client("s3", **_get_aws_client_kwargs(region)).put_object(
        Bucket=bucket,
        Key=key,
        Body=image_buffer.getvalue(),
        ContentType="image/png",
        CacheControl="public, max-age=86400",
    )

    if not base_url:
        if region:
            base_url = f"https://{bucket}.s3.{region}.amazonaws.com"
        else:
            base_url = f"https://{bucket}.s3.amazonaws.com"

    return {
        "url": _public_url(base_url, key),
        "bucket": bucket,
        "key": key,
    }
