import copy
import hashlib
import json
import os
from decimal import Decimal

from .generated_images import _get_aws_client_kwargs, _public_url

CACHE_VERSION = "v1"


def _clean_prefix(prefix):
    if not prefix:
        return ""
    return prefix.strip("/")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value):
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


def make_cache_key(route_name, inputs):
    prefix = _clean_prefix(os.environ.get("CAT_CACHE_PREFIX", "cache/"))
    key_material = {
        "route": route_name,
        "version": CACHE_VERSION,
        "inputs": inputs,
    }
    digest = hashlib.sha256(_canonical_json(key_material).encode("utf-8")).hexdigest()
    key_parts = [
        part for part in [prefix, route_name, CACHE_VERSION, f"{digest}.json"] if part
    ]
    return "/".join(key_parts)


def _cache_bucket():
    bucket = os.environ.get("CAT_CACHE_BUCKET") or os.environ.get(
        "TF_VAR_CAT_CACHE_BUCKET_NAME"
    )
    if bucket:
        return bucket

    project_prefix = (
        os.environ.get("TF_VAR_PROJECT_PREFIX")
        or os.environ.get("PROJECT_PREFIX")
        or "sbn-cat-"
    )
    return f"{project_prefix.lower().replace('_', '-')}cache"


def _s3_client():
    import boto3

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("TF_VAR_AWS_REGION")
        or "us-east-1"
    )
    return boto3.client("s3", **_get_aws_client_kwargs(region))


def _read_json(bucket, key):
    try:
        response = _s3_client().get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in {
            "NoSuchKey",
            "404",
        }:
            return None
        raise
    return json.loads(response["Body"].read().decode("utf-8"))


def _write_json(bucket, key, payload):
    body = _canonical_json(payload).encode("utf-8")
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        CacheControl="private, max-age=0",
    )


def _without_cache_metadata(payload):
    if isinstance(payload, dict):
        clean = copy.deepcopy(payload)
        clean.pop("cache", None)
        return clean
    return payload


def _add_cache_metadata(payload, hit, key, enabled=True, error=None):
    if not isinstance(payload, dict):
        return payload
    payload = copy.deepcopy(payload)
    payload["cache"] = {
        "enabled": enabled,
        "hit": hit,
        "key": key,
    }
    if error:
        payload["cache"]["error"] = error
    return payload


def _refresh_image_urls(payload):
    if not isinstance(payload, dict):
        return payload

    payload = copy.deepcopy(payload)
    base_url = os.environ.get("CAT_IMAGES_BASE_URL")
    bucket = (
        os.environ.get("CAT_IMAGES_BUCKET")
        or os.environ.get("TF_VAR_CAT_IMAGES_BUCKET_NAME")
        or os.environ.get("TF_VAR_S3_BUCKET_NAME")
    )
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("TF_VAR_AWS_REGION")
        or "us-east-1"
    )
    if not base_url and bucket:
        base_url = f"https://{bucket}.s3.{region}.amazonaws.com"

    if base_url and payload.get("centroid_figure_s3_key"):
        url = _public_url(base_url, payload["centroid_figure_s3_key"])
        payload["centroid_figure"] = url
        payload["centroid_figure_url"] = url

    if base_url and payload.get("aperture_figure_s3_key"):
        url = _public_url(base_url, payload["aperture_figure_s3_key"])
        payload["aperture_figure"] = url
        payload["aperture_figure_url"] = url

    return payload


def _is_fully_successful(payload):
    if not isinstance(payload, dict):
        return False

    status = payload.get("status")
    if status is not None and status != "success":
        return False

    for value in payload.values():
        if isinstance(value, dict) and value.get("status") not in (None, "success"):
            return False

    return True


def get_or_compute(route_name, inputs, compute):
    bucket = _cache_bucket()

    if not bucket:
        return _add_cache_metadata(compute(), False, None, enabled=False)

    try:
        key = make_cache_key(route_name, inputs)
    except TypeError as exc:
        result = compute()
        return _add_cache_metadata(
            result,
            False,
            None,
            enabled=False,
            error=f"disabled:non_json_input:{type(exc).__name__}",
        )

    try:
        cached = _read_json(bucket, key)
    except Exception as exc:
        result = compute()
        return _add_cache_metadata(
            result, False, key, error=f"read_failed:{type(exc).__name__}"
        )

    if cached is not None:
        cached = _refresh_image_urls(cached)
        return _add_cache_metadata(cached, True, key)

    result = compute()
    if not _is_fully_successful(result):
        return _add_cache_metadata(
            result, False, key, error="not_cached:result_not_fully_successful"
        )

    try:
        _write_json(bucket, key, _without_cache_metadata(result))
    except Exception as exc:
        return _add_cache_metadata(
            result, False, key, error=f"write_failed:{type(exc).__name__}"
        )

    return _add_cache_metadata(result, False, key)
