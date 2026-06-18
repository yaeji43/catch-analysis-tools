import base64
import json
import logging
import os
from tempfile import NamedTemporaryFile
from uuid import uuid4

import requests
from flask import Response
from werkzeug.exceptions import BadRequest

from ..astrometry_readiness.get_astrometry_readiness_status import (
    get_astrometry_readiness_status,
)
from ..astrometry_readiness.is_astrometry_ready import (
    is_astrometry_ready,
)
from ..services.astrometry import (
    AstrometrySolveError,
    AstrometryValidationError,
    run_pipeline,
    validate_and_normalize,
)

logger = logging.getLogger(__name__)


def do_astrometry(body):
    """Handle POST /astrometry and translate service results to HTTP responses."""
    if not is_astrometry_ready():
        payload = {
            "status": "not_ready",
            "message": "Astrometry index files are not ready yet.",
            "astrometry_data": get_astrometry_readiness_status(),
        }
        return Response(
            json.dumps(payload),
            status=503,
            mimetype="application/json",
            headers={"Retry-After": "30"},
        )

    request_id = uuid4().hex[:12]
    stage = "validate_request"
    # Capture raw request context for error logs before validation normalizes it.
    image_url = body.get("image_url")
    return_plot = body.get("return_plot")
    plot_type = body.get("plot_type")

    try:
        cfg = validate_and_normalize(body)

        image_url = cfg["image_url"]
        return_plot = cfg["meta"]["return_plot"]
        plot_type = cfg["meta"]["plot_type"]

        stage = "fetch_fits"
        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            raise BadRequest("Could not retrieve FITS image")

        stage = "write_temp_fits"
        with NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            stage = "run_pipeline"
            results = run_pipeline(tmp_path, cfg)

            stage = "build_response"
            if return_plot:
                if plot_type not in results.get("plots", {}):
                    raise BadRequest(f"Unknown plot_type: {plot_type}")

                image_bytes = base64.b64decode(results["plots"][plot_type])
                return Response(image_bytes, mimetype="image/png")

            return results, 200, {"Content-Type": "application/json"}
        finally:
            os.remove(tmp_path)
    except AstrometryValidationError as exc:
        raise BadRequest(str(exc))
    except BadRequest:
        raise
    except AstrometrySolveError as exc:
        logger.warning(
            "Astrometry solve did not produce WCS "
            "[request_id=%s stage=%s image_url=%r return_plot=%r plot_type=%r]",
            request_id,
            stage,
            image_url,
            return_plot,
            plot_type,
        )
        payload = {
            "status": "solve_failed",
            "message": str(exc),
            "request_id": request_id,
        }
        return Response(json.dumps(payload), status=422, mimetype="application/json")
    except Exception:
        logger.exception(
            "Astrometry request failed "
            "[request_id=%s stage=%s image_url=%r return_plot=%r plot_type=%r]",
            request_id,
            stage,
            image_url,
            return_plot,
            plot_type,
        )
        raise
