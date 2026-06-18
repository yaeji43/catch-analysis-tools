from ..astrometry_readiness.get_astrometry_readiness_status import (
    get_astrometry_readiness_status,
)


def health():
    """Return service health plus astrometry index readiness details."""
    return {
        "status": "ok",
        "astrometry_data": get_astrometry_readiness_status(),
    }
