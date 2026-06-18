from ..astrometry_readiness.start_astrometry_background_check import (
    start_astrometry_background_check,
)


def reset():
    """Force an astrometry index readiness recheck in the background."""
    status = start_astrometry_background_check(force=True)
    return {
        "status": "reset_started",
        "astrometry_data": status,
    }, 202
