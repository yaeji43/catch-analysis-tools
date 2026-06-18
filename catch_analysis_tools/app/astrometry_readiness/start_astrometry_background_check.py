import threading

from . import state
from .background_prepare_astrometry_data import background_prepare_astrometry_data
from .constants import INDEX_URL
from .get_current_time import get_current_time
from .get_index_dir import get_index_dir


def start_astrometry_background_check(force=False):
    """Start one background thread to check/download astrometry index files."""
    with state.state_lock:
        if state.worker is not None and state.worker.is_alive():
            return dict(state.status)

        state.status.update(
            {
                "state": "checking",
                "ready": False,
                "message": "Astrometry data readiness check has started.",
                "index_dir": str(get_index_dir()),
                "index_url": INDEX_URL,
                "expected_files": None,
                "error": None,
                "updated_at": get_current_time(),
            }
        )
        state.worker = threading.Thread(
            target=background_prepare_astrometry_data,
            kwargs={"force": force},
            daemon=True,
        )
        state.worker.start()
        return dict(state.status)
