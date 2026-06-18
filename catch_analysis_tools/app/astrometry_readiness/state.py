import threading

from .constants import DEFAULT_INDEX_DIR, INDEX_URL

state_lock = threading.RLock()
worker = None
status = {
    "state": "unknown",
    "ready": False,
    "message": "Astrometry data has not been checked yet.",
    "files_present": 0,
    "expected_files": None,
    "index_dir": DEFAULT_INDEX_DIR,
    "index_url": INDEX_URL,
    "updated_at": None,
    "error": None,
}
