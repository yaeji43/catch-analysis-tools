import threading

from .constants import INDEX_URL
from .get_index_dir import get_index_dir

state_lock = threading.RLock()
worker = None
status = {
    "state": "unknown",
    "ready": False,
    "message": "Astrometry data has not been checked yet.",
    "files_present": 0,
    "expected_files": None,
    "index_dir": get_index_dir().absolute,
    "index_url": INDEX_URL,
    "updated_at": None,
    "error": None,
}
