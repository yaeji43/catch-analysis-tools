import fcntl

import requests

from .acquire_index_download_lock import acquire_index_download_lock
from .constants import INDEX_URL
from .count_complete_index_files import count_complete_index_files
from .download_index_files import download_index_files
from .get_astrometry_readiness_status import get_astrometry_readiness_status
from .get_index_dir import get_index_dir
from .get_remote_index_files import get_remote_index_files
from .index_files_complete import index_files_complete
from .set_astrometry_readiness_status import set_astrometry_readiness_status
from .write_ready_sentinel import write_ready_sentinel


def prepare_astrometry_data(force=False):
    """Check/download index files and mark astrometry ready when complete."""
    target_dir = get_index_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        expected_files = get_remote_index_files(session)
        expected_count = len(expected_files)

        set_astrometry_readiness_status(
            state="checking",
            ready=False,
            message="Checking astrometry index files.",
            index_dir=str(target_dir),
            index_url=INDEX_URL,
            expected_files=expected_count,
            error=None,
        )

        complete, files_present = index_files_complete(expected_files)
        if complete and not force:
            write_ready_sentinel(files_present, expected_files)
            set_astrometry_readiness_status(
                state="ready",
                ready=True,
                message="Astrometry index files are ready.",
                files_present=files_present,
                error=None,
            )
            return get_astrometry_readiness_status()

        lock_handle = acquire_index_download_lock()
        try:
            complete, files_present = index_files_complete(expected_files)
            if complete and not force:
                write_ready_sentinel(files_present, expected_files)
                set_astrometry_readiness_status(
                    state="ready",
                    ready=True,
                    message="Astrometry index files are ready.",
                    files_present=files_present,
                    error=None,
                )
                return get_astrometry_readiness_status()

            set_astrometry_readiness_status(
                state="downloading",
                ready=False,
                message=(
                    "Astrometry index files are incomplete. Downloading missing files."
                ),
                files_present=files_present,
                error=None,
            )
            download_index_files(session, expected_files)

            complete, files_present = index_files_complete(expected_files)
            if not complete:
                raise RuntimeError(
                    "Astrometry index files incomplete: "
                    f"{files_present} / {expected_count}"
                )

            write_ready_sentinel(files_present, expected_files)
            set_astrometry_readiness_status(
                state="ready",
                ready=True,
                message="Astrometry index files are ready.",
                files_present=files_present,
                error=None,
            )
            return get_astrometry_readiness_status()
        except Exception as exc:
            set_astrometry_readiness_status(
                state="error",
                ready=False,
                message="Astrometry index preparation failed.",
                files_present=count_complete_index_files(expected_files),
                error=str(exc),
            )
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
