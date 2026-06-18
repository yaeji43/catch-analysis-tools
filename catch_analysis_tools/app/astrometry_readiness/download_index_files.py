import os
from urllib.parse import urljoin

from .constants import INDEX_URL
from .count_complete_index_files import count_complete_index_files
from .get_index_dir import get_index_dir
from .set_astrometry_readiness_status import set_astrometry_readiness_status


def download_index_files(session, expected_files):
    """Download any missing Astrometry.net index files into the index dir."""
    target_dir = get_index_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    for href in expected_files:
        final_path = target_dir / href
        if final_path.exists() and final_path.stat().st_size > 0:
            continue

        tmp_path = final_path.with_name(f".{final_path.name}.tmp")
        file_url = urljoin(INDEX_URL, href)
        set_astrometry_readiness_status(
            state="downloading",
            ready=False,
            message=f"Downloading {href}",
            files_present=count_complete_index_files(expected_files),
            error=None,
        )

        with session.get(file_url, stream=True, timeout=300) as download:
            download.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        os.replace(tmp_path, final_path)
