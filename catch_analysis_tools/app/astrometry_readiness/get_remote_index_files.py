from bs4 import BeautifulSoup

from catch_analysis_tools.app.astrometry_readiness.constants import INDEX_URL


def get_remote_index_files(session):
    """Return the Astrometry.net index FITS filenames advertised upstream."""
    response = session.get(INDEX_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    files = sorted(
        {
            link["href"]
            for link in soup.find_all("a", href=True)
            if link["href"].startswith("index-") and link["href"].endswith(".fits")
        }
    )
    if not files:
        raise RuntimeError(f"No astrometry index files found at {INDEX_URL}")
    return files
