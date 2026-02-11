import os
import glob
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

INDEX_URL = "https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/"
LOCAL_DIR = Path(
    os.environ.get("ASTROMETRY_INDEX_DIR", "/tmp/astrometry/index_files")
)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_FILES = 300


def index_files_complete(directory):
    files = glob.glob(os.path.join(directory, "index-*.fits"))

    if len(files) != EXPECTED_FILES:
        print(f"Incomplete index set: {len(files)} / {EXPECTED_FILES}")
        return False

    return True


def download_index_files(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    with requests.Session() as session:
        response = session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]

            if not (href.startswith("index-") and href.endswith(".fits")):
                continue

            file_url = urljoin(url, href)
            local_path = os.path.join(target_dir, href)

            print(f"Downloading: {href}")

            with session.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)


def main():
    if index_files_complete(LOCAL_DIR):
        print("Index files are complete. Skipping download.")
        return

    print("Index files incomplete or missing. Downloading...")
    download_index_files(INDEX_URL, LOCAL_DIR)

    if not index_files_complete(LOCAL_DIR):
        raise RuntimeError("Index files still incomplete after download")

    print("Index files downloaded successfully.")


if __name__ == "__main__":
    main()
