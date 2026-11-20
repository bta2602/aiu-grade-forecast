"""Fetch the three public, version-pinned CSV files used to reproduce the experiment."""

import argparse
import hashlib
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "public_kz"
REVISION = "8583bfb15a7ab8f8f8d308ea01004f5e54c64924"
BASE_URL = (
    f"https://raw.githubusercontent.com/Mikhail31415/cgpa-prediction/{REVISION}/"
    "data/final_grade_prediction_dataset"
)
CHECKSUMS = {
    "train.csv": "1d62b003b0cfe606989e4cd037f215df53b29ccf081d317e41314a470279a771",
    "val.csv": "7e2c8de824bc759d7e79275a0cbe6a3d47a55f61efdc78c89f768990fdea5589",
    "test.csv": "0432f35ce9c8e8e6f851a8795efeb26906f8af138536a6dcd5aa32701dc9d294",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_file(directory: Path, name: str, digest: str) -> None:
    target = directory / name
    if target.is_file() and sha256(target) == digest:
        print(f"{name}: verified")
        return

    request = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "aiu-grade-forecast"})
    for attempt in range(3):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as output:
                temp_path = Path(output.name)
                with urllib.request.urlopen(request, timeout=40) as response:
                    shutil.copyfileobj(response, output)
            if sha256(temp_path) != digest:
                raise ValueError(f"{name}: unexpected SHA-256 for revision {REVISION}")
            os.replace(temp_path, target)
            print(f"{name}: downloaded and verified")
            return
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    directory = parser.parse_args().data_dir
    directory.mkdir(exist_ok=True, parents=True)
    for name, digest in CHECKSUMS.items():
        download_file(directory, name, digest)


if __name__ == "__main__":
    main()
