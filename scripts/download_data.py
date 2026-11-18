"""Download the original public Kaggle archive, or validate an existing copy.

No credential is embedded. If Kaggle requires authentication in your region,
download from the source page in your browser and supply --archive PATH.
"""
import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, save_json, sha256_file

URL = "https://www.kaggle.com/api/v1/datasets/download/usdot/flight-delays"
SOURCE = "https://www.kaggle.com/datasets/usdot/flight-delays"

def download(destination):
    temporary = destination.with_suffix(".part")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "CDS4003-Academic-Project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, 1024 * 1024)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("The response is not a ZIP archive. Download the archive using the Kaggle source page.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

def extract_archive(archive, target):
    """Extract only expected basenames; never trust archive paths."""
    required = {"flights.csv", "airlines.csv", "airports.csv"}
    with zipfile.ZipFile(archive) as handle:
        for name in required:
            matches = [x for x in handle.infolist() if Path(x.filename).name.lower() == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name}; found {len(matches)}")
            with handle.open(matches[0]) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    target = ROOT / "data/raw"
    target.mkdir(parents=True, exist_ok=True)
    archive = args.archive or target / "flight-delays.zip"
    if not archive.is_file():
        if args.archive:
            raise FileNotFoundError(archive)
        try:
            download(archive)
        except (urllib.error.URLError, TimeoutError) as error:
            raise SystemExit(f"Download failed: {error}. Download {SOURCE} and rerun with --archive PATH. No data was fabricated.")
    extract_archive(archive, target)
    manifest = {
        "source_url": SOURCE, "download_url": URL,
        "license_as_listed_by_kaggle": "CC0 Public Domain",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_method": "user_provided_archive" if args.archive else "public_kaggle_download",
        "archive_sha256": sha256_file(archive),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in [target / x for x in ["flights.csv", "airlines.csv", "airports.csv"]]},
        "row_count_status": "Not counted yet; run scripts/prepare_data.py"
    }
    save_json(manifest, ROOT / "reports/download_manifest.json")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
