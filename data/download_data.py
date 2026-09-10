# -*- coding: utf-8 -*-
"""
============================================================
data/download_data.py
============================================================
Fetches the real Crop Recommendation dataset and verifies it.

This replaces the previous `generate_data.py`, which synthesised
rows from hand-written per-crop Gaussians. That generator drew
every feature independently, which is precisely the generative
model GaussianNB assumes -- so any accuracy measured on it scored
the generator, not agronomy.

Source
------
The dataset is the widely used "Crop Recommendation Dataset"
(Atharva Ingle, Kaggle). Kaggle requires authentication, so this
script pulls a byte-identical public mirror and verifies its MD5
before writing anything.

Run: python data/download_data.py
============================================================
"""

import argparse
import hashlib
import os
import sys
import urllib.request

HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_PATH  = os.path.join(HERE, "crop_data.csv")

# Public mirror of the Kaggle dataset.
SOURCE_URL = (
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/"
    "master/Data-processed/crop_recommendation.csv"
)

# Checksum of the file this project was built and evaluated against.
# A mismatch means the upstream mirror changed: stop rather than silently
# retrain on different data.
EXPECTED_MD5 = "fa83739710074b21a20b156034538279"

EXPECTED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
EXPECTED_ROWS    = 2200
EXPECTED_CLASSES = 22


def fetch(url: str) -> bytes:
    """Download the dataset and return its raw bytes."""
    print(f"[DATA] Fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Download failed with HTTP {response.status}")
        return response.read()


def verify(payload: bytes, allow_checksum_mismatch: bool = False) -> None:
    """
    Verify the payload before it is written to disk.

    Checks the checksum, then the structure. Structure is checked even when
    the checksum is waived, so a mirror that starts serving an HTML error
    page can never be mistaken for a dataset.
    """
    actual_md5 = hashlib.md5(payload).hexdigest()
    if actual_md5 != EXPECTED_MD5:
        message = (
            f"Checksum mismatch.\n"
            f"  expected {EXPECTED_MD5}\n"
            f"  actual   {actual_md5}\n"
            f"The upstream mirror has changed. Re-run with "
            f"--allow-checksum-mismatch only after inspecting the new file."
        )
        if not allow_checksum_mismatch:
            raise RuntimeError(message)
        print(f"[WARN] {message}")
    else:
        print(f"[OK]   Checksum verified: {actual_md5}")

    text   = payload.decode("utf-8")
    lines  = [ln for ln in text.splitlines() if ln.strip()]
    header = [c.strip() for c in lines[0].split(",")]

    if header != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected columns: {header}")

    n_rows = len(lines) - 1
    labels = {ln.rsplit(",", 1)[1].strip() for ln in lines[1:]}

    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {n_rows}")
    if len(labels) != EXPECTED_CLASSES:
        raise RuntimeError(f"Expected {EXPECTED_CLASSES} classes, found {len(labels)}")

    print(f"[OK]   Structure verified: {n_rows} rows x {len(header)} cols, "
          f"{len(labels)} crops")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the crop dataset")
    parser.add_argument(
        "--allow-checksum-mismatch",
        action="store_true",
        help="Proceed even if the mirror's checksum differs (inspect it first)",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists")
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f"[DATA] Already present: {OUT_PATH} (use --force to re-download)")
        return 0

    payload = fetch(SOURCE_URL)
    verify(payload, allow_checksum_mismatch=args.allow_checksum_mismatch)

    # Write only after every check has passed.
    with open(OUT_PATH, "wb") as handle:
        handle.write(payload)

    print(f"[OK]   Dataset written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
