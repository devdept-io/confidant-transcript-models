#!/usr/bin/env python3
"""Download all ML models required by the transcribe CLI.

This script populates a local ``models/`` directory with the exact cache
structure expected by HuggingFace Hub and PyTorch, then creates a tar.gz
archive ready for upload to GitHub Releases.

Usage:
    pip install faster-whisper pyannote.audio huggingface-hub
    export HF_TOKEN=hf_...
    python download_models.py          # downloads + creates archive
    python download_models.py --no-archive  # download only
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path("models")


def download_models():
    """Download Whisper and pyannote models into a local cache structure."""
    hf_home = MODELS_DIR / "huggingface"
    torch_home = MODELS_DIR / "torch"

    # Point caches to our local directory
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(torch_home)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
        print("Get a token at: https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    # Login to HuggingFace
    from huggingface_hub import login
    login(token=token)

    # Download Whisper model
    print("Downloading Whisper large-v3-turbo...")
    from faster_whisper import WhisperModel
    WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    print("  ✓ Whisper model downloaded")

    # Download pyannote diarization model (pulls segmentation + speaker embedding too)
    print("Downloading pyannote speaker-diarization-3.1...")
    from pyannote.audio import Pipeline
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    print("  ✓ Pyannote models downloaded")

    # Show what we got
    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"\nTotal models size: {total / 1024 / 1024:.0f} MB")


def create_archive():
    """Create transcribe-models.tar.gz from the models directory."""
    archive = Path("transcribe-models.tar.gz")
    print(f"\nCreating {archive}...")
    subprocess.run(
        ["tar", "czf", str(archive), "-C", str(MODELS_DIR.parent), MODELS_DIR.name],
        check=True,
    )
    size = archive.stat().st_size / 1024 / 1024
    print(f"  ✓ Archive created: {archive} ({size:.0f} MB)")
    return archive


def main():
    parser = argparse.ArgumentParser(description="Download ML models for transcribe CLI")
    parser.add_argument("--no-archive", action="store_true", help="Skip archive creation")
    args = parser.parse_args()

    # Clean previous downloads
    if MODELS_DIR.exists():
        print(f"Removing existing {MODELS_DIR}/...")
        shutil.rmtree(MODELS_DIR)

    MODELS_DIR.mkdir(parents=True)

    download_models()

    if not args.no_archive:
        create_archive()

    print("\nDone!")


if __name__ == "__main__":
    main()
