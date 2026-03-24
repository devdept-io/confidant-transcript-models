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

IMPORTANT: HF_HOME and TORCH_HOME must be set BEFORE importing any ML library,
because huggingface_hub caches the directory at import time.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path("models").resolve()


def main():
    parser = argparse.ArgumentParser(description="Download ML models for transcribe CLI")
    parser.add_argument("--no-archive", action="store_true", help="Skip archive creation")
    args = parser.parse_args()

    hf_home = MODELS_DIR / "huggingface"
    torch_home = MODELS_DIR / "torch"

    # ── CRITICAL: set env vars BEFORE any ML library is imported ──
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")  # explicit hub cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")  # legacy alias

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
        print("Get a token at: https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    # Clean previous downloads
    if MODELS_DIR.exists():
        print(f"Removing existing {MODELS_DIR}/...")
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True)

    # ── Now import ML libraries (after env vars are set) ──
    # Force huggingface_hub to use our cache dir, even if it was already resolved
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HOME = str(hf_home)
    huggingface_hub.constants.HF_HUB_CACHE = str(hf_home / "hub")
    huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = str(hf_home / "hub")

    from huggingface_hub import login
    login(token=token)

    print(f"  HF cache dir: {huggingface_hub.constants.HF_HUB_CACHE}")
    print(f"  TORCH_HOME: {os.environ.get('TORCH_HOME')}")

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

    # ── Verify everything ended up in our local cache ──
    # Some pyannote/speechbrain versions ignore TORCH_HOME and write to ~/.cache/torch/
    default_torch_cache = Path.home() / ".cache" / "torch" / "pyannote"
    local_torch_pyannote = torch_home / "pyannote"
    if default_torch_cache.exists() and not local_torch_pyannote.exists():
        print(f"  Copying pyannote torch cache from {default_torch_cache} → {local_torch_pyannote}")
        shutil.copytree(default_torch_cache, local_torch_pyannote)

    # Verify pyannote models are in torch cache (pyannote stores them there, not in HF hub)
    pyannote_models = [
        "models--pyannote--speaker-diarization-3.1",
        "models--pyannote--segmentation-3.0",
        "models--pyannote--wespeaker-voxceleb-resnet34-LM",
    ]
    all_found = True
    for model_name in pyannote_models:
        torch_path = torch_home / "pyannote" / model_name
        if torch_path.exists():
            print(f"  ✓ Found {model_name} (torch)")
        else:
            print(f"  ✗ Missing {model_name} in torch cache")
            all_found = False

    if not all_found:
        print("\nERROR: Some pyannote models are missing from torch cache.", file=sys.stderr)
        print("Contents of torch_home:", file=sys.stderr)
        for p in sorted(torch_home.rglob("*"))[:30]:
            print(f"  {p.relative_to(torch_home)}", file=sys.stderr)
        sys.exit(1)

    # Show size breakdown
    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    hf_size = sum(f.stat().st_size for f in hf_home.rglob("*") if f.is_file()) if hf_home.exists() else 0
    torch_size = sum(f.stat().st_size for f in torch_home.rglob("*") if f.is_file()) if torch_home.exists() else 0
    print(f"\nTotal models size: {total / 1024 / 1024:.0f} MB")
    print(f"  huggingface/: {hf_size / 1024 / 1024:.0f} MB")
    print(f"  torch/: {torch_size / 1024 / 1024:.0f} MB")

    # Create archive
    if not args.no_archive:
        archive = Path("transcribe-models.tar.gz")
        print(f"\nCreating {archive}...")
        subprocess.run(
            ["tar", "czf", str(archive), "-C", str(MODELS_DIR.parent), MODELS_DIR.name],
            check=True,
        )
        size = archive.stat().st_size / 1024 / 1024
        print(f"  ✓ Archive created: {archive} ({size:.0f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
