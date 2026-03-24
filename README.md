# Confidant Transcriber Models

ML models for the [confidant-transcriber](https://github.com/devdept-io/confidant-transcriber) CLI.

This is a public repository so the transcribe executable can download models without authentication.

## Models included

| Model | Purpose | Size |
|-------|---------|------|
| [faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | Speech-to-text | ~1.5 GB |
| [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | Speaker diarization | ~50 MB |
| [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | Voice activity detection | ~20 MB |
| [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | Speaker embedding | ~50 MB |

## Releasing a new version

Every push to `main` runs **Release Models** and publishes a GitHub Release with a **semver tag** (`v1.0.0`, `v1.0.1`, …): the workflow looks at existing `v*.*.*` tags, takes the highest, and bumps the patch. If no such tags exist yet, it starts at `v1.0.0`.

You can also trigger the workflow manually under **Actions** → **Release Models** → **Run workflow**. Leave the version field empty for the same automatic semver, or set a tag explicitly (e.g. `v2.0.0` for a major/minor bump).

Then update `REQUIRED_MODEL_VERSION` in the main repo's `src/config.py` to match.
