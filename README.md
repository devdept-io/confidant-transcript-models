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

1. Go to **Actions** → **Release Models**
2. Click **Run workflow**
3. Enter a version tag (e.g. `v1.0.0`)
4. The workflow downloads all models and creates a GitHub Release with `transcribe-models.tar.gz`

Then update `REQUIRED_MODEL_VERSION` in the main repo's `src/config.py` to match.
