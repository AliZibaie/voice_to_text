# VoiceToText

Local desktop app that converts voice notes (and other audio/video files) into **Persian transcript text**. Everything runs on your machine — no cloud transcription API.

## Features

- Drag and drop one or many voice files; each file gets its own result card
- Live progress while Faster-Whisper listens to the file
- Model picker: **small** (fast), **medium** (accurate), **large-v3** (most accurate)
- Automatic fallback to the local **small** model when the selected model is not fully downloaded
- Copy / save transcripts, queue multiple files, RTL Persian UI
- Optional WebView2 desktop shell (`shell/VoiceToText.exe`) with Edge / browser fallback

## Requirements

- Windows 10/11
- Python 3.10+ (tested with 3.13)
- Enough disk space for Whisper weights (small ≈ 500 MB; medium/large are larger)
- Microsoft Edge (fallback window) or the bundled WebView2 shell

## Install

```bash
cd VoiceToText
python -m pip install -r requirements.txt
```

On first use, Faster-Whisper downloads the selected model into your Hugging Face cache:

`%USERPROFILE%\.cache\huggingface\hub\`

If a download was interrupted (`.incomplete` files / missing `model.bin`), pick **small** or re-download the model before using medium/large.

## Run

```bash
python app.py
```

The app:

1. Starts a local HTTP server on `127.0.0.1` (random free port)
2. Writes the port to `%TEMP%\pardaz-voice-port.txt`
3. Opens the desktop shell if present, otherwise Edge app mode / the default browser
4. Reuses an already-running instance when you launch again

Set `VOICETOTEXT_NO_WINDOW=1` to start the server without opening a UI window.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Server, transcription pipeline, and embedded RTL web UI |
| `requirements.txt` | Python dependencies |
| `fonts/Estedad.woff2` | UI font |
| `logo.png`, `app.ico` | Branding / window icon |
| `shell/VoiceToText.exe` | Optional WebView2 host that wraps the local UI |
| `shell/*.dll` | WebView2 runtime bindings used by the host |

`shell/host.json` is generated at runtime with absolute paths for the current machine and is ignored by git.

## How it works

```text
UI (drag/drop or file picker)
        │  POST /transcribe?model=...&name=...
        ▼
Local HTTP server (app.py)
        │  save upload to %TEMP%\pardaz-voice-in\
        ▼
Faster-Whisper (CPU, int8) + PyAV decode
        │  NDJSON stream: {progress} … {done, text, duration, fallback}
        ▼
Result card in the UI
```

### Key pieces in `app.py`

- **`_open_compat`** — patches `av.open` so newer PyAV builds still work with Faster-Whisper’s `metadata_errors` argument
- **`weights_dir` / `resolve_model`** — locate a complete local snapshot; fall back to `small` when needed
- **`load_model`** — caches one Whisper model in memory
- **`transcribe`** — Persian (`fa`) transcription with VAD, beam search, and progress callbacks
- **`Handler`** — serves the HTML UI, fonts, logo, health/shutdown, and chunked NDJSON transcription
- **`install_shell` / `open_window`** — register and launch the desktop host or Edge
- **`main`** — single-instance server bootstrap

### HTTP endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Embedded web UI |
| `GET` | `/health` | Liveness check |
| `GET` | `/shutdown` | Stop the process |
| `GET` | `/logo.png`, `/fonts/...` | Static assets |
| `POST` | `/transcribe?model=&name=` | Raw file body → NDJSON progress + final text |

## Usage tips

1. Drop `.ogg`, `.opus`, `.mp3`, `.wav`, `.m4a`, `.webm`, `.flac`, `.aac`, or video files that contain audio
2. Prefer **small** until larger models are fully cached locally
3. Use **Copy** / **Save** after transcription finishes
4. Close via the UI **Close** button (calls `/shutdown`) or stop the Python process

## License notes

- Application code: use and modify as needed for your project
- Estedad font: see `fonts/OFL.txt` (SIL Open Font License)
- Whisper / Faster-Whisper: follow their upstream licenses

## Push to a remote

This repository is initialized with an initial commit and **no remote**. Add yours and push:

```bash
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
