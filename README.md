# Note Assistant

Live audio transcription + summarization → Apple Notes. Launched from a Desktop icon.

## Features

- 🎙️ **Audio input**: Microphone, System Audio (via BlackHole), or audio file
- 📝 **Live transcription**: Apple Speech (on-device, Neural Engine) or faster-whisper
- ✨ **Live summarization**: Apple Foundation Models (`apple-fm-sdk`), MLX (on-device), or Ollama
- 🌐 **Language support**: Transcribe in one language, summarize in another
- 🍎 **Apple Notes output**: Real-time updates to a new Note each session
- 🖥️ **Desktop app**: Double-click `NoteAssistant.app` to launch
- ⚙️ **Auto-configured**: setup script detects your macOS version and chip

## Requirements

| Tier | macOS | Chip | Backends |
|---|---|---|---|
| Full | 26+ | Apple Silicon (M1+) | Apple Speech + Apple Foundation Models |
| Partial | 13–25 | Any | Apple Speech + Ollama |
| Fallback | 10.15+ | Any | faster-whisper + Ollama |

MLX (`summarization.backend: mlx`) can be used as an alternative on-device summarizer on Apple Silicon at any supported macOS version.

## Quick Start

```bash
bash setup_mac.sh
```

The script will:
1. Detect your macOS version and chip
2. Write `config.yaml` with the best defaults for your system
3. Install `uv` and all Python dependencies
4. Install BlackHole (system audio capture)
5. Build `NoteAssistant.app` and place it on your Desktop

## Manual Usage

```bash
# Default (reads config.yaml)
uv run python -m note_assistant

# Override audio source
uv run python -m note_assistant --source system

# Transcribe from an audio file (set path in TUI or config.yaml)
uv run python -m note_assistant --source file

# Override backends
uv run python -m note_assistant --transcription faster-whisper --whisper-model small

# Override summarization backend
uv run python -m note_assistant --summarization mlx

# Auto-detect best backends for this machine
uv run python -m note_assistant --auto

# List audio devices
uv run python -m note_assistant devices

# Disable Apple Notes output
uv run python -m note_assistant --no-notes

# Set log level (DEBUG | INFO | WARNING | ERROR)
uv run python -m note_assistant --log-level DEBUG
```

## Configuration (`config.yaml`)

```yaml
audio:
  source: mic            # mic | system | file
  file_path:             # absolute path, required when source is file
  sample_rate: 16000
  chunk_seconds: 2       # seconds per transcription chunk

transcription:
  backend: apple         # apple | faster-whisper
  whisper_model: base    # tiny | base | small | medium | large-v3

summarization:
  backend: apple         # apple | mlx | ollama
  summarize_every: 3     # summarize after N transcript chunks
  mlx_model: mlx-community/Llama-3.2-3B-Instruct-4bit
  ollama_model: llama3.2:3b
  ollama_host: http://localhost:11434

output:
  apple_notes: true
  apple_notes_title: "Note Assistant — {date}"
  save_transcript: true
  save_summary: true
  output_dir: ./notes

language_input: English   # language of spoken audio
language_output: English  # language of the generated summary

log_level: WARNING        # DEBUG | INFO | WARNING | ERROR
```

Logs are written to `note_assistant.log` in the current directory.

## System Audio Setup (BlackHole)

1. Run `setup_mac.sh` (installs BlackHole automatically)
2. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
3. Click **+** → **Create Aggregate Device**
4. Check **BlackHole 2ch** + your microphone
5. Set new Aggregate Device as input in **System Settings → Sound**
6. Run with `--source system`

## Keyboard Shortcuts (in TUI)

| Key | Action |
|---|---|
| `Ctrl+C` | Quit and save notes |
| `Ctrl+P` | Pause / Resume |
