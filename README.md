# Note Assistant

Live audio transcription + summarization → Apple Notes. Launched from a Desktop icon.

## Features

- 🎙️ **Audio input**: Microphone, System Audio (via BlackHole), or audio/video file
- 📝 **Live transcription**: Apple Speech (on-device, Neural Engine), MLX Whisper (Apple Silicon), or faster-whisper
- ✨ **Live summarization**: Apple Foundation Models (`apple-fm-sdk`), MLX (on-device), or Ollama — with automatic fallback on failure
- 🌐 **Language support**: Transcribe in one language, summarize in another (English, Thai, Japanese, Chinese, Auto-detect)
- 🍎 **Apple Notes output**: Real-time updates to a new Note each session, with auto-generated title
- 🖥️ **Desktop app**: Double-click `NoteAssistant.app` to launch
- ⚙️ **Settings UI**: Configure audio source, backends, and language before each session
- 🔄 **Restart without relaunching**: Return to settings and start a fresh session from the done screen

## Requirements

| Tier | macOS | Chip | Backends |
|---|---|---|---|
| Full | 26+ | Apple Silicon (M1+) | Apple Speech + Apple Foundation Models |
| Partial | 13–25 | Any | Apple Speech + Ollama |
| Fallback | < 13 | Any | faster-whisper + Ollama |

MLX Whisper and MLX summarization can be used as alternative on-device options on Apple Silicon at any supported macOS version.

## Quick Start

```bash
bash scripts/setup_mac.sh
```

The script will:
1. Detect your macOS version and chip
2. Write `config.yaml` with the best defaults for your system
3. Install `uv` and all Python dependencies
4. Install BlackHole (system audio capture)
5. Build `NoteAssistant.app` and place it on your Desktop

## Rebuild the App

To rebuild `NoteAssistant.app` after changing the project directory or config:

```bash
bash scripts/build_app.sh
```

## Manual Usage

```bash
# Default (reads config.yaml, opens settings UI)
uv run python -m note_assistant

# Override audio source
uv run python -m note_assistant --source system

# Transcribe from an audio file (set path in TUI or config.yaml)
uv run python -m note_assistant --source file

# Override transcription backend
uv run python -m note_assistant --transcription mlx-whisper
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
  source: mic           # mic | system | file
  file_path: null       # path to audio file when source is "file"
  sample_rate: 16000
  chunk_seconds: 5.0

transcription:
  backend: apple        # apple | faster-whisper | mlx-whisper
  language: null        # null = auto-detect; e.g. "en", "ja" to force
  whisper_model: base   # tiny | base | small | medium | large-v3
  mlx_whisper_model: mlx-community/whisper-base-mlx
  device: auto          # auto | cpu | mps | cuda

summarization:
  backend: apple        # apple | mlx | ollama
  summarize_every: 3    # summarize after N transcript chunks
  prompt_template: |
    Summarize the following transcript into concise bullet-point notes.
    Write each bullet point on its own line starting with '- '.

    {transcript}
  mlx_model: mlx-community/Qwen3-8B-4bit
  mlx_fallback_model: mlx-community/gemma-4-e4b-it-OptiQ-4bit
  ollama_model: qwen3:8b
  ollama_fallback_model: qwen3:4b
  ollama_host: http://localhost:11434

output:
  apple_notes: true
  apple_notes_title: "Note Assistant — {date}"
  auto_title: true              # generate a short title from the summary via LLM
  title_prompt_template: |
    Generate a concise, informative title of 5 words or less for these notes.
    Write the title in {language}.
    Reply with ONLY the title — no quotes, no punctuation at the end:

    {summary}
  save_transcript: true
  save_summary: true
  output_dir: ./notes

language_input: English   # English | Thai | Japanese | Chinese | Auto
language_output: English  # English | Thai | Japanese | Chinese

log_level: WARNING        # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

Logs are written to `note_assistant.log` in the current directory.

## Backend Resilience

If a summarization backend produces repeated empty or identical responses, it is automatically skipped and the next configured backend takes over (e.g. MLX → Ollama). The backend resets to primary on the next session.

## System Audio Setup (BlackHole)

1. Run `scripts/setup_mac.sh` (installs BlackHole automatically)
2. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
3. Click **+** → **Create Aggregate Device**
4. Check **BlackHole 2ch** + your microphone
5. Set new Aggregate Device as input in **System Settings → Sound**
6. Run with `--source system`

## Keyboard Shortcuts (in TUI)

| Key | Action |
|---|---|
| `Ctrl+Q` | Quit and save notes |
| `Ctrl+P` | Pause / Resume |
| `⏹ Stop` button | Stop recording — shows Restart or Quit options |
