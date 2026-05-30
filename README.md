# Note Assistant

Live audio transcription + summarization → Apple Notes. Launched from a Desktop icon.

## Features

- 🎙️ **Audio input**: Microphone or System Audio (via BlackHole)
- 📝 **Live transcription**: Apple Speech (on-device, Neural Engine) or faster-whisper
- ✨ **Live summarization**: Apple Foundation Models (Apple Intelligence) or Ollama
- 🍎 **Apple Notes output**: Real-time updates to a new Note each session
- 🖥️ **Desktop app**: Double-click `NoteAssistant.app` to launch
- ⚙️ **Auto-configured**: setup script detects your macOS and chip

## Requirements

| Tier | macOS | Chip |
|---|---|---|
| Full (Apple Speech + Apple Foundation Models) | 15.1+ | Apple Silicon (M1+) |
| Partial (Apple Speech + Ollama) | 13+ | Any |
| Fallback (faster-whisper + Ollama) | 10.15+ | Any |

## Quick Start

```bash
bash setup_mac.sh
```

That's it. The script will:
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

# Override backends
uv run python -m note_assistant --transcription faster-whisper --whisper-model small

# Auto-detect best backends
uv run python -m note_assistant --auto

# List audio devices
uv run python -m note_assistant devices

# Disable Apple Notes output
uv run python -m note_assistant --no-notes
```

## Configuration (`config.yaml`)

```yaml
audio:
  source: mic            # mic | system
  sample_rate: 16000
  chunk_seconds: 5       # seconds per transcription chunk

transcription:
  backend: apple         # apple | faster-whisper
  whisper_model: base    # tiny | base | small | medium | large-v3

summarization:
  backend: apple         # apple | ollama
  summarize_every: 3     # summarize after N chunks
  ollama_model: llama3.2

output:
  apple_notes: true
  apple_notes_title: "Note Assistant — {date}"
  save_transcript: true
  save_summary: true
  output_dir: ./notes
```

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
