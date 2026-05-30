#!/usr/bin/env bash
# =============================================================================
# Build NoteAssistant.app — shell wrapper bundle, no py2app needed.
# Places the .app on ~/Desktop.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="NoteAssistant"
APP_DIR="$SCRIPT_DIR/${APP_NAME}.app"
DESKTOP="$HOME/Desktop"
VENV="$SCRIPT_DIR/.venv"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[info]${NC} $*"; }
success() { echo -e "${GREEN}[✓]${NC} $*"; }

# Clean existing
rm -rf "$APP_DIR"

# Create bundle structure
info "Creating app bundle structure..."
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# ── Info.plist ──────────────────────────────────────────────────────────────
cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>             <string>NoteAssistant</string>
    <key>CFBundleDisplayName</key>      <string>Note Assistant</string>
    <key>CFBundleIdentifier</key>       <string>com.local.note-assistant</string>
    <key>CFBundleVersion</key>          <string>1.0</string>
    <key>CFBundleExecutable</key>       <string>NoteAssistant</string>
    <key>CFBundleIconFile</key>         <string>NoteAssistant</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>LSMinimumSystemVersion</key>   <string>13.0</string>
    <key>NSHighResolutionCapable</key>  <true/>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>Note Assistant needs microphone access for live transcription.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Note Assistant records audio for live transcription.</string>
</dict>
</plist>
PLIST

# ── Launcher shell script ───────────────────────────────────────────────────
cat > "$APP_DIR/Contents/MacOS/NoteAssistant" <<LAUNCHER
#!/usr/bin/env bash
# Note Assistant launcher — activates uv venv and runs the app
PROJECT="$SCRIPT_DIR"
VENV="\$PROJECT/.venv"

# Open a new Terminal window with the app running
osascript <<EOF
tell application "Terminal"
    do script "source '\$VENV/bin/activate' && cd '\$PROJECT' && python -m note_assistant; exec bash"
    activate
end tell
EOF
LAUNCHER

chmod +x "$APP_DIR/Contents/MacOS/NoteAssistant"

# ── Icon ─────────────────────────────────────────────────────────────────────
# Generate a simple icon using Python (no external tool needed)
ICON_PY="$SCRIPT_DIR/.venv/bin/python"
if [ -f "$ICON_PY" ] && python3 -c "import PIL" &>/dev/null 2>&1; then
    python3 - <<PYICON
from PIL import Image, ImageDraw, ImageFont
import os, struct

def make_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    r = size - margin
    draw.ellipse([margin, margin, r, r], fill=(52, 120, 246, 255))
    emoji_size = size // 2
    draw.text((size // 2, size // 2), "📝", anchor="mm",
              font=ImageFont.load_default(size=emoji_size))
    return img

iconset = "$SCRIPT_DIR/NoteAssistant.iconset"
os.makedirs(iconset, exist_ok=True)
for s in [16, 32, 64, 128, 256, 512]:
    make_icon(s).save(f"{iconset}/icon_{s}x{s}.png")
    make_icon(s*2).save(f"{iconset}/icon_{s}x{s}@2x.png")
PYICON

    if command -v iconutil &>/dev/null; then
        iconutil -c icns "$SCRIPT_DIR/NoteAssistant.iconset" \
            -o "$APP_DIR/Contents/Resources/NoteAssistant.icns" 2>/dev/null || true
        rm -rf "$SCRIPT_DIR/NoteAssistant.iconset"
    fi
fi

# ── Place on Desktop ─────────────────────────────────────────────────────────
info "Placing NoteAssistant.app on Desktop..."
cp -r "$APP_DIR" "$DESKTOP/${APP_NAME}.app"

# Touch to refresh Finder
touch "$DESKTOP/${APP_NAME}.app"

success "NoteAssistant.app is on your Desktop!"
success "Double-click it to launch Note Assistant."
