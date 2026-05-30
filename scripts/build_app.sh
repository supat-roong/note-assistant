#!/usr/bin/env bash
# =============================================================================
# Build NoteAssistant.app — shell wrapper bundle, no py2app needed.
# Places the .app on ~/Desktop.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="NoteAssistant"
APP_DIR="$PROJECT_DIR/${APP_NAME}.app"
DESKTOP="$HOME/Desktop"
VENV="$PROJECT_DIR/.venv"
MEDIA_DIR="$PROJECT_DIR/media"
mkdir -p "$MEDIA_DIR"

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
PROJECT="$PROJECT_DIR"
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
ICONSET="$MEDIA_DIR/NoteAssistant.iconset"
SOURCE_PNG="$MEDIA_DIR/NoteAssistant.png"
PROCESSED_PNG="$MEDIA_DIR/.NoteAssistant_processed.png"

if [ -f "$SOURCE_PNG" ]; then
    info "Generating NoteAssistant.icns from media/NoteAssistant.png..."

    # Pre-process: crop to content bounding box, fill transparent areas with dark background
    ICON_SOURCE="$SOURCE_PNG"
    if [ -f "$PROJECT_DIR/.venv/bin/python" ] && \
       "$PROJECT_DIR/.venv/bin/python" -c "import PIL" &>/dev/null 2>&1; then
        "$PROJECT_DIR/.venv/bin/python" - <<PYPROCESS
from PIL import Image
img = Image.open("$SOURCE_PNG").convert("RGBA")
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)
side = max(img.size)
bg = Image.new("RGBA", (side, side), (13, 17, 23, 255))
bg.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
bg.convert("RGB").save("$PROCESSED_PNG")
PYPROCESS
        ICON_SOURCE="$PROCESSED_PNG"
    fi

    rm -rf "$ICONSET" && mkdir -p "$ICONSET"
    for size in 16 32 128 256 512; do
        sips -z $size $size             "$ICON_SOURCE" --out "$ICONSET/icon_${size}x${size}.png"      &>/dev/null
        sips -z $((size*2)) $((size*2)) "$ICON_SOURCE" --out "$ICONSET/icon_${size}x${size}@2x.png"   &>/dev/null
    done
    rm -f "$PROCESSED_PNG"
    iconutil -c icns "$ICONSET" -o "$MEDIA_DIR/NoteAssistant.icns"
    rm -rf "$ICONSET"
    cp "$MEDIA_DIR/NoteAssistant.icns" "$APP_DIR/Contents/Resources/NoteAssistant.icns"
    success "Icon generated from media/NoteAssistant.png"
else
    warn "No media/NoteAssistant.png found — app will use default macOS icon."
fi

# ── Place on Desktop ─────────────────────────────────────────────────────────
info "Placing NoteAssistant.app on Desktop..."
# Remove first so Finder treats it as a new app and busts its icon cache
rm -rf "$DESKTOP/${APP_NAME}.app"
cp -r "$APP_DIR" "$DESKTOP/${APP_NAME}.app"

# Force Finder to reload the icon cache for this app
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$DESKTOP/${APP_NAME}.app" 2>/dev/null || true
killall Finder 2>/dev/null || true

success "NoteAssistant.app is on your Desktop!"
success "Double-click it to launch Note Assistant."
