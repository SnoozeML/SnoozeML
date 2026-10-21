#!/bin/sh
# SnoozeML Zero-Sudo Installer for Linux & macOS
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --cpu
#   curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --no-service
#   curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --uninstall

set -eu

REPO="${SNOOZE_REPO:-SnoozeML/SnoozeML}"
VERSION="${SNOOZE_VERSION:-latest}"
VARIANT="${SNOOZE_VARIANT:-standard}"
INSTALL_ROOT="${SNOOZE_INSTALL_DIR:-$HOME/.local/share/snoozeml}"
BIN_DIR="${SNOOZE_BIN_DIR:-$HOME/.local/bin}"
MODELS_DIR="${SNOOZE_MODELS_DIR:-$INSTALL_ROOT/models}"
SETUP_SERVICE="${SNOOZE_NO_SERVICE:-0}"
ACTION="install"

# ANSI colors (if terminal attached)
if [ -t 1 ]; then
    BOLD="\033[1m"
    GREEN="\033[32m"
    CYAN="\033[36m"
    YELLOW="\033[33m"
    RED="\033[31m"
    RESET="\033[0m"
else
    BOLD=""
    GREEN=""
    CYAN=""
    YELLOW=""
    RED=""
    RESET=""
fi

info() {
    printf "${CYAN}==>${RESET} ${BOLD}%s${RESET}\n" "$1"
}

success() {
    printf "${GREEN}✓${RESET} %s\n" "$1"
}

warn() {
    printf "${YELLOW}!${RESET} %s\n" "$1" >&2
}

error() {
    printf "${RED}✗ Error:${RESET} %s\n" "$1" >&2
    exit 1
}

usage() {
    cat <<EOF
SnoozeML Installer (Rootless / Zero-Sudo)

Usage:
  install.sh [OPTIONS]

Options:
  --version <TAG>       Install a specific release tag (e.g. v0.1.0, default: latest)
  --cpu                 Install the lightweight CPU-only variant (*-cpu)
  --install-dir <DIR>   Installation directory (default: ~/.local/share/snoozeml)
  --bin-dir <DIR>       Symlink binary directory (default: ~/.local/bin)
  --models-dir <DIR>    Default models directory (default: ~/.local/share/snoozeml/models)
  --no-service          Do not create or start the user-mode background daemon
  --uninstall           Stop user daemon and remove SnoozeML binary & runtime files
  -h, --help            Print this help message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || error "--version requires a value"
            VERSION="$2"
            shift 2
            ;;
        --cpu)
            VARIANT="cpu"
            shift
            ;;
        --install-dir)
            [ $# -ge 2 ] || error "--install-dir requires a value"
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --bin-dir)
            [ $# -ge 2 ] || error "--bin-dir requires a value"
            BIN_DIR="$2"
            shift 2
            ;;
        --models-dir)
            [ $# -ge 2 ] || error "--models-dir requires a value"
            MODELS_DIR="$2"
            shift 2
            ;;
        --no-service)
            SETUP_SERVICE="1"
            shift
            ;;
        --uninstall)
            ACTION="uninstall"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1 (run with --help for usage)"
            ;;
    esac
done

uninstall_snooze() {
    info "Uninstalling SnoozeML..."

    # Stop and remove Linux systemd user service if present
    SERVICE_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/snooze.service"
    if command -v systemctl >/dev/null 2>&1 && [ -f "$SERVICE_FILE" ]; then
        systemctl --user stop snooze.service >/dev/null 2>&1 || true
        systemctl --user disable snooze.service >/dev/null 2>&1 || true
        rm -f "$SERVICE_FILE"
        systemctl --user daemon-reload >/dev/null 2>&1 || true
        success "Removed systemd user service ($SERVICE_FILE)"
    fi

    # Stop and remove macOS LaunchAgent if present
    PLIST_FILE="$HOME/Library/LaunchAgents/io.snoozeml.daemon.plist"
    if [ -f "$PLIST_FILE" ]; then
        launchctl unload "$PLIST_FILE" >/dev/null 2>&1 || true
        rm -f "$PLIST_FILE"
        success "Removed macOS LaunchAgent ($PLIST_FILE)"
    fi

    rm -f "$BIN_DIR/snooze"
    rm -rf "$INSTALL_ROOT/bin" "$INSTALL_ROOT/lib"

    success "Removed SnoozeML binary from $BIN_DIR/snooze"
    if [ -d "$MODELS_DIR" ]; then
        info "Preserved your models directory at $MODELS_DIR (delete manually if no longer needed)."
    fi
    exit 0
}

if [ "$ACTION" = "uninstall" ]; then
    uninstall_snooze
fi

detect_platform() {
    OS_NAME="$(uname -s)"
    ARCH_NAME="$(uname -m)"

    case "$OS_NAME" in
        Linux)
            PLATFORM_OS="linux"
            ;;
        Darwin)
            PLATFORM_OS="macos"
            ;;
        *)
            error "Unsupported operating system: $OS_NAME. On Windows, use install.ps1."
            ;;
    esac

    case "$ARCH_NAME" in
        x86_64|amd64)
            PLATFORM_ARCH="x86_64"
            ;;
        aarch64|arm64)
            PLATFORM_ARCH="aarch64"
            ;;
        *)
            error "Unsupported CPU architecture: $ARCH_NAME"
            ;;
    esac

    if [ "$PLATFORM_OS" = "macos" ] && [ "$PLATFORM_ARCH" = "x86_64" ]; then
        error "Prebuilt macOS binaries target Apple Silicon (aarch64). On Intel Macs, install via: cargo install --git https://github.com/$REPO"
    fi

    if [ "$VARIANT" = "cpu" ]; then
        ARTIFACT_NAME="snooze-${PLATFORM_OS}-${PLATFORM_ARCH}-cpu"
    else
        ARTIFACT_NAME="snooze-${PLATFORM_OS}-${PLATFORM_ARCH}"
    fi
}

download_file() {
    URL="$1"
    DEST="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$URL" -o "$DEST"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$DEST" "$URL"
    else
        error "Neither 'curl' nor 'wget' was found. Please install curl and try again."
    fi
}

verify_checksum() {
    ARCHIVE_PATH="$1"
    CHECKSUM_PATH="$2"

    EXPECTED_HASH="$(awk '{print $1}' "$CHECKSUM_PATH")"
    if [ -z "$EXPECTED_HASH" ]; then
        warn "Checksum file was empty; skipping verification."
        return 0
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        ACTUAL_HASH="$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        ACTUAL_HASH="$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')"
    else
        warn "Neither sha256sum nor shasum found; skipping SHA-256 verification."
        return 0
    fi

    if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
        error "SHA-256 checksum mismatch! Expected $EXPECTED_HASH, got $ACTUAL_HASH"
    fi
    success "Verified SHA-256 checksum ($ACTUAL_HASH)"
}

configure_linux_user_service() {
    if ! command -v systemctl >/dev/null 2>&1; then
        warn "systemctl not found; skipping background service setup."
        return 0
    fi

    SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    SERVICE_FILE="$SYSTEMD_USER_DIR/snooze.service"
    mkdir -p "$SYSTEMD_USER_DIR"

    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=SnoozeML High-Performance Inference Daemon (User Service)
Documentation=https://github.com/$REPO
After=network-online.target

[Service]
Type=simple
ExecStart=$BIN_DIR/snooze serve --models-dir $MODELS_DIR
Restart=on-failure
RestartSec=5
Environment="RUST_LOG=info"
Environment="SNOOZE_MODELS_DIR=$MODELS_DIR"

[Install]
WantedBy=default.target
EOF

    if systemctl --user daemon-reload >/dev/null 2>&1; then
        if systemctl --user enable --now snooze.service >/dev/null 2>&1; then
            success "Started rootless systemd user daemon (snooze.service)"
            printf "  • View logs:   ${BOLD}journalctl --user -u snooze -f${RESET}\n"
            printf "  • Stop daemon: ${BOLD}systemctl --user stop snooze${RESET}\n"
        else
            warn "Created $SERVICE_FILE, but could not auto-start service. Start manually with: systemctl --user start snooze"
        fi
    else
        warn "Created $SERVICE_FILE (systemd user bus is inactive in this session/container). Run 'snooze serve --models-dir $MODELS_DIR' directly."
    fi
}

configure_macos_launch_agent() {
    LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$LAUNCH_AGENTS_DIR/io.snoozeml.daemon.plist"
    LOG_DIR="$INSTALL_ROOT/logs"
    mkdir -p "$LAUNCH_AGENTS_DIR" "$LOG_DIR"

    cat > "$PLIST_FILE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.snoozeml.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BIN_DIR/snooze</string>
        <string>serve</string>
        <string>--models-dir</string>
        <string>$MODELS_DIR</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>RUST_LOG</key>
        <string>info</string>
        <key>SNOOZE_MODELS_DIR</key>
        <string>$MODELS_DIR</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/snooze.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/snooze.err.log</string>
</dict>
</plist>
EOF

    launchctl unload "$PLIST_FILE" >/dev/null 2>&1 || true
    if launchctl load "$PLIST_FILE" >/dev/null 2>&1; then
        success "Started rootless macOS LaunchAgent (io.snoozeml.daemon)"
        printf "  • View logs:   ${BOLD}tail -f $LOG_DIR/snooze.log${RESET}\n"
        printf "  • Stop daemon: ${BOLD}launchctl unload $PLIST_FILE${RESET}\n"
    else
        warn "Created $PLIST_FILE. Run 'snooze serve --models-dir $MODELS_DIR' to start."
    fi
}

detect_platform

if [ "$VERSION" = "latest" ]; then
    BASE_URL="https://github.com/$REPO/releases/latest/download"
else
    BASE_URL="https://github.com/$REPO/releases/download/$VERSION"
fi

ARCHIVE_URL="${BASE_URL}/${ARTIFACT_NAME}.tar.gz"
CHECKSUM_URL="${BASE_URL}/${ARTIFACT_NAME}.tar.gz.sha256"

TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t snoozeml)"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

info "Downloading SnoozeML ($ARTIFACT_NAME, version: $VERSION)..."
download_file "$ARCHIVE_URL" "$TMP_DIR/${ARTIFACT_NAME}.tar.gz"

if download_file "$CHECKSUM_URL" "$TMP_DIR/${ARTIFACT_NAME}.tar.gz.sha256" 2>/dev/null; then
    verify_checksum "$TMP_DIR/${ARTIFACT_NAME}.tar.gz" "$TMP_DIR/${ARTIFACT_NAME}.tar.gz.sha256"
else
    warn "Checksum file not found at release URL; proceeding with extraction."
fi

info "Installing to $INSTALL_ROOT (no sudo required)..."
tar -xzf "$TMP_DIR/${ARTIFACT_NAME}.tar.gz" -C "$TMP_DIR"

mkdir -p "$INSTALL_ROOT/bin" "$INSTALL_ROOT/lib" "$BIN_DIR" "$MODELS_DIR"
cp "$TMP_DIR/$ARTIFACT_NAME/bin/snooze" "$INSTALL_ROOT/bin/snooze"
chmod 755 "$INSTALL_ROOT/bin/snooze"

# Copy bundled shared libraries (e.g. libonnxruntime) if present
if [ -d "$TMP_DIR/$ARTIFACT_NAME/lib" ]; then
    cp -P "$TMP_DIR/$ARTIFACT_NAME/lib/"* "$INSTALL_ROOT/lib/" 2>/dev/null || true
fi

ln -sf "$INSTALL_ROOT/bin/snooze" "$BIN_DIR/snooze"
success "Installed snooze binary to $BIN_DIR/snooze"

# Initialize default models.toml if not present
if [ ! -f "$MODELS_DIR/models.toml" ]; then
    "$BIN_DIR/snooze" init "$MODELS_DIR" >/dev/null 2>&1 || true
    success "Initialized models directory at $MODELS_DIR"
fi

# Configure user-level background daemon unless --no-service was passed
if [ "$SETUP_SERVICE" = "0" ]; then
    if [ "$PLATFORM_OS" = "linux" ]; then
        configure_linux_user_service
    elif [ "$PLATFORM_OS" = "macos" ]; then
        configure_macos_launch_agent
    fi
fi

# Check whether BIN_DIR is in the user's PATH
case ":$PATH:" in
    *":$BIN_DIR:"*)
        ;;
    *)
        printf "\n${YELLOW}Note:${RESET} Add ${BOLD}%s${RESET} to your PATH if it is not already present:\n" "$BIN_DIR"
        printf "  export PATH=\"%s:\$PATH\"\n" "$BIN_DIR"
        ;;
esac

printf "\n${GREEN}${BOLD}SnoozeML is installed!${RESET} Quick commands to get started:\n"
printf "  ${BOLD}snooze status${RESET}                                  # Check running daemon status\n"
printf "  ${BOLD}snooze download BAAI/bge-small-en-v1.5 -o %s${RESET} # Download a model\n" "$MODELS_DIR"
printf "  ${BOLD}snooze monitor${RESET}                                 # Open live TUI dashboard\n"
