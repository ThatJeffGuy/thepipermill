#!/usr/bin/env bash
# Start Voice Studio automatically when you log in (systemd user service). Remove with:  systemctl --user disable --now voice-studio
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/voice-studio.service <<UNIT
[Unit]
Description=Voice Studio (Piper voice training web page)

[Service]
ExecStart=$DIR/.venv/bin/python -m voicestudio serve
WorkingDirectory=$DIR
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now voice-studio.service
echo "Running at http://127.0.0.1:8765"
