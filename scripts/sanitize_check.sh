#!/usr/bin/env bash
# Run before publishing. Looks for anything personal or secret in the files that would be committed.
# Exit status 0 = clean. Anything it prints needs a look.
cd "$(dirname "$0")/.."
FILES=$(git ls-files 2>/dev/null || find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './vendor/*')
[ -n "$FILES" ] || FILES=$(find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './vendor/*' -not -name '*.pyc')
bad=0
scan() { # label, regex
  hits=$(echo "$FILES" | xargs -d '\n' grep -nIE -- "$2" 2>/dev/null | grep -v 'scripts/sanitize_check.sh' || true)
  [ -n "$hits" ] && { echo "--- $1"; echo "$hits"; bad=1; }
}
scan "private/LAN IP addresses"        '\b(10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+)\b'
scan "home directories / usernames"    '/home/[a-z][a-z0-9_-]+|/Users/[A-Za-z]|C:\\\\Users'
scan "tokens, keys, passwords"         '(eyJ[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC|PRIVATE) |password *=|secret *=|api[_-]?key *=)'
scan "email addresses"                 '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}'
scan "MAC addresses"                   '\b([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b'
scan "personal/home-lab names"         '(jeff|thebakery|thebeast|jarvis|proxmox|portainer|donutmoney|wiltron|newmarket)'
# files that must never be committed
BIN=$(echo "$FILES" | grep -E '\.(onnx|ckpt|pt|pth|safetensors|wav|mp3|m4a|flac|ogg|toml)$' | grep -v -E 'config.example.toml|pyproject.toml' || true)
[ -n "$BIN" ] && { echo "--- models / audio / real config in the file list"; echo "$BIN"; bad=1; }
[ "$bad" = 0 ] && echo "sanitize check: clean ($(echo "$FILES" | wc -l) files scanned)"
exit $bad
