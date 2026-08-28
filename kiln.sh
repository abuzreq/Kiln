#!/usr/bin/env bash
# Kiln launcher for Linux. First run installs everything; later runs just start.
set -e
cd "$(dirname "$0")"

# Modern distributions ship python3 only; some minimal images ship neither under
# a bare `python`. Say which package to install rather than dying on
# "command not found" from inside the script.
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  cat >&2 <<'MSG'
Kiln needs Python 3.10 or newer, and no python3 was found on PATH.

  Debian/Ubuntu:  sudo apt install python3 python3-venv python3-pip
  Fedora/RHEL:    sudo dnf install python3 python3-pip
  Arch:           sudo pacman -S python python-pip

Then run ./kiln.sh again.
MSG
  exit 1
fi

exec "$PY" install.py --launch "$@"
