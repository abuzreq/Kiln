#!/usr/bin/env bash
# Kiln launcher for macOS (double-clickable). First run installs; later runs start.
set -e
cd "$(dirname "$0")"

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python
fi

exec "$PY" install.py --launch "$@"
