#!/usr/bin/env bash
# Kiln launcher for macOS (double-clickable). First run installs; later runs start.
set -e
cd "$(dirname "$0")"

# The python3 that ships with macOS (Xcode's) is 3.9, and Kiln needs 3.10+.
# A python.org or Homebrew install sits beside it under a versioned name, so
# take the first interpreter new enough rather than whichever is first on PATH.
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  cat >&2 <<'MSG'
Kiln needs Python 3.10 or newer, and none was found. (The python3 that comes
with macOS is 3.9.)

Install the macOS (Apple Silicon) build from:
  https://www.python.org/downloads/

Then open this launcher again.
MSG
  exit 1
fi

exec "$PY" install.py --launch "$@"
