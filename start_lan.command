#!/usr/bin/env bash
# Kiln on macOS (double-clickable), reachable from the other machines on this
# network (the lab).
#
# Same launcher as kiln.command -- first run installs everything, later runs
# just start -- with --lan added, which binds every network interface instead of
# this machine only. The address to open from another machine is printed below
# as "On this network: ...".
#
# Anything you add is passed through, e.g.:
#     ./start_lan.command --no-window
#     ./start_lan.command --port 9000
set -e
cd "$(dirname "$0")"

cat <<'MSG'

  Starting Kiln with network access enabled.

  Kiln has no password: anyone who can reach this machine can use it,
  including your datasets, models and files. Only do this on a network
  you trust.

  If another machine cannot connect, allow incoming connections for
  Python in System Settings > Network > Firewall > Options. macOS
  usually asks the first time.

MSG

# Run through bash rather than ./kiln.command so this still works if the execute
# bit was lost in transit (a zip, a Windows checkout, a shared drive).
exec bash ./kiln.command --lan "$@"
