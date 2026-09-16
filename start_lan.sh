#!/usr/bin/env bash
# Kiln, reachable from the other machines on this network (the lab).
#
# Same launcher as kiln.sh -- first run installs everything, later runs just
# start -- with --lan added, which binds every network interface instead of this
# machine only. The address to open from another machine is printed below as
# "On this network: ...".
#
# Anything you add is passed through, e.g.:
#     ./start_lan.sh --no-window
#     ./start_lan.sh --port 9000
set -e
cd "$(dirname "$0")"

cat <<'MSG'

  Starting Kiln with network access enabled.

  Kiln has no password: anyone who can reach this machine can use it,
  including your datasets, models and files. Only do this on a network
  you trust.

  If another machine cannot connect, open the port in this machine's
  firewall (ufw: sudo ufw allow 8777/tcp) -- or whichever port Kiln
  reports below, if 8777 was taken.

MSG

# Run through bash rather than ./kiln.sh so this still works if the execute bit
# was lost in transit (a zip, a Windows checkout, a shared drive).
exec bash ./kiln.sh --lan "$@"
