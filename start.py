"""Start the Kiln backend and open the desktop window.

Runs the Flask server in a background thread and opens a native window via
pywebview. If pywebview (or its platform runtime) is unavailable, falls back to
printing the local URL so you can open Kiln in a browser.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.core.config import DEFAULT_PORT  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("start")


FRONTEND_BUILD = ROOT / "app" / "frontend" / "build"


def ensure_frontend():
    """Guarantee the built SPA exists so the server serves the real UI, not the fallback."""
    index = FRONTEND_BUILD / "index.html"
    if index.exists():
        log.info("Serving frontend from %s", FRONTEND_BUILD)
        return True

    log.warning("No frontend build at %s — attempting to build it.", FRONTEND_BUILD)
    import platform
    import subprocess

    npm = "npm.cmd" if platform.system() == "Windows" else "npm"
    frontend_dir = ROOT / "app" / "frontend"
    try:
        subprocess.check_call([npm, "install"], cwd=str(frontend_dir))
        subprocess.check_call([npm, "run", "build"], cwd=str(frontend_dir))
    except Exception as e:  # noqa: BLE001
        log.error(
            "Could not build the frontend (%s). Install Node 18+ and run "
            "`npm install && npm run build` in app/frontend.", e
        )
    if index.exists():
        log.info("Frontend build ready at %s", FRONTEND_BUILD)
        return True
    log.warning("Frontend build still missing; a placeholder page will be shown.")
    return False


# Loopback: reachable only from this machine. Anything else is a network
# address, and Kiln has no login of any kind -- see the warning in main().
LOCAL_HOST = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"


def serve(app, port: int, host: str = LOCAL_HOST):
    # threaded WSGI server; fine for a single user, on this machine or the LAN.
    app.run(host=host, port=port, threaded=True, use_reloader=False)


def local_url(host: str, port: int) -> str:
    """The URL to open on *this* machine, whatever the server is bound to.

    ``0.0.0.0`` is a bind address, not somewhere you can browse to, so the
    window and the health check always talk to a real address.
    """
    reachable = LOCAL_HOST if host in (ALL_INTERFACES, "::", "") else host
    return f"http://{reachable}:{port}"


def lan_addresses() -> list[str]:
    """This machine's addresses on the network, for the "open it on your phone" line."""
    import socket

    found = []
    # Opening a UDP socket sends nothing; it just asks the routing table which
    # local address would be used to reach the internet -- the LAN address.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except Exception:  # noqa: BLE001
        pass
    return found


def port_is_free(port: int, host: str = LOCAL_HOST) -> bool:
    """Can we bind this port right now?

    The server runs on a daemon thread, so a bind failure there is invisible:
    the thread dies, wait_for_server times out, and we would go on to open a
    window pointing at nothing. Check up front instead -- on the same address
    the server will use, since a port can be free on one and taken on another.
    """
    import socket

    # Deliberately no SO_REUSEADDR: on Windows it lets a bind succeed against a
    # port that is already LISTENING, which would make this probe always say
    # "free" — the exact opposite of what it is for.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(preferred: int, tries: int = 20, host: str = LOCAL_HOST) -> int:
    """The preferred port, or the next free one above it."""
    for candidate in range(preferred, preferred + tries):
        if port_is_free(candidate, host):
            return candidate
    raise SystemExit(
        f"Could not find a free port in {preferred}-{preferred + tries - 1}.\n"
        "Something else is using them. Free one up, or pick your own with:\n"
        "    python start.py --port <number>\n"
        "(or set the KILN_PORT environment variable)"
    )


def wait_for_server(url: str, timeout: float = 20.0) -> bool:
    import urllib.request

    url = url.rstrip("/") + "/api/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("KILN_PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default=os.environ.get("KILN_HOST", LOCAL_HOST),
                    help="address to bind (default 127.0.0.1: this machine only)")
    ap.add_argument("--lan", action="store_true",
                    help="also serve to the local network, so another device can open Kiln "
                         "(shorthand for --host 0.0.0.0)")
    ap.add_argument("--no-window", action="store_true", help="run headless (browser only)")
    args = ap.parse_args()

    host = ALL_INTERFACES if args.lan else args.host

    ensure_frontend()

    from app.backend.app import create_app

    port = pick_port(args.port, host=host)
    if port != args.port:
        log.warning("port %s is in use — using %s instead", args.port, port)

    app = create_app()
    url = local_url(host, port)

    server = threading.Thread(target=serve, args=(app, port, host), daemon=True)
    server.start()

    if not wait_for_server(url):
        raise SystemExit(
            f"The Kiln server did not come up on port {port} within 20 seconds.\n"
            "Check the messages above for the reason."
        )
    log.info("Kiln is running at %s", url)

    # The first Generate otherwise spends most of its time importing libraries.
    # Not any earlier: importing alongside create_app made both several times
    # slower (31-82 s for the warm-up, against 16-19 s once the server is up).
    from app.core.engine import warmup
    warmup.start()

    if host != LOCAL_HOST:
        for ip in (lan_addresses() if host == ALL_INTERFACES else [host]):
            log.info("On this network: http://%s:%s", ip, port)
        # Said plainly, every time: there is no login, and the API can read and
        # write the whole workspace. Fine on a home network, not on a café one.
        log.warning(
            "Kiln has no password: anyone who can reach that address can use it, "
            "including your datasets, models and files. Only do this on a network you trust.")

    if args.no_window:
        log.info("Headless mode: open %s in your browser. Ctrl+C to stop.", url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    try:
        import webview  # pywebview

        webview.create_window("Kiln", url, width=1440, height=920, min_size=(1024, 680))
        webview.start()
    except Exception as e:  # noqa: BLE001
        import platform

        log.warning("Native window unavailable (%s).", e)
        if platform.system() == "Linux":
            # pywebview installs from pip, but its Linux backend is system GTK +
            # WebKit, which pip cannot provide. Everything works without it --
            # the window is the only thing missing -- so say how to get it and
            # carry on rather than treating this as an error.
            log.info(
                "On Linux the desktop window needs a WebKit runtime: "
                "sudo apt install python3-gi gir1.2-webkit2-4.1 "
                "(Fedora: python3-gobject webkit2gtk4.1). "
                "Or pass --no-window to skip it.")
        log.info("Open Kiln in your browser at %s — Ctrl+C to stop.", url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
