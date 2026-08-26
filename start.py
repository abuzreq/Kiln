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


def serve(app, port: int):
    # threaded WSGI server; fine for a single local user.
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


def wait_for_server(port: int, timeout: float = 20.0) -> bool:
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/health"
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
    ap.add_argument("--no-window", action="store_true", help="run headless (browser only)")
    args = ap.parse_args()

    ensure_frontend()

    from app.backend.app import create_app

    app = create_app()
    url = f"http://127.0.0.1:{args.port}"

    server = threading.Thread(target=serve, args=(app, args.port), daemon=True)
    server.start()

    if not wait_for_server(args.port):
        log.error("server did not start in time")
    log.info("Kiln is running at %s", url)

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
        log.warning("Native window unavailable (%s).", e)
        log.info("Open Kiln in your browser at %s . Ctrl+C to stop.", url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
