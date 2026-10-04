"""Kiln Flask application factory.

Serves the prebuilt React frontend (app/frontend/build) and exposes the JSON API
used by every screen. Routes are organized into blueprints under
``app/backend/routes`` (system, projects, datasets, train, perform, craft,
library, tools).
"""
import mimetypes
from pathlib import Path

from flask import Flask, send_from_directory
from werkzeug.exceptions import HTTPException
from flask_cors import CORS

from utils.api_responses import err
from utils.exceptions import KilnError
from utils.logger import get_logger

log = get_logger("app")

# On Windows, mimetypes reads the registry and often maps .js to "text/plain",
# which makes browsers refuse to execute the ES-module frontend bundle (blank
# screen). Force correct JS/CSS MIME types so <script type="module"> loads.
# WebP is missing from the table on some Windows installs, and the example
# pictures the frontend ships are WebP.
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"),
                    (".webp", "image/webp")):
    mimetypes.add_type(_type, _ext)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_BUILD = ROOT / "app" / "frontend" / "build"


def _register_blueprints(app: Flask):
    """Import and register every available route blueprint.

    Imports are guarded so the app still boots if an optional module (or a heavy
    dependency it needs) is missing during early development.
    """
    from app.backend.routes import system

    app.register_blueprint(system.bp)

    optional = [
        "projects",
        "datasets",
        "train",
        "perform",
        "craft",
        "library",
        "tools",
    ]
    for name in optional:
        try:
            module = __import__(f"app.backend.routes.{name}", fromlist=["bp"])
            app.register_blueprint(module.bp)
            log.info("registered blueprint: %s", name)
        except Exception as e:  # noqa: BLE001
            log.warning("blueprint '%s' not registered: %s", name, e)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app)  # allow the Vite dev server to talk to the API during development

    _register_blueprints(app)

    # Generations wait in lanes; size them (and the model cache) once.
    from app.core.engine import lanes
    lanes.configure(lanes.default_lanes())

    # Model cards and thumbnails used to sit beside each model; move them into
    # the models folder's .kiln so the folder shows only the models.
    try:
        from app.core import library
        if moved := library.tidy_model_folders():
            log.info("moved %d model cards and thumbnails into .kiln", moved)
    except Exception as e:  # noqa: BLE001
        log.warning("could not tidy the models folder: %s", e)

    # --- static frontend --------------------------------------------------
    def _api_endpoint_exists(path: str) -> bool:
        """Does any real route serve this path, under any method?

        The SPA catch-all below matches every GET, so without this an /api/ path
        cannot tell "endpoint missing" from "wrong verb" — and a GET to a
        POST-only endpoint would quietly return the HTML shell.
        """
        adapter = app.url_map.bind("localhost")
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            try:
                endpoint, _ = adapter.match(path, method=method)
            except HTTPException:
                continue
            if endpoint != "serve_frontend":
                return True
        return False

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        # An unknown /api/* path is a missing endpoint, not a client-side route.
        # Without this the catch-all claims the URL and returns the HTML shell.
        if path.startswith("api/"):
            if _api_endpoint_exists("/" + path):
                return err("method not allowed", 405)
            return err("not found", 404)
        build = FRONTEND_BUILD
        target = build / path
        if path and target.exists() and target.is_file():
            explicit = {
                ".js": "text/javascript",
                ".mjs": "text/javascript",
                ".css": "text/css",
            }.get(target.suffix)
            return send_from_directory(build, path, mimetype=explicit)
        index = build / "index.html"
        if index.exists():
            return send_from_directory(build, "index.html")
        return (
            "<h1>Kiln</h1><p>Frontend build not found. Run "
            "<code>npm install &amp;&amp; npm run build</code> in "
            "<code>app/frontend</code>, or use the Vite dev server "
            "(<code>npm run dev</code>) which proxies to this API.</p>",
            200,
        )

    # --- uniform error handling ------------------------------------------
    @app.errorhandler(KilnError)
    def handle_kiln_error(e: KilnError):
        return err(e.message, e.status_code)

    @app.errorhandler(404)
    def handle_404(e):
        return err("not found", 404)

    @app.errorhandler(405)
    def handle_405(e):
        # The SPA catch-all rule matches every path for GET, so a POST to an
        # endpoint that does not exist is rejected at routing time as 405. For
        # an /api/* path that is misleading — nothing is there at all.
        from flask import request

        if request.path.startswith("/api/"):
            if _api_endpoint_exists(request.path):
                return err("method not allowed", 405)
            return err("not found", 404)
        return err("method not allowed", 405)

    @app.errorhandler(Exception)
    def handle_unexpected(e: Exception):
        # Werkzeug raises HTTPException for 400/405/413/... — those already carry
        # the right status, so preserve it. Only genuinely unexpected errors are
        # 500, and only those are worth a stack trace in the log.
        if isinstance(e, HTTPException):
            return err(e.description or e.name, e.code or 500)
        log.exception("unhandled error")
        return err(str(e), 500)

    return app
