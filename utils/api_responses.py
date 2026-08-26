"""Uniform JSON responses for the Flask API."""
from flask import jsonify


def ok(data=None, **extra):
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return jsonify(payload)


def err(message: str, status_code: int = 400, **extra):
    payload = {"ok": False, "error": message}
    payload.update(extra)
    resp = jsonify(payload)
    resp.status_code = status_code
    return resp
