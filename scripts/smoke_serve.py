"""Verify the Flask app serves the built React frontend and its assets."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.backend.app import create_app

c = create_app().test_client()
r = c.get("/")
print("root status:", r.status_code, "| has #root:", b'id="root"' in r.data)
m = re.search(rb'src="([^"]+\.js)"', r.data)
asset = m.group(1).decode() if m else None
print("asset ref:", asset)
if asset:
    a = c.get(asset)
    print("asset status:", a.status_code, "| bytes:", len(a.data))
print("OK")
