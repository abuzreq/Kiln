"""Take the README's screenshots from a running Kiln.

    pip install playwright pillow        # any Python; not one of Kiln's dependencies
    python scripts/make_readme_shots.py [--url http://127.0.0.1:8777] [--model NAME]

Drives the installed Microsoft Edge (or --browser chrome) through the app at
1440 x 900 and writes WebP files to screenshots/. It generates on the way --
a canvas image, a bent one, a merge ladder -- so it uses the GPU for a minute
or two, and those results land in the running Kiln's history like any other.

The pictures show whatever is in the workspace: the datasets, runs and models
there. --model picks the one used in Create (default: the sample model
txplsa-200, from Prepare > Models > Sample models); --merge-a and --merge-b
the merge pair, best two snapshots of one run so the ladder blends smoothly.
"""
import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"

# Jobs that run in the background on their own and never "finish" for a shot.
BACKGROUND = ("explore", "discover", "warmup", "preview", "previews")


def wait_idle(page, url, timeout=240):
    """Until no job the shot depends on is queued or running."""
    start = time.time()
    time.sleep(2)
    while time.time() - start < timeout:
        jobs = page.request.get(url + "/api/jobs?live=1&light=1").json().get("data") or []
        if not [j for j in jobs if j.get("status") in ("queued", "running")
                and j.get("kind") not in BACKGROUND]:
            return
        time.sleep(1)
    print("  (still busy after %ds; taking the shot anyway)" % timeout)


def shot(page, name, wait_ms=1500):
    page.wait_for_timeout(wait_ms)
    png = page.screenshot()
    Image.open(io.BytesIO(png)).convert("RGB").save(OUT / f"{name}.webp", quality=82, method=6)
    print("  " + name)


def click_text(page, text):
    page.get_by_text(text, exact=True).first.click()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8777")
    ap.add_argument("--model", default="txplsa-200")
    ap.add_argument("--browser", default="msedge", choices=["msedge", "chrome"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--merge-a", default="run1-step14200")
    ap.add_argument("--merge-b", default="dnd-run1-step1100")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        page = ctx.new_page()
        models = page.request.get(url + "/api/models").json()["data"]

        def path_of(name):
            for m in models:
                if m["name"] == name:
                    return m["path"]
            raise SystemExit(f"No model named {name!r} in this Kiln's library.")

        model = path_of(args.model)
        # The selected model is remembered in localStorage; set it before the app reads it.
        ctx.add_init_script(f"localStorage.setItem('kiln.model', {json.dumps(model)});")

        page.goto(url)
        page.wait_for_timeout(3000)

        click_text(page, "Prepare")
        shot(page, "data", 5000)
        click_text(page, "Train")
        page.wait_for_timeout(1500)
        click_text(page, "View previous runs")
        shot(page, "train", 5000)
        click_text(page, "Models")
        shot(page, "models", 5000)

        click_text(page, "Create")
        page.wait_for_timeout(2000)
        click_text(page, "Canvas")
        page.wait_for_timeout(1000)
        page.get_by_placeholder("random").first.fill(args.seed)
        page.get_by_role("button", name="Generate", exact=True).last.click()
        page.mouse.move(5, 450)  # no hover tooltip in the picture
        wait_idle(page, url)
        shot(page, "canvas", 2500)

        click_text(page, "Bend")
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Presets").first.click()
        shot(page, "bend-presets", 2500)
        # bottleneck-flare: the third starter.
        page.get_by_role("button", name="Load", exact=True).nth(2).click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Generate with bends").first.click()
        wait_idle(page, url)
        shot(page, "bend", 2500)

        click_text(page, "Merge")
        page.wait_for_timeout(2000)
        # Two snapshots of one run: related models, so every step of the
        # ladder is a picture rather than the mush of two trained apart.
        page.locator("select[aria-label='Model A']").select_option(path_of(args.merge_a))
        page.wait_for_timeout(1000)
        page.locator("select[aria-label='Model B']").select_option(path_of(args.merge_b))
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Render ladder").first.click()
        wait_idle(page, url)
        shot(page, "merge", 2500)

        click_text(page, "Sweep")
        shot(page, "sweep", 2500)

        # Last, so the hub's model cards have had the whole run to make their
        # previews (Kiln samples them on first view and caches them).
        page.get_by_role("button", name="Kiln — back to your hub").click()
        page.wait_for_timeout(20000)
        page.mouse.wheel(0, 260)
        shot(page, "start", 3000)
        browser.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
