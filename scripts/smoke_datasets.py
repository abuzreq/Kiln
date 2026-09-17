"""Smoke test for record-based datasets: images stay where they are.

Against a throwaway workspace and a separate "external" folder, no GPU:

- linking a folder and a file, excluding and restoring, uploads, new files in a
  linked folder showing up
- deleting a dataset never touches linked files; a legacy dataset folder that
  is itself a link loses only the link
- /api/thumb serves linked images and nothing else
- video frames are extracted once, cached, and re-extracted when fps changes
- training-time augmentation: sizes, ranges, determinism, the snapshot a run
  trains from, and ManifestDataset tensors for both engines
- leftover drafts from the old copy-then-build flow become datasets

    python scripts/smoke_datasets.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKSPACE = Path(tempfile.mkdtemp(prefix="kiln_ds_ws_"))
EXTERNAL = Path(tempfile.mkdtemp(prefix="kiln_ds_ext_"))
os.environ["KILN_WORKSPACE"] = str(WORKSPACE)

from PIL import Image  # noqa: E402

from app.backend.app import create_app  # noqa: E402
from app.backend.data import augment, manifest  # noqa: E402

app = create_app()
client = app.test_client()


def api(method, url, **kw):
    r = getattr(client, method)(f"/api{url}", **kw)
    body = r.get_json(silent=True)
    return r.status_code, (body or {}).get("data", body)


def make_image(path: Path, size=(96, 64), color=(200, 40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def link_dir(link: Path, target: Path):
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              stdout=subprocess.DEVNULL)


def check_records():
    photos = EXTERNAL / "photos"
    for i in range(4):
        make_image(photos / f"p{i}.png", color=(40 * i, 80, 120))
    make_image(photos / "nested" / "deep.jpg")
    loose = make_image(EXTERNAL / "loose.png")

    code, d = api("post", "/datasets", json={"name": "cats"})
    assert code == 200 and d["kind"] == "record" and d["count"] == 0, (code, d)
    code, d = api("post", "/datasets", json={"name": "cats"})
    assert code == 409

    code, d = api("post", "/datasets/cats/link", json={"path": str(photos)})
    assert code == 200 and d["count"] == 5, d
    code, d = api("post", "/datasets/cats/link", json={"path": str(loose)})
    assert d["count"] == 6
    code, _ = api("post", "/datasets/cats/link", json={"path": str(photos)})
    assert code == 400, "the same folder linked twice"

    victim = photos / "p0.png"
    code, d = api("post", "/datasets/cats/exclude", json={"path": str(victim)})
    assert d["count"] == 5 and d["excluded_count"] == 1
    assert victim.exists(), "excluding a linked file deleted it"
    code, files = api("get", "/datasets/cats/files?limit=200")
    assert str(victim) not in files["files"]
    code, d = api("post", "/datasets/cats/exclude", json={"path": str(victim), "restore": True})
    assert d["count"] == 6

    import io
    data = {"files": [(io.BytesIO(open(photos / "p1.png", "rb").read()), "up.png")]}
    code, d = api("post", "/datasets/cats/upload", data=data, content_type="multipart/form-data")
    assert code == 200 and d["count"] == 7 and d["uploads"] == 1, d
    upload = WORKSPACE / "datasets" / "cats" / "files" / "up.png"
    code, d = api("post", "/datasets/cats/exclude", json={"path": str(upload)})
    assert not upload.exists() and d["count"] == 6, "removing an upload should delete Kiln's copy"

    make_image(photos / "p9.png")
    code, d = api("get", "/datasets/cats")
    assert d["count"] == 7, "a file dropped into a linked folder did not show up"

    code, d = api("post", "/datasets/cats/recipe",
                  json={"recipe": {"width": 64, "height": 48, "augmentations": ["vflip", "bogus"]}})
    assert d["recipe"]["width"] == 64 and d["recipe"]["augmentations"] == ["vflip"], d["recipe"]

    code, d = api("post", "/datasets/preview", json={"dataset": "cats"})
    assert code == 200 and [p["label"] for p in d["previews"]] == ["Framed", "V flip"], d

    code, listed = api("get", "/datasets")
    cats = next(x for x in listed if x["name"] == "cats")
    assert cats["kind"] == "record" and cats["width"] == 64 and cats["count"] == 7

    # media: linked files are served, anything else outside the workspace is not
    code_ok = client.get("/api/thumb", query_string={"path": str(photos / "p1.png")}).status_code
    stranger = make_image(Path(tempfile.mkdtemp(prefix="kiln_ds_other_")) / "secret.png")
    code_no = client.get("/api/media", query_string={"path": str(stranger)}).status_code
    code_dots = client.get("/api/media", query_string={
        "path": str(photos / ".." / ".." / stranger.parent.name / "secret.png")}).status_code
    assert code_ok == 200 and code_no == 404 and code_dots == 404, (code_ok, code_no, code_dots)
    shutil.rmtree(stranger.parent, ignore_errors=True)

    code, d = api("delete", "/datasets/cats", json={})
    assert code == 200 and not (WORKSPACE / "datasets" / "cats").exists()
    assert len(list(photos.rglob("*.*"))) == 6 and loose.exists(), "deleting the dataset touched linked files"
    print("records: link, exclude/restore, uploads, live folders, recipe, preview, media, delete")


def check_linked_legacy_delete():
    """The old danger: a dataset folder that is a link to the user's real folder."""
    real = EXTERNAL / "real_dataset"
    for i in range(3):
        make_image(real / f"{i:06d}.png")
    link_dir(WORKSPACE / "datasets" / "mine", real)
    code, listed = api("get", "/datasets")
    mine = next(x for x in listed if x["name"] == "mine")
    assert mine["kind"] == "folder" and mine["count"] == 3
    code, d = api("delete", "/datasets/mine", json={"delete_files": True})
    assert code == 200 and d.get("unlinked"), d
    assert not os.path.lexists(WORKSPACE / "datasets" / "mine")
    assert len(list(real.glob("*.png"))) == 3, "deleting a linked dataset deleted the originals"

    # a link inside a folder being deleted is also only unlinked
    from utils.fs import safe_rmtree

    holder = WORKSPACE / "holder"
    holder.mkdir()
    link_dir(holder / "inner", real)
    safe_rmtree(holder)
    assert not holder.exists() and len(list(real.glob("*.png"))) == 3
    print("legacy: a linked dataset folder loses only the link")


def check_video_frames():
    import cv2
    import numpy as np

    vid = EXTERNAL / "clip.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    w.release()

    api("post", "/datasets", json={"name": "clips", "recipe": {"video_fps": 5}})
    code, d = api("post", "/datasets/clips/link", json={"path": str(vid)})
    assert d["videos_pending"] == 1 and d["count"] == 0, d
    ds_dir = WORKSPACE / "datasets" / "clips"
    n = manifest.extract_pending(ds_dir)
    assert n == 10, n
    code, d = api("get", "/datasets/clips")
    assert d["videos_pending"] == 0 and d["count"] == 10, d
    assert manifest.extract_pending(ds_dir) == 0, "frames were re-extracted without a change"
    api("post", "/datasets/clips/recipe", json={"recipe": {"video_fps": 10}})
    code, d = api("get", "/datasets/clips")
    assert d["videos_pending"] == 1
    assert manifest.extract_pending(ds_dir) == 20
    print("video: frames cached per fps, re-extracted on change")


def check_augment():
    img = Image.new("RGB", (64, 48), (100, 150, 200))
    ops = ["hflip", "vflip", "rotate", "brightness", "contrast"]
    settings = {"brightness": {"min": 0.5, "max": 0.6, "levels": 2},
                "contrast": {"min": 0.9, "max": 1.1, "levels": 3}}

    # Each augmentation offers its own options; the recipe is their product.
    # Contrast asks for three levels between 0.9 and 1.1, but the middle one is
    # 1.0 -- no change -- and the untouched image is already in the set.
    assert augment.counts(ops, settings) == {"hflip": 2, "vflip": 2, "rotate": 4,
                                             "brightness": 3, "contrast": 3}, augment.counts(ops, settings)
    assert augment.count(ops, settings) == 2 * 2 * 4 * 3 * 3
    assert augment.count(["hflip"]) == 2
    assert augment.count(["rotate"], {"rotate": {"mode": "fixed", "angle": 45}}) == 2
    assert augment.count([]) == 1

    combos = augment.combinations(ops, settings)
    assert len(combos) == augment.count(ops, settings)
    assert combos[0] == {}, "the untouched image should come first"
    assert len({tuple(sorted(c.items())) for c in combos}) == len(combos), "a version was repeated"
    assert combos == augment.combinations(ops, settings), "the same recipe gave a different set"
    for c in combos:
        assert augment.apply(img, c).size == img.size

    flat = Image.new("RGB", (8, 8), (200, 200, 200))
    for c in augment.combinations(["brightness"], settings)[1:]:
        px = augment.apply(flat, c).getpixel((0, 0))[0]
        assert 200 * 0.5 - 1 <= px <= 200 * 0.6 + 1, px
    assert augment.combinations([], None) == [{}]
    assert augment.apply(img, {}).tobytes() == img.tobytes()
    print("augment: each option multiplies the set, and the set never changes")


def check_snapshot_and_dataset():
    import torch

    from app.core.engine.train_data import ManifestDataset

    src = EXTERNAL / "train_src"
    for i in range(5):
        make_image(src / f"t{i}.png", size=(90, 60), color=(i * 50, 100, 30))
    api("post", "/datasets", json={"name": "trainme",
                                   "recipe": {"width": 48, "height": 32,
                                              "augmentations": ["vflip", "rotate"]}})
    api("post", "/datasets/trainme/link", json={"path": str(src)})
    api("post", "/datasets/trainme/exclude", json={"path": str(src / "t4.png")})
    run = WORKSPACE / "runs" / "r1"
    snap = manifest.snapshot(WORKSPACE / "datasets" / "trainme", run)
    data = json.loads(snap.read_text(encoding="utf-8"))
    assert len(data["files"]) == 4 and data["recipe"]["width"] == 48
    assert data["variants"] == 8 and data["total"] == 32, data  # 4 files x (2 flips x 4 turns)

    x = ManifestDataset.from_snapshot(snap, 32, "resize", engine="xurdif")
    t = x[0]
    assert t.shape == (3, 32, 32) and -0.5 <= float(t.min()) and float(t.max()) <= 0.5
    dfs = ManifestDataset.from_snapshot(snap, 32, "crop", engine="diffusers")
    t = dfs[1]
    assert t.shape == (3, 32, 32) and -1.0 <= float(t.min()) and float(t.max()) <= 1.0
    assert x.framed(0).size == (48, 32) and 0 in x._cache

    # Four images, eight versions of each, and both engines build the same set.
    assert len(x) == 32 and x.variants == 8 and x.combinations == dfs.combinations
    assert x.sample(0).tobytes() == x.framed(0).tobytes(), "the first version is the framed image"
    batch = torch.stack([x[i] for i in range(len(x))])
    assert batch.shape == (32, 3, 32, 32)

    empty = WORKSPACE / "datasets" / "empty"
    manifest.new(empty)
    try:
        manifest.snapshot(empty, WORKSPACE / "runs" / "r2")
        raise AssertionError("snapshot of an empty dataset")
    except Exception as e:
        assert "no images" in str(e)
    assert manifest.snapshot(WORKSPACE / "datasets" / "legacy_none", run) is None
    print("training: snapshot freezes files + recipe; ManifestDataset gives both engines' tensors")


def check_config_from_run():
    """A run trained from a snapshot resumes from it, even if the dataset is gone."""
    from app.core.engine.trainer import TrainConfig, _write_run_meta, config_from_run

    run = WORKSPACE / "runs" / "r1"
    cfg = TrainConfig(dataset=str(WORKSPACE / "datasets" / "trainme"), out_dir=str(run),
                      manifest=str(run / "dataset.json"), save_every=10)
    _write_run_meta(run, cfg)
    assert "--manifest" in cfg.to_args()
    (run / "model-1.pt").write_bytes(b"x")
    shutil.rmtree(WORKSPACE / "datasets" / "trainme")
    resumed = config_from_run(run, 100)
    assert resumed.manifest == str(run / "dataset.json")
    assert "--manifest" in resumed.to_args()
    print("resume: continues from the run's snapshot after the dataset was deleted")


def check_adopt_drafts():
    draft = WORKSPACE / "drafts" / "oldstuff"
    make_image(draft / "a.png")
    make_image(draft / "sub" / "b.png")
    (WORKSPACE / "drafts" / "emptyone").mkdir(parents=True)
    code, listed = api("get", "/datasets")
    old = next(x for x in listed if x["name"] == "oldstuff")
    assert old["kind"] == "record" and old["count"] == 2, old
    assert not draft.exists() and not (WORKSPACE / "drafts" / "emptyone").exists()
    print("drafts: leftover drafts become datasets")


def main():
    try:
        check_records()
        check_linked_legacy_delete()
        check_video_frames()
        check_augment()
        check_snapshot_and_dataset()
        check_config_from_run()
        check_adopt_drafts()
    finally:
        from utils.fs import safe_rmtree

        safe_rmtree(WORKSPACE)
        shutil.rmtree(EXTERNAL, ignore_errors=True)
    print("smoke_datasets: ok")


if __name__ == "__main__":
    main()
