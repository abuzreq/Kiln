"""Smoke test for model housekeeping: preview failures, Delete vs Hide.

Runs against a throwaway workspace, no GPU:

- a model whose previews fail is attempted once, not on every hub poll, and is
  retried after the file changes;
- a model Kiln owns can be deleted; a model outside the workspace -- or linked
  into it -- refuses delete, can be hidden, drops out of the model list, and
  comes back with ``hidden=1``.

    python scripts/smoke_library.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKSPACE = Path(tempfile.mkdtemp(prefix="kiln_lib_ws_"))
OUTSIDE = Path(tempfile.mkdtemp(prefix="kiln_lib_ext_"))
os.environ["KILN_WORKSPACE"] = str(WORKSPACE)

from app.core import library, previews  # noqa: E402
from utils.exceptions import ValidationError  # noqa: E402


def link_dir(link: Path, target: Path):
    """A directory link: a symlink where allowed, a junction on Windows otherwise."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              stdout=subprocess.DEVNULL)


def check_preview_failures():
    pt = WORKSPACE / "models" / "broken.pt"
    pt.parent.mkdir(parents=True, exist_ok=True)
    pt.write_bytes(b"not a checkpoint")

    calls = []

    def failing(path, count=previews.PREVIEW_COUNT):
        calls.append(path)
        raise ValueError("cannot load this model")

    real = previews.generate
    previews.generate = failing
    try:
        assert previews.enqueue(str(pt)) is True
        t0 = time.time()
        while previews.pending_count() and time.time() - t0 < 10:
            time.sleep(0.05)
        assert calls == [str(pt)], calls
        assert previews.failure(str(pt)) == "cannot load this model"
        for _ in range(5):  # the hub polling again and again
            assert previews.enqueue(str(pt)) is False
        assert len(calls) == 1, "a failed model was retried without changing"

        time.sleep(0.05)
        pt.write_bytes(b"a different, still broken file")
        assert previews.failure(str(pt)) is None, "a changed file keeps its old failure"
        assert previews.enqueue(str(pt)) is True
        t0 = time.time()
        while previews.pending_count() and time.time() - t0 < 10:
            time.sleep(0.05)
        assert len(calls) == 2
    finally:
        previews.generate = real
    print("previews: failure remembered, retried only after the file changed")


def check_delete_vs_hide():
    owned = WORKSPACE / "models" / "mine.pt"
    owned.write_bytes(b"x")
    card = library.card_path(owned)
    card.parent.mkdir(exist_ok=True)
    card.write_text("{}", encoding="utf-8")
    assert library.owned_by_kiln(owned)
    library.delete_model_files(owned)
    assert not owned.exists() and not card.exists()

    external = OUTSIDE / "theirs.pt"
    external.write_bytes(b"x")
    assert not library.owned_by_kiln(external)
    try:
        library.delete_model_files(external)
        raise AssertionError("deleted a model outside the workspace")
    except ValidationError:
        pass
    assert external.exists()

    # A folder linked into the workspace: the path looks like the workspace's,
    # the file is the user's.
    linked_dir = WORKSPACE / "models" / "linked"
    link_dir(linked_dir, OUTSIDE)
    through_link = linked_dir / "theirs.pt"
    assert through_link.exists()
    assert not library.owned_by_kiln(through_link), "a linked-in model counted as Kiln's"
    try:
        library.delete_model_files(through_link)
        raise AssertionError("deleted a model through a link")
    except ValidationError:
        pass
    assert external.exists()

    library.set_hidden(str(external), True)
    assert library._norm_star_path(str(external)) in library.list_hidden()
    library.set_hidden(str(external), False)
    assert library.list_hidden() == []
    print("models: owned deletes; external and linked refuse delete; hide/unhide round-trips")


def check_scan_public_hidden():
    """Hidden models drop out of the list and come back on request, with flags."""
    import torch

    from app.core.engine.arch import build_unet
    from app.core.model_manager import manager

    mults = [1, 2, 2, 2]
    unet = build_unet("tinyunet_with_attention3", mults)
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    ckpt = {"step": 0, "model": state, "ema": state, "mults": mults,
            "mtype": "tinyunet_with_attention3", "pred": "x0"}
    mine = WORKSPACE / "models" / "listed.pt"
    torch.save(ckpt, mine)
    theirs = OUTSIDE / "found.pt"
    torch.save(ckpt, theirs)

    real_sources = manager._sources
    manager._sources = lambda extra_dirs=None: real_sources(extra_dirs) + [(OUTSIDE, "custom")]
    try:
        by_name = {m["name"]: m for m in manager.scan_public()}
        assert by_name["listed"]["owned"] is True and by_name["listed"]["hidden"] is False
        assert by_name["found"]["owned"] is False

        library.set_hidden(str(theirs), True)
        names = {m["name"] for m in manager.scan_public()}
        assert "found" not in names and "listed" in names
        again = {m["name"]: m for m in manager.scan_public(include_hidden=True)}
        assert again["found"]["hidden"] is True
    finally:
        manager._sources = real_sources
        library.set_hidden(str(theirs), False)
    print("model list: owned/hidden flags; hidden models listed only on request")


def _tiny_checkpoint(dest):
    """A real, loadable xurdif checkpoint, small enough to write in a test."""
    import torch

    from app.core.engine.arch import build_unet

    mults = [1, 2, 2, 2]
    unet = build_unet("tinyunet_with_attention3", mults)
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    torch.save({"step": 0, "model": state, "ema": state, "mults": mults,
                "mtype": "tinyunet_with_attention3", "pred": "x0"}, dest)
    return dest


def check_import():
    """Importing a .pt the user already has: copied in, validated, and Kiln's.

    A tester could not find a way to get their own models in, guessed at
    models/pretrained, and then lost track of the files. Copying into the
    workspace is what makes "where do my models live" have one answer -- and what
    makes the copy deletable rather than only hideable.
    """
    from app.backend.app import create_app
    from app.core.model_manager import manager

    c = create_app().test_client()
    src = _tiny_checkpoint(OUTSIDE / "theirs-to-import.pt")

    r = c.post("/api/library/model/import", json={"path": str(src), "name": "brought-in"})
    assert r.status_code == 200, r.get_json()
    dest = WORKSPACE / "models" / "brought-in.pt"
    assert dest.exists(), "import wrote nothing"
    assert src.exists(), "import moved the original instead of copying it"

    manager.clear_cache()
    listed = {m["name"]: m for m in manager.scan_public()}
    assert listed["brought-in"]["owned"] is True, listed["brought-in"]
    assert library.card_path(dest).exists(), "an imported model got no card"

    # the same name twice would silently replace a model, so it is refused
    r = c.post("/api/library/model/import", json={"path": str(src), "name": "brought-in"})
    assert r.status_code >= 400, "a duplicate name was accepted"

    # and a file that is not a checkpoint must not be left lying in the library
    junk = OUTSIDE / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    r = c.post("/api/library/model/import", json={"path": str(junk), "name": "junky"})
    assert r.status_code >= 400, "a non-checkpoint was imported"
    assert not (WORKSPACE / "models" / "junky.pt").exists(), \
        "a rejected import was left on disk"
    assert c.post("/api/library/model/import",
                  json={"path": str(OUTSIDE / "nope.pt")}).status_code == 404

    # a file Kiln cannot read is reported rather than silently absent
    shutil.copy2(junk, WORKSPACE / "models" / "broken.pt")
    manager.clear_cache()
    bad = c.get("/api/library/models/unreadable").get_json()["data"]
    assert any(b["name"] == "broken.pt" and b["why"] for b in bad), bad
    assert "broken" not in {m["name"] for m in manager.scan_public()}
    (WORKSPACE / "models" / "broken.pt").unlink()

    library.delete_model_files(dest)
    print("import: copies in, validates, is Kiln's to delete; unreadable files are named")


def check_lineage():
    """Every model Kiln writes names its parents, and a parent is found again
    after a rename -- by its card's fingerprint, or failing that by hashing."""
    import json

    from app.backend.app import create_app

    c = create_app().test_client()
    models = WORKSPACE / "models"
    models.mkdir(parents=True, exist_ok=True)
    runs = WORKSPACE / "runs"

    def run_dir(name, meta):
        d = runs / name
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"name": name, **meta}), encoding="utf-8")
        return d

    parent = _tiny_checkpoint(models / "lineage-parent.pt")
    fp = library.fingerprint(parent)
    assert fp and len(fp) == 64

    # a continue: an xurdif run that loaded a library model
    d = run_dir("lin-run", {"resume": str(parent), "save_every": 10})
    _tiny_checkpoint(d / "model-1.pt")
    r = c.post("/api/runs/lin-run/checkpoints/save",
               json={"filename": "model-1.pt", "save_as": "lin-child"})
    assert r.status_code == 200, r.get_json()
    child = models / "lin-child.pt"
    card = library.read_card(child)
    lin = card["lineage"]
    assert lin["kind"] == "continue" and lin["run"] == "lin-run", lin
    (p,) = lin["parents"]
    assert p["role"] == "start" and p["path"] == str(parent) and p["fingerprint"] == fp, p
    assert card["fingerprint"] == library.fingerprint(child), "a card's own fingerprint is stale"

    # a scratch run has no parents
    assert library.run_lineage(run_dir("lin-scratch", {"resume": ""})) == \
        {"kind": "scratch", "parents": [], "run": "lin-scratch"}

    # a Diffusers fine-tune: its snapshot folders read the base from run.json
    d = run_dir("lin-ft", {"backend": "diffusers", "mode": "finetune", "base_model": str(parent)})
    (d / "model-1").mkdir()
    lin = library.lineage_of(d / "model-1")
    assert lin["kind"] == "finetune" and lin["parents"][0]["role"] == "base", lin
    assert lin["parents"][0]["fingerprint"] == fp, lin
    # a base from the Hub is recorded by name, with nothing to fingerprint
    d = run_dir("lin-hub", {"backend": "diffusers", "mode": "lora",
                            "base_model": "google/ddpm-cifar10-32"})
    (base,) = library.run_lineage(d)["parents"]
    assert base["name"] == "ddpm-cifar10-32" and base["fingerprint"] is None, base

    # a merge names A and B by role
    other = _tiny_checkpoint(models / "lineage-other.pt")
    r = c.post("/api/craft/merge", json={"model_a": str(parent), "model_b": str(other),
                                         "out_name": "lin-merge", "method": "linear", "alpha": 0.5})
    assert r.status_code == 200, r.get_json()
    lin = library.read_card(models / "lin-merge.pt")["lineage"]
    assert lin["kind"] == "merge", lin
    assert [(q["role"], q["fingerprint"]) for q in lin["parents"]] == \
        [("a", fp), ("b", library.fingerprint(other))], lin

    # a file picked from disk is a copy that keeps its source
    src = _tiny_checkpoint(OUTSIDE / "lin-source.pt")
    r = c.post("/api/library/model/import", json={"path": str(src), "name": "lin-copy"})
    assert r.status_code == 200, r.get_json()
    lin = library.read_card(models / "lin-copy.pt")["lineage"]
    assert lin["kind"] == "copy" and lin["parents"][0]["role"] == "source", lin
    assert lin["parents"][0]["fingerprint"] == library.fingerprint(src) \
        == library.read_card(models / "lin-copy.pt")["fingerprint"], "a copy differs from its source"

    # the parent renamed in Kiln: found through its card
    library.ensure_card(parent)
    renamed = library.rename_model(str(parent), "lineage-renamed")["path"]
    assert library.resolve_parent(p) == renamed, "a renamed parent was lost"
    # then moved by hand, leaving its card behind: found by hashing same-size files
    moved = models / "moved-by-hand.pt"
    Path(renamed).rename(moved)
    assert library.resolve_parent(p) == str(moved), "a moved parent was not found by fingerprint"
    moved.unlink()
    assert library.resolve_parent(p) is None, "a deleted parent should resolve to nothing"

    for name in ("lin-child", "lin-merge", "lin-copy", "lineage-other"):
        library.delete_model_files(models / f"{name}.pt")
    shutil.rmtree(runs, ignore_errors=True)
    print("lineage: continue, fine-tune, merge and copy name their parents; "
          "a renamed or moved parent is found by fingerprint")


def check_hidden_follows_rename():
    """Both path lists have to follow a rename, or the flag goes stale.

    A hidden model that is renamed would otherwise reappear in the lists *and*
    leave a dead entry behind -- worse than either outcome alone.
    """
    from app.core.model_manager import manager

    src = _tiny_checkpoint(WORKSPACE / "models" / "to-rename.pt")
    library.ensure_card(src, name="to-rename", original_name="to-rename")
    library.set_hidden(str(src), True)
    library.toggle_star(str(src))

    result = library.rename_model(str(src), "renamed")
    moved = Path(result["path"])
    assert moved.name == "renamed.pt", result
    hidden = library.list_hidden()
    assert len(hidden) == 1, hidden
    assert Path(hidden[0]).name == "renamed.pt", hidden
    assert Path(library.list_stars()[0]).name == "renamed.pt", library.list_stars()

    # and deleting the folder a hidden model lived in clears it out
    library.remove_hidden_under(WORKSPACE / "models")
    assert library.list_hidden() == [], library.list_hidden()
    moved.unlink(missing_ok=True)
    manager.clear_cache()
    print("hide: follows a rename, and is cleared when its folder goes")


def check_sidecars_tucked_away():
    """Cards and thumbnails live in models/.kiln, so the folder lists only models.

    Old ones beside the model are still read, and tidying moves them in; a
    picture with no model of its name is someone's own and stays put.
    """
    from app.core.model_manager import manager

    models = WORKSPACE / "models"
    old = _tiny_checkpoint(models / "old-style.pt")
    (models / "old-style.card.json").write_text('{"name": "Old style"}', encoding="utf-8")
    (models / "old-style.png").write_bytes(b"png")
    (models / "holiday.png").write_bytes(b"png")
    assert library.read_card(old)["name"] == "Old style", "an old card beside the model was not read"
    assert library.own_thumb(old) == models / "old-style.png"

    assert library.tidy_model_folders() == 2
    side = models / ".kiln"
    assert (side / "old-style.card.json").exists() and (side / "old-style.png").exists()
    assert not (models / "old-style.card.json").exists() and not (models / "old-style.png").exists()
    assert (models / "holiday.png").exists(), "tidying moved a picture that is not a thumbnail"
    assert library.read_card(old)["name"] == "Old style"

    manager.clear_cache()
    listed = {m["path"]: m for m in manager.scan_public()}
    assert listed[str(old)]["name"] == "Old style", listed[str(old)]
    assert Path(listed[str(old)]["thumbnail"]) == side / "old-style.png", listed[str(old)]

    # a new card is written into .kiln, and a rename takes the sidecars along
    new = _tiny_checkpoint(models / "fresh.pt")
    library.ensure_card(new, name="fresh")
    assert (side / "fresh.card.json").exists() and not (models / "fresh.card.json").exists()
    moved = Path(library.rename_model(str(old), "renamed-old")["path"])
    assert (side / "renamed-old.png").exists() and (side / "renamed-old.card.json").exists()
    assert not (side / "old-style.png").exists() and not (side / "old-style.card.json").exists()
    assert library.read_card(moved)["original_name"] == "old-style"

    for pt in (moved, new):
        library.delete_model_files(pt)
    left = [f.name for f in side.iterdir() if f.stem.split(".")[0] in ("renamed-old", "fresh")]
    assert not left, f"delete left sidecars behind: {left}"
    (models / "holiday.png").unlink()
    manager.clear_cache()
    print("sidecars: cards and thumbnails kept in models/.kiln; old ones read and tidied")


def check_cold_start_listing():
    """Requests that arrive together on a fresh server all see every engine.

    The desktop window opens as the server starts and asks for several things
    at once. The engines used to be marked loaded before they were, so the
    model list that lost the race came back empty: "No models yet".
    Needs a process where nothing has loaded them, hence the subprocess.
    """
    code = (
        "import threading\n"
        "from app.core import backends\n"
        "out = []\n"
        "ts = [threading.Thread(target=lambda: out.append(len(backends.available()))) for _ in range(8)]\n"
        "[t.start() for t in ts]; [t.join() for t in ts]\n"
        "print(sorted(out))\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                         text=True, env=os.environ.copy(), check=True)
    counts = eval(res.stdout.strip().splitlines()[-1])  # noqa: S307 -- our own print
    assert min(counts) == max(counts) > 0, f"some first callers saw no engines: {counts}"
    print(f"cold start: {len(counts)} callers at once all saw {counts[0]} engines")


def check_model_index():
    """A launch lists unchanged models from the on-disk index, not by reading them.

    The index is what lets the model list skip torch and every checkpoint read;
    a model edited since must be read again, and a folder that is not a model
    is remembered as one so it does not send every scan back to the backends.
    """
    import json

    from app.core import model_index
    from app.core.model_manager import manager

    models = WORKSPACE / "models"
    pt = _tiny_checkpoint(models / "indexed.pt")
    (models / "not-a-model").mkdir()
    manager.clear_cache()
    assert "indexed" in {m.name for m in manager.scan()}
    index = model_index.load()
    assert index[str(pt)]["meta"]["name"] == "indexed", index.get(str(pt))
    assert index[str(models / "not-a-model")]["meta"] is None, "a plain folder was not remembered"
    assert not any(".kiln" in k for k in index), "Kiln's own sidecar folder was indexed"

    # tamper with the stored name: an unchanged file must come from the index
    data = json.loads(model_index._file().read_text(encoding="utf-8"))
    data["entries"][str(pt)]["meta"]["name"] = "from-the-index"
    model_index._file().write_text(json.dumps(data), encoding="utf-8")
    assert "from-the-index" in {m.name for m in manager.scan()}, "an unchanged file was read again"

    # a changed file is read again, and the index follows it
    st = pt.stat()
    os.utime(pt, (st.st_atime, st.st_mtime + 5))
    names = {m.name for m in manager.scan()}
    assert "indexed" in names and "from-the-index" not in names, names
    assert model_index.load()[str(pt)]["meta"]["name"] == "indexed"

    pt.unlink()
    (models / "not-a-model").rmdir()
    assert str(pt) not in model_index.load() or "indexed" not in {m.name for m in manager.scan()}
    manager.clear_cache()
    print("model index: unchanged models listed without a read; changed ones read again")


def main():
    try:
        check_cold_start_listing()
        check_model_index()
        check_preview_failures()
        check_delete_vs_hide()
        check_scan_public_hidden()
        check_import()
        check_lineage()
        check_hidden_follows_rename()
        check_sidecars_tucked_away()
    finally:
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        shutil.rmtree(OUTSIDE, ignore_errors=True)
    print("smoke_library: ok")


if __name__ == "__main__":
    main()
