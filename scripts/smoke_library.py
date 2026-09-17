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


def main():
    try:
        check_preview_failures()
        check_delete_vs_hide()
        check_scan_public_hidden()
    finally:
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        shutil.rmtree(OUTSIDE, ignore_errors=True)
    print("smoke_library: ok")


if __name__ == "__main__":
    main()
