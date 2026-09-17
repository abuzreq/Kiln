"""Filesystem helpers that never follow a link into someone else's files.

Kiln lets people link folders into the workspace instead of copying them. A
plain ``shutil.rmtree`` is not safe around those: on Windows before Python 3.12
it treats a directory junction as an ordinary folder and deletes the *target's*
contents, and anything that calls ``.resolve()`` first deletes the target itself.
Everything that removes a folder the user might have linked goes through here.
"""
import os
import shutil
import stat
from pathlib import Path


def is_link(path: str | Path) -> bool:
    """A symlink, or a Windows junction / other reparse point."""
    p = str(path)
    try:
        if os.path.islink(p):
            return True
        st = os.lstat(p)
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def unlink_link(path: str | Path):
    """Remove a link itself, leaving whatever it points at alone."""
    p = str(path)
    try:
        os.unlink(p)
    except (IsADirectoryError, PermissionError, OSError):
        # A directory symlink or junction on Windows is removed with rmdir,
        # which for a link deletes only the link.
        os.rmdir(p)


def safe_rmtree(path: str | Path):
    """Delete a folder tree, removing links inside it without following them.

    If ``path`` is itself a link, only the link goes.
    """
    path = Path(path)
    if is_link(path):
        unlink_link(path)
        return
    if not path.exists():
        return
    for entry in os.scandir(path):
        child = Path(entry.path)
        if is_link(child):
            unlink_link(child)
        elif entry.is_dir(follow_symlinks=False):
            safe_rmtree(child)
        else:
            try:
                os.unlink(child)
            except PermissionError:
                os.chmod(child, stat.S_IWRITE)
                os.unlink(child)
    os.rmdir(path)


def safe_move(src: str | Path, dest: str | Path):
    """Move a folder; a link is re-created at ``dest`` rather than its target moved."""
    src, dest = Path(src), Path(dest)
    if is_link(src):
        target = os.readlink(src) if os.path.islink(src) else os.path.realpath(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(target, dest, target_is_directory=True)
        except OSError:
            import subprocess

            subprocess.check_call(["cmd", "/c", "mklink", "/J", str(dest), str(target)],
                                  stdout=subprocess.DEVNULL)
        unlink_link(src)
        return
    shutil.move(str(src), str(dest))
