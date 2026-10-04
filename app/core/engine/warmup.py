"""Load what the first generation needs while the person is still looking around.

The first Generate of a session used to take 15-25 seconds and every later one
about 4. Nearly all of the difference was Python importing libraries on first
use, not sampling: the solvers come from ``diffusers``, whose package init
pulls in ``torch._dynamo`` (around 9 s on its own), and post-processing imports
xurdif's ``postproc26`` and with it ``kornia`` (around 5 s). On a cold disk most
of that is reading their files.

So the launcher starts this once the server is up. It does those imports in a
background thread and nothing else: no model is loaded and the GPU is not
touched, so it costs no VRAM.

Work that needs these libraries calls ``wait_ready`` first. ``diffusers``
resolves its classes lazily, and a second thread asking for one while the
package is still being set up gets "module 'diffusers' has no attribute ..."
-- a generation pressed seconds after launch failed that way. Waiting is never
slower than importing: the request would have had to do the same work.
"""
import threading
import time

from utils.logger import get_logger

log = get_logger("warmup")

_done = threading.Event()
_thread: "threading.Thread | None" = None


def _run():
    t0 = time.perf_counter()
    try:
        import torch  # noqa: F401
        import diffusers

        from app.core.engine.sampler import SAMPLERS, _postproc_fn

        for spec in SAMPLERS.values():
            getattr(diffusers, spec["cls"])  # each solver is imported on first access
        from diffusers import ModelMixin, UNet2DModel  # noqa: F401  (Diffusers models)
        _postproc_fn()
        log.info("ready to generate: libraries loaded in %.1fs", time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        # Only a head start: whatever failed here loads, or fails, on first use as before.
        log.info("warm-up stopped early: %s", e)
    finally:
        _done.set()


def start() -> threading.Thread:
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_run, name="warmup", daemon=True)
        _thread.start()
    return _thread


def wait_ready(timeout: float = 300.0) -> None:
    """Block until a running warm-up has finished; return at once if none was started.

    Never called from the warm-up thread itself, which would wait on its own
    finish -- the imports it does go through ``_postproc_fn``, which does not wait.
    """
    if _thread is None or _done.is_set() or threading.current_thread() is _thread:
        return
    _done.wait(timeout)
