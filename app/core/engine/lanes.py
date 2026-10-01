"""Run generations a few at a time, and never two on one model.

Every user-started generation waits here in one line, first come first served,
and starts when a lane is free. More than one lane lets runs on *different*
models overlap: the models Kiln trains are small enough for several to sit in
VRAM, and runs on their own CUDA streams overlap where one alone leaves the GPU
idle. Two runs on the same model never overlap. That is what batching is for,
and it keeps two runs from sharing one module mid-forward.

A job whose model is busy is passed over, not waited on, so one model's long
run cannot hold up the line for the others.

Lanes order generations among themselves and nothing else. They never look at
training: sampling while a model trains is a supported workflow, and memory
pressure is handled by the job's own fallbacks and by the `solo` retry below,
never by refusing to start.

    lanes.submit(job, model_key(path), work)   # work(job) runs it to the end
    lanes.submit(job, {key_a, key_b}, work)    # a run that reaches several models
    lanes.position(job.id)                     # 0 running, 1 next, ... None gone
    registry.cancel(job.id)                    # a queued job never starts

`work` is the job's worker, as it was when each route started its own thread.
It may finish the job itself (status done/error/cancelled); if it returns with
the job still running, the lane marks it done, or cancelled if it was asked to
stop. An exception it lets out becomes the job's error -- except running out of
memory while another lane was busy. That puts the job back at the head of the
line marked `solo`, to start again when nothing else is running. A worker that
catches its own exceptions should re-raise out-of-memory to get that retry.
"""
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

from utils.logger import get_logger
from utils.process_control import Job

log = get_logger("lanes")

DEFAULT_LANES = 2
MAX_LANES = 3


def model_key(path: str) -> str:
    """The identity two runs must not share: the same one ``manager.evict`` uses."""
    from app.core import backends

    try:
        return str(backends.parse_ref(str(path)))
    except Exception:  # noqa: BLE001
        return str(path)


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


@dataclass(eq=False)
class _Entry:
    job: Job
    keys: frozenset[str]        # every model the run uses; none may be busy
    work: Callable[[Job], None]
    solo: bool = False          # start only with every other lane idle
    overlapped: bool = False    # another run was going at some point during this one


class Lanes:
    def __init__(self, n: int = DEFAULT_LANES, streams: bool = True):
        self._n = _clamp(n)
        self._streams_on = streams
        self._lock = threading.Lock()
        self._pending: list[_Entry] = []
        self._running: dict[int, _Entry] = {}       # lane index -> entry
        # One CUDA stream per lane, made once and kept. A fresh stream per run
        # grows reserved VRAM, because the caching allocator pools per stream.
        self._streams: dict[int, object] = {}

    # --- public ---------------------------------------------------------
    @property
    def count(self) -> int:
        return self._n

    def resize(self, n: int):
        with self._lock:
            self._n = _clamp(n)
        self._dispatch()

    def submit(self, job: Job, model_key: str | Iterable[str],
               work: Callable[[Job], None]) -> Job:
        keys = frozenset([model_key] if isinstance(model_key, str) else model_key)
        with job._state_lock:
            if job.status == "running":     # created the old way; nothing has run
                job.status = "queued"
        if job.status != "queued":
            return job                      # cancelled before it was submitted
        job.message = job.message or "waiting for a free lane"
        job.queue_position = lambda: self.position(job.id)
        with self._lock:
            self._pending.append(_Entry(job, keys, work))
        self._dispatch()
        return job

    def position(self, job_id: str) -> int | None:
        with self._lock:
            if any(e.job.id == job_id for e in self._running.values()):
                return 0
            waiting = [e for e in self._pending if e.job.status == "queued"]
        for i, e in enumerate(waiting):
            if e.job.id == job_id:
                return i + 1
        return None

    def busy_models(self) -> set[str]:
        with self._lock:
            return {k for e in self._running.values() for k in e.keys}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "lanes": self._n,
                "running": [e.job.id for _, e in sorted(self._running.items())],
                "queued": [e.job.id for e in self._pending if e.job.status == "queued"],
            }

    # --- dispatch -------------------------------------------------------
    def _dispatch(self):
        started: list[tuple[int, _Entry]] = []
        with self._lock:
            # Cancelled while waiting: drop them, they will never run.
            self._pending = [e for e in self._pending if e.job.status == "queued"]
            if any(e.solo for e in self._running.values()):
                return                      # a solo run has the GPU to itself
            busy = {k for e in self._running.values() for k in e.keys}
            for e in list(self._pending):
                free = [i for i in range(self._n) if i not in self._running]
                if not free:
                    break
                if e.solo and self._running:
                    break                   # waits for idle; nothing passes it
                if e.keys & busy:
                    continue                # passed over, not waited on
                if not e.job.start():
                    self._pending.remove(e)  # cancelled a moment ago
                    continue
                self._pending.remove(e)
                e.job.message = "starting..."    # until the work reports progress
                if self._running:
                    e.overlapped = True
                    for other in self._running.values():
                        other.overlapped = True
                lane = free[0]
                self._running[lane] = e
                busy |= e.keys
                started.append((lane, e))
                if e.solo:
                    break
        for lane, e in started:
            t = threading.Thread(target=self._run, args=(lane, e),
                                 name=f"lane-{lane}-{e.job.kind}", daemon=True)
            e.job.thread = t
            t.start()

    def _run(self, lane: int, e: _Entry):
        job = e.job
        retry = False
        try:
            with self._on_stream(lane):
                e.work(job)
        except Exception as exc:  # noqa: BLE001
            if is_oom(exc) and e.overlapped and not e.solo and not job.cancelled():
                retry = True
                log.info("job %s ran out of memory beside another run; retrying alone", job.id)
            else:
                log.warning("job %s failed: %s", job.id, exc)
                job.detail["paused"] = False
                job.message = str(exc) or type(exc).__name__
                job.status = "error"
        finally:
            if not retry and job.status == "running":
                job.status = "cancelled" if job.cancelled() else "done"
                if job.status == "done":
                    job.progress = 1.0
            with self._lock:
                self._running.pop(lane, None)
                if retry and job.cancelled():     # stopped while it failed
                    job.status, job.message = "cancelled", "stopped"
                elif retry:
                    e.solo, e.overlapped = True, False
                    job.progress = 0.0
                    job.finished_at = None
                    job.detail["paused"] = False
                    job.message = "ran out of GPU memory beside another run; waiting to run alone"
                    job.status = "queued"
                    self._pending.insert(0, e)
            self._dispatch()

    @contextmanager
    def _on_stream(self, lane: int):
        """Run a lane's job on that lane's own CUDA stream, finished on the way out.

        Threads in one process otherwise share the default stream, and their
        kernels queue behind each other. Synchronising at the end means nothing
        a run launched is still reading a model once the run has let go of it.
        """
        stream = self._stream(lane)
        if stream is None:
            yield
            return
        import torch

        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            try:
                stream.synchronize()
            except Exception:  # noqa: BLE001
                pass

    def _stream(self, lane: int):
        if not self._streams_on:
            return None
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        with self._lock:
            if lane not in self._streams:
                self._streams[lane] = torch.cuda.Stream()
            return self._streams[lane]


def _clamp(n: int) -> int:
    return max(1, min(MAX_LANES, int(n)))


lanes = Lanes()


def enqueue(job: Job, model_paths: str | Iterable[str], work: Callable[[Job], None]) -> Job:
    """Put a job in the app's line, keyed by every model it will sample from."""
    paths = [model_paths] if isinstance(model_paths, str) else list(model_paths)
    return lanes.submit(job, {model_key(p) for p in paths}, work)


def configure(n: int):
    """Set the lane count, and let the model cache hold one model more.

    The extra slot lets the next queued model load while every lane runs,
    without evicting a model a lane is using.
    """
    from app.core.model_manager import manager

    lanes.resize(n)
    manager.MAX_LOADED = lanes.count + 1
