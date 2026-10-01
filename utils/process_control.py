"""Track long-running background jobs (training, sampling, sweeps)."""
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

# Statuses a job can still leave on its own. A queued job is not running yet,
# but it is not finished either: nothing may prune it, stamp it finished, or
# treat the GPU as idle because of it.
LIVE = ("running", "queued")


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    detail: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _pause: threading.Event = field(default_factory=threading.Event)
    _live: dict = field(default_factory=dict)
    _live_lock: threading.Lock = field(default_factory=threading.Lock)
    _state_lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None
    # optional hook (e.g. terminate a subprocess) invoked the moment cancel is
    # requested, so stopping never waits on the job producing more output.
    stopper: Callable[[], None] | None = None
    # set by the lanes while the job waits: its 1-based place in line.
    queue_position: Callable[[], int | None] | None = None

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def paused(self) -> bool:
        return self._pause.is_set()

    def pause(self) -> bool:
        if self.status != "running" or self.cancelled():
            return False
        self._pause.set()
        self.detail["paused"] = True
        self.message = "paused"
        return True

    def resume(self, updates: dict | None = None) -> bool:
        if self.status != "running":
            return False
        if updates:
            with self._live_lock:
                self._live.update(updates)
        self._pause.clear()
        self.detail["paused"] = False
        if self.message == "paused":
            self.message = "resuming..."
        return True

    def pop_updates(self) -> dict:
        with self._live_lock:
            if not self._live:
                return {}
            out = dict(self._live)
            self._live.clear()
            return out

    def start(self) -> bool:
        """Move a queued job to running. False if it was cancelled while waiting.

        Taken under the same lock as a queued cancel, so a job cannot be both
        cancelled in line and picked up by a lane.
        """
        with self._state_lock:
            if self.status != "queued":
                return False
            self.status = "running"
            self.queue_position = None
            self.detail.pop("queue", None)
            return True

    def cancel_queued(self) -> bool:
        """Cancel a job that has not started. It never reaches a lane."""
        with self._state_lock:
            if self.status != "queued":
                return False
            self._cancel.set()
            self.status = "cancelled"
            self.finished_at = time.time()
            self.queue_position = None
            self.detail.pop("queue", None)
            self.message = "cancelled before it started"
            return True

    def finish(self, status: str):
        self.status = status
        self.finished_at = time.time()

    def to_dict(self) -> dict:
        if self.finished_at is None and self.status not in LIVE:
            self.finished_at = time.time()
        where = self.queue_position
        if self.status == "queued" and where is not None:
            self.detail["queue"] = {"position": where()}
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "detail": self.detail,
            "created_at": self.created_at,
        }


class JobRegistry:
    """Thread-safe registry of active/finished jobs."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, status: str = "running") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, status=status)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs if kind is None or j.kind == kind]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job and job.cancel_queued():
            return True
        if job and job.status == "running":
            job._cancel.set()
            # wake a paused sampler so it can notice cancel
            job._pause.clear()
            job.detail["paused"] = False
            job.message = "cancelling..."
            if job.stopper is not None:
                try:
                    job.stopper()
                except Exception:  # noqa: BLE001
                    pass
            return True
        return False

    def pause(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        return job.pause()

    def resume(self, job_id: str, updates: dict | None = None) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        return job.resume(updates)

    # A finished job's ``detail`` holds base64 data-URLs, which add up fast. But
    # the results ARE what the user came back for, so only the genuinely
    # redundant copies are dropped early: `frame_raw` / `frames_raw` are the
    # pre-postproc originals, kept solely as the live postproc source. Anything
    # the user actually looks at — the frame, the batch, a sweep's cells and
    # contact sheet — survives until the whole record expires.
    HEAVY_KEYS = ("frame_raw", "frames_raw")

    def prune(self, max_age: float = 3600.0, frame_age: float = 120.0):
        now = time.time()
        with self._lock:
            for jid in list(self._jobs):
                j = self._jobs[jid]
                if j.status in LIVE:
                    continue
                age = now - (j.finished_at or j.created_at)
                if age > max_age:
                    del self._jobs[jid]
                elif age > frame_age and not j.detail.get("_pruned"):
                    for k in self.HEAVY_KEYS:
                        j.detail.pop(k, None)
                    j.detail["_pruned"] = True


registry = JobRegistry()
