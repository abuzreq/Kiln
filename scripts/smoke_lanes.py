"""Dev smoke test for the generation lanes (no GPU, no models).

Every job's work is a stub that records when it starts and waits to be let go,
so each check controls exactly what is running when.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.engine.lanes import Lanes  # noqa: E402
from utils.process_control import LIVE, registry  # noqa: E402


class Stub:
    """Work that logs its start, then blocks until released (or raises)."""

    def __init__(self, log, name, fail=None):
        self.log, self.name, self.fail = log, name, fail
        self.go = threading.Event()
        self.started = threading.Event()
        self.calls = 0

    def __call__(self, job):
        self.calls += 1
        self.log.append(("start", self.name))
        self.started.set()
        if not self.go.wait(10):
            raise TimeoutError(f"{self.name} was never released")
        self.log.append(("end", self.name))
        err = self.fail(self.calls) if self.fail else None
        if err is not None:
            raise err


def submit(lanes, model, stub, kind="sample"):
    job = registry.create(kind, status="queued")
    lanes.submit(job, model, stub)
    return job


def wait_until(pred, what, timeout=10.0):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def settled(job):
    return job.status not in LIVE


def main():
    check_fifo()
    check_same_model_exclusion()
    check_cancel_queued()
    check_solo_retry()
    check_no_retry_alone()
    check_dispatch_after_error()
    check_job_states()
    check_streams()
    print("OK")


def check_fifo():
    """One lane is a plain queue: jobs run one at a time, in the order sent."""
    lanes, log = Lanes(1, streams=False), []
    stubs = [Stub(log, f"j{i}") for i in range(4)]
    jobs = [submit(lanes, f"m{i}", s) for i, s in enumerate(stubs)]
    assert [j.status for j in jobs] == ["running", "queued", "queued", "queued"]
    assert [lanes.position(j.id) for j in jobs] == [0, 1, 2, 3]
    for s in stubs:
        s.go.set()
    wait_until(lambda: all(settled(j) for j in jobs), "the queue to drain")
    assert [j.status for j in jobs] == ["done"] * 4
    assert log == [(k, f"j{i}") for i in range(4) for k in ("start", "end")], log
    print("fifo: one lane runs jobs one at a time, in order")


def check_same_model_exclusion():
    """Two lanes overlap different models, never one model; a busy model is passed over."""
    lanes, log = Lanes(2, streams=False), []
    a1, a2, b1 = Stub(log, "a1"), Stub(log, "a2"), Stub(log, "b1")
    ja1 = submit(lanes, "a", a1)
    ja2 = submit(lanes, "a", a2)
    jb1 = submit(lanes, "b", b1)
    # a2 is ahead of b1, but its model is busy: b1 takes the free lane.
    wait_until(b1.started.is_set, "b1 to pass the waiting a2")
    assert ja2.status == "queued" and not a2.started.is_set()
    assert lanes.busy_models() == {"a", "b"}
    b1.go.set()
    wait_until(lambda: settled(jb1), "b1 to finish")
    assert ja2.status == "queued", "a2 started while a1 still held its model"
    a1.go.set()
    wait_until(a2.started.is_set, "a2 to start once a1 let go")
    a2.go.set()
    wait_until(lambda: settled(ja2), "a2 to finish")
    assert log.index(("end", "a1")) < log.index(("start", "a2"))
    assert [j.status for j in (ja1, ja2, jb1)] == ["done"] * 3
    print("same model: never overlaps, and does not hold up other models")


def check_cancel_queued():
    """A queued job cancels at once and never runs; the job behind it moves up."""
    lanes, log = Lanes(1, streams=False), []
    first, doomed, after = Stub(log, "first"), Stub(log, "doomed"), Stub(log, "after")
    j1 = submit(lanes, "m1", first)
    jx = submit(lanes, "m2", doomed)
    j3 = submit(lanes, "m3", after)
    assert lanes.position(j3.id) == 2
    assert registry.cancel(jx.id)
    assert jx.status == "cancelled" and jx.finished_at is not None
    assert lanes.position(jx.id) is None and lanes.position(j3.id) == 1
    assert "queue" not in jx.to_dict()["detail"]
    first.go.set()
    after.go.set()
    wait_until(lambda: settled(j3), "the job behind the cancelled one")
    assert doomed.calls == 0, "a cancelled queued job ran"
    assert (j1.status, j3.status) == ("done", "done")
    print("cancel: a queued job never starts, and the line moves up")


def check_solo_retry():
    """Out of memory beside another run: back to the head, run again alone."""
    lanes, log = Lanes(2, streams=False), []
    other = Stub(log, "other")
    oom = Stub(log, "oom", fail=lambda n: RuntimeError("CUDA out of memory.") if n == 1 else None)
    later = Stub(log, "later")
    j_other = submit(lanes, "a", other)
    j_oom = submit(lanes, "b", oom)
    wait_until(oom.started.is_set, "the two runs to overlap")
    oom.go.set()                                # fails while `other` still runs
    wait_until(lambda: j_oom.status == "queued", "the failed run to be queued again")
    assert j_oom.detail.get("paused") is False and j_oom.progress == 0.0
    assert lanes.position(j_oom.id) == 1
    oom.go.clear()                              # hold the retry once it starts
    # Sent after the retry, on a free model, with a lane free: it still may not
    # pass a job waiting to run alone.
    j_later = submit(lanes, "c", later)
    time.sleep(0.2)
    assert not later.started.is_set(), "a job passed one waiting to run alone"
    assert oom.calls == 1
    other.go.set()
    wait_until(lambda: oom.calls == 2, "the retry to start once the GPU was idle")
    assert lanes.snapshot()["running"] == [j_oom.id], "something ran beside a solo job"
    time.sleep(0.1)
    assert not later.started.is_set(), "a job started beside a solo run"
    oom.go.set()
    later.go.set()
    wait_until(lambda: settled(j_later), "the job behind the solo run")
    assert (j_other.status, j_oom.status, j_later.status) == ("done", "done", "done")
    print("oom beside another run: retried alone, nothing passes or joins it")


def check_no_retry_alone():
    """Out of memory with nothing else running is the job's own error, as today."""
    lanes, log = Lanes(2, streams=False), []
    oom = Stub(log, "oom", fail=lambda n: RuntimeError("CUDA out of memory."))
    j = submit(lanes, "a", oom)
    oom.go.set()
    wait_until(lambda: settled(j), "the lone run to fail")
    assert j.status == "error" and "out of memory" in j.message and oom.calls == 1
    print("oom alone: an error, not a retry")


def check_dispatch_after_error():
    """A job that raises frees its lane and the next one runs."""
    lanes, log = Lanes(1, streams=False), []
    bad = Stub(log, "bad", fail=lambda n: ValueError("bad recipe"))
    good = Stub(log, "good")
    jb = submit(lanes, "m", bad)
    jg = submit(lanes, "m", good)
    bad.go.set()
    good.go.set()
    wait_until(lambda: settled(jg), "the job after the failed one")
    assert (jb.status, jb.message) == ("error", "bad recipe")
    assert jg.status == "done" and not lanes.busy_models()
    print("error: the lane is freed and the next job runs")


def check_job_states():
    """What a queued job looks like to the rest of the app."""
    # Imported before anything is held: on a cold start this import alone can
    # take longer than a Stub will wait to be released.
    from app.core.craft.explore import explorer

    lanes, log = Lanes(1, streams=False), []
    hold, waiting = Stub(log, "hold"), Stub(log, "waiting")
    jh = submit(lanes, "m1", hold, kind="other")    # not one the explorer yields to
    jw = submit(lanes, "m2", waiting)
    d = jw.to_dict()
    assert d["status"] == "queued" and d["detail"]["queue"] == {"position": 1}
    assert jw.finished_at is None, "a queued job was stamped finished"
    assert not registry.pause(jw.id), "a queued job paused"
    registry.prune(max_age=0, frame_age=0)
    assert registry.get(jw.id) is not None, "prune dropped a queued job"

    # The explorer steps aside for a waiting generation, not only a running one.
    assert explorer._yield_to() == "sample"

    hold.go.set()
    waiting.go.set()
    wait_until(lambda: settled(jw), "the queued job to run")
    assert "queue" not in jw.to_dict()["detail"] and jh.status == "done"
    print("job states: queued reports its place, cannot pause, is not pruned, "
          "and makes the explorer yield")


def check_streams():
    """Each lane runs on its own CUDA stream, the same one every time."""
    import torch

    if not torch.cuda.is_available():
        print("streams: skipped (no CUDA)")
        return
    lanes, seen = Lanes(2), {}
    gate = threading.Barrier(2, timeout=10)

    def work(job):
        gate.wait()                             # both lanes busy at once
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        seen.setdefault(threading.current_thread().name.split("-")[1], set()).add(
            torch.cuda.current_stream().cuda_stream)

    for _ in range(2):
        jobs = [submit(lanes, m, work) for m in ("a", "b")]
        wait_until(lambda: all(settled(j) for j in jobs), "the stream jobs")
        assert all(j.status == "done" for j in jobs), [j.message for j in jobs]
    default = torch.cuda.default_stream().cuda_stream
    streams = [next(iter(v)) for v in seen.values()]
    assert all(len(v) == 1 for v in seen.values()), f"a lane changed stream: {seen}"
    assert len(set(streams)) == 2 and default not in streams, seen
    print("streams: each lane has its own, reused across jobs")


if __name__ == "__main__":
    main()
