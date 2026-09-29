"""Learning-rate plans: one schedule format, read by both training loops.

``<run>/lr_plan.json`` describes the future, ``<run>/lr_events.jsonl`` records the
past. Kiln writes the plan; the training loop re-reads it when its mtime changes
and writes ``opt.param_groups[*]["lr"]``. Choosing a curve before a run and
dropping the rate mid-run are therefore the same code path, and the plan survives
a resume, a continue, and a Kiln restart.

**Pure stdlib, deliberately.** ``vendor/xurdif/xurdiftrainer.py`` imports this
module inside the training subprocess, so it must not pull in torch, Flask, or
anything that touches the workspace. ``utils.exceptions`` is the one import, and
it is stdlib-pure too.

The plan on disk is always absolute steps. Presets are defined as fractions of the
run and compiled at launch: a fractional plan stored on disk would silently
rescale the whole schedule the moment someone bumped ``train_steps`` on a
continue. ``preset`` and ``params`` are provenance -- read by the UI, ignored by
the evaluator -- so retuning a preset later never changes what an old run did.

See ``docs/lr-schedule-design.md``.
"""
import json
import math
import os
import tempfile
from pathlib import Path

from utils.exceptions import ValidationError

VERSION = 1

PLAN_NAME = "lr_plan.json"
EVENTS_NAME = "lr_events.jsonl"

# Shapes a segment can take. `const` holds until the next segment; the ramps
# interpolate from `lr` to `to_lr` between `from` and `until` and then hold
# `to_lr`; `cyclic` is a triangle whose cycle starts at `lr`, the high point.
KINDS = ("const", "linear", "cosine", "exp", "cyclic")

# Structural bounds. Stricter than a hand-edit needs and looser than the routes,
# which bound the *base* rate to [1e-6, 1] -- a preset's own floor is allowed to
# land below that (L/50 of a 5e-5 fine-tune is 1e-6 exactly).
LR_MIN = 1e-8
LR_MAX = 1.0


# --- formatting -------------------------------------------------------------

def fmt_lr(x: float) -> str:
    """``5e-4``, ``2.5e-5`` -- mantissa-and-exponent, which ``%g`` will not do
    consistently (it prints 5e-4 as 0.0005 and 5e-5 as 5e-05)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "?"
    if x <= 0:
        return "0"
    e = math.floor(math.log10(x))
    m = x / (10.0 ** e)
    # Guard the log10 edge: 1e-4 can come back as 0.99999...e-4.
    if m >= 9.995:
        m, e = m / 10.0, e + 1
    ms = f"{m:.1f}".rstrip("0").rstrip(".")
    return f"{ms}e{e}"


def fmt_step(n) -> str:
    n = int(n)
    if n >= 1000 and n % 100 == 0:
        return f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:g}k"
    return str(n)


def _sig(x: float, digits: int = 2) -> float:
    """Round to N significant figures, so a compiled plan reads as 1.0e-4 rather
    than 9.99998e-05 in the file, the log line and the chart label alike."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x))
    q = round(x / (10.0 ** e), digits - 1)
    return max(LR_MIN, q * (10.0 ** e))


def _snap(step: float, save_every: int) -> int:
    """Put a breakpoint on a snapshot boundary, so there is always a checkpoint
    from just before a drop to compare against."""
    save_every = max(int(save_every or 1), 1)
    return max(save_every, int(round(step / save_every)) * save_every)


# --- presets ----------------------------------------------------------------
#
# Each builds a segment list from the base rate L and total steps T. Fractions of
# T, not absolute steps, because Kiln's own presets span 120k-400k steps.

def _p_constant(L, T, se):
    return [{"from": 0, "kind": "const", "lr": L}]


def _p_drops_2(L, T, se):
    return [
        {"from": 0, "kind": "const", "lr": L},
        {"from": _snap(0.30 * T, se), "kind": "const", "lr": _sig(L / 5)},
        {"from": _snap(0.65 * T, se), "kind": "const", "lr": _sig(L / 10)},
    ]


def _p_drops_3(L, T, se):
    return _p_drops_2(L, T, se) + [
        {"from": _snap(0.85 * T, se), "kind": "const", "lr": _sig(L / 50)},
    ]


def _p_cosine_floor(L, T, se):
    return [{"from": 0, "kind": "cosine", "lr": L,
             "to_lr": _sig(L / 20), "until": int(T)}]


def _p_cyclic(L, T, se):
    # An even multiple of save_every, so period/2 is one too and a snapshot lands
    # on every trough as well as every peak.
    se = max(int(se or 1), 1)
    period = max(2 * se, int(round(0.1 * T / (2 * se))) * 2 * se)
    return [{"from": 0, "kind": "cyclic", "lr": L,
             "to_lr": _sig(L / 10), "period": period}]


PRESETS: dict[str, dict] = {
    "constant": {
        "label": "Constant",
        "blurb": "One rate for the whole run.",
        "build": _p_constant,
    },
    "drops-2": {
        "label": "Two drops",
        "blurb": "Full rate while the model learns general features, then a fifth, "
                 "then a tenth. The engine author's own habit, which is where the "
                 "numbers come from.",
        "build": _p_drops_2,
    },
    "drops-3": {
        "label": "Three drops",
        "blurb": "Two drops, plus a very low rate for the last stretch to settle "
                 "fine detail without moving the image around.",
        "build": _p_drops_3,
    },
    "cosine-floor": {
        "label": "Smooth decay",
        "blurb": "Eases from the full rate down to a twentieth across the run, with "
                 "no steps. Nothing to time, but nothing to read off the curve either.",
        "build": _p_cosine_floor,
    },
    "cyclic": {
        "label": "Cyclical",
        "blurb": "Sweeps between the full rate and a tenth, ten times over the run. "
                 "Snapshots land on both the peaks and the troughs -- pick from the "
                 "troughs, a peak snapshot is a genuinely worse model.",
        "build": _p_cyclic,
    },
}

DEFAULT_PRESET = "constant"


# --- compiling and validating ----------------------------------------------

def compile_plan(spec, *, lr: float, train_steps: int, save_every: int) -> dict:
    """Turn a spec into an absolute-step plan.

    ``spec`` is ``{"preset": id}``, or ``{"segments": [...]}`` for a hand-built
    one, or an already-compiled plan (which is re-validated and passed through).
    A falsy spec compiles the constant plan, so every run gets a plan file and
    there is one code path in the loops.
    """
    spec = spec or {}
    if not isinstance(spec, dict):
        raise ValidationError("learning-rate plan must be an object")
    lr = float(lr)
    train_steps = max(int(train_steps or 1), 1)
    save_every = max(int(save_every or 1), 1)

    segments = spec.get("segments")
    if "segments" in spec and not (isinstance(segments, list) and segments):
        # An absent key means "use a preset"; an empty one is a caller mistake.
        raise ValidationError("a learning-rate plan needs at least one segment")
    preset = spec.get("preset") or (None if segments else DEFAULT_PRESET)

    if segments:
        # A hand-built or round-tripped plan: take the segments as given. `preset`
        # rides along as provenance when the caller kept it.
        segs = [dict(s) for s in segments]
    else:
        if preset not in PRESETS:
            known = ", ".join(PRESETS)
            raise ValidationError(f"unknown learning-rate schedule '{preset}' (known: {known})")
        segs = PRESETS[preset]["build"](lr, train_steps, save_every)

    plan = {
        "version": VERSION,
        "preset": preset,
        "params": {"base_lr": lr, "train_steps": train_steps, "save_every": save_every},
        "warmup": int(spec.get("warmup") or 0),
        "segments": segs,
    }
    return validate(plan)


def override_from(plan: dict, step: int, lr: float) -> dict:
    """Hold ``lr`` from ``step`` onward, keeping what came before.

    This is what the one-click drop does. Replacing the whole plan would work
    just as well for the optimizer, but the earlier segments are the run's own
    history: keeping them means the plan still reads as "5e-4 from the start, then
    1e-4 from where I changed my mind".
    """
    step = max(int(step), 0)
    segs = [dict(s) for s in (plan or {}).get("segments") or []
            if int(s.get("from", 0)) < step]
    segs.append({"from": step, "kind": "const", "lr": float(lr)})
    if segs[0]["from"] != 0:
        segs[0] = dict(segs[0], **{"from": 0})
    out = dict(plan or {})
    out.update({"version": VERSION, "segments": segs, "preset": "custom",
                "warmup": int((plan or {}).get("warmup") or 0)})
    out.setdefault("params", {})
    return validate(out)


def validate(plan: dict) -> dict:
    """Check a plan can be evaluated, and normalise its numbers. Raises
    ValidationError. Returns the plan so callers can write ``plan = validate(p)``."""
    if not isinstance(plan, dict):
        raise ValidationError("learning-rate plan must be an object")
    segs = plan.get("segments")
    if not isinstance(segs, list) or not segs:
        raise ValidationError("a learning-rate plan needs at least one segment")

    warmup = int(plan.get("warmup") or 0)
    if warmup < 0:
        raise ValidationError("warmup cannot be negative")
    plan["warmup"] = warmup

    prev_from = None
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            raise ValidationError(f"segment {i + 1} must be an object")
        kind = s.get("kind") or "const"
        if kind not in KINDS:
            raise ValidationError(
                f"segment {i + 1}: unknown kind '{kind}' (known: {', '.join(KINDS)})")
        s["kind"] = kind

        try:
            start = int(s.get("from", 0))
        except (TypeError, ValueError):
            raise ValidationError(f"segment {i + 1}: 'from' must be a whole number of steps")
        if start < 0:
            raise ValidationError(f"segment {i + 1}: 'from' cannot be negative")
        if i == 0 and start != 0:
            raise ValidationError("the first segment must start at step 0")
        if prev_from is not None and start <= prev_from:
            raise ValidationError(
                f"segment {i + 1} starts at step {start}, which is not after the "
                f"previous segment's {prev_from} -- segments must be in order")
        prev_from = start
        s["from"] = start

        s["lr"] = _check_lr(s.get("lr"), i, "lr")
        if kind in ("linear", "cosine", "exp"):
            s["to_lr"] = _check_lr(s.get("to_lr"), i, "to_lr")
            try:
                until = int(s.get("until", 0))
            except (TypeError, ValueError):
                raise ValidationError(f"segment {i + 1}: 'until' must be a whole number of steps")
            if until <= start:
                raise ValidationError(
                    f"segment {i + 1}: a {kind} ramp needs 'until' after 'from' "
                    f"(got {until}, which is not after {start})")
            s["until"] = until
        elif kind == "cyclic":
            s["to_lr"] = _check_lr(s.get("to_lr"), i, "to_lr")
            try:
                period = int(s.get("period", 0))
            except (TypeError, ValueError):
                raise ValidationError(f"segment {i + 1}: 'period' must be a whole number of steps")
            if period < 2 or period % 2:
                raise ValidationError(
                    f"segment {i + 1}: a cyclical period must be an even number of "
                    f"steps of at least 2 (got {period})")
            s["period"] = period
    return plan


def _check_lr(v, i: int, field: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"segment {i + 1}: '{field}' must be a number")
    if not (LR_MIN <= x <= LR_MAX):
        raise ValidationError(
            f"segment {i + 1}: '{field}' is {x}, outside the usable range "
            f"{fmt_lr(LR_MIN)} to {LR_MAX}")
    return x


# --- evaluating -------------------------------------------------------------

def segment_index(plan: dict, step: int) -> int:
    """Index of the segment in force at ``step``, or -1 for an empty plan."""
    segs = (plan or {}).get("segments") or []
    idx = -1
    for i, s in enumerate(segs):
        if int(s.get("from", 0)) <= step:
            idx = i
        else:
            break
    return idx


def lr_at(plan: dict, step: int) -> float:
    """The learning rate this plan asks for at ``step``."""
    segs = (plan or {}).get("segments") or []
    i = segment_index(plan, step)
    if i < 0:
        # Before the first segment (which validate() pins to 0, so only reachable
        # for an unvalidated plan): hold the first value rather than guess.
        return float(segs[0]["lr"]) if segs else 0.0
    s = segs[i]
    kind = s.get("kind", "const")
    lo_hi = float(s["lr"])

    if kind == "const":
        lr = lo_hi
    elif kind == "cyclic":
        period = int(s["period"])
        to_lr = float(s["to_lr"])
        q = ((step - int(s["from"])) % period) / period
        # q=0 is the high point, q=0.5 the low one, so every trough sits at
        # from + period/2 + k*period -- a save_every multiple by construction.
        lr = to_lr + (lo_hi - to_lr) * abs(1.0 - 2.0 * q)
    else:
        start, until = int(s["from"]), int(s["until"])
        to_lr = float(s["to_lr"])
        p = (step - start) / max(until - start, 1)
        p = min(max(p, 0.0), 1.0)          # hold the end value past `until`
        if kind == "linear":
            lr = lo_hi + (to_lr - lo_hi) * p
        elif kind == "cosine":
            lr = to_lr + (lo_hi - to_lr) * 0.5 * (1.0 + math.cos(math.pi * p))
        else:  # exp
            lr = lo_hi * (to_lr / lo_hi) ** p if lo_hi > 0 else to_lr

    warmup = int((plan or {}).get("warmup") or 0)
    if warmup > 0:
        lr *= min(step + 1, warmup) / warmup
    return lr


def summarize(plan: dict) -> str:
    """One line for the UI and run.json, e.g. ``5e-4 -> 1e-4 @84k -> 5e-5 @182k``."""
    segs = (plan or {}).get("segments") or []
    if not segs:
        return ""
    parts = []
    for i, s in enumerate(segs):
        kind = s.get("kind", "const")
        if kind == "const":
            # A plain breakpoint reads best with its step trailing: "1e-4 @84k".
            parts.append(f"{fmt_lr(s['lr'])}" + ("" if i == 0 else f" @{fmt_step(s['from'])}"))
            continue
        # A shape already ends in a step of its own, so its start goes in front.
        at = "" if i == 0 else f"from {fmt_step(s['from'])}, "
        if kind == "cyclic":
            parts.append(f"{at}{fmt_lr(s['lr'])} <-> {fmt_lr(s['to_lr'])} "
                         f"every {fmt_step(s['period'])}")
        else:
            word = {"cosine": "smooth", "linear": "linear", "exp": "exponential"}[kind]
            parts.append(f"{at}{fmt_lr(s['lr'])} -> {fmt_lr(s['to_lr'])} by "
                         f"{fmt_step(s['until'])} ({word})")
    out = " -> ".join(parts) if all(s.get("kind", "const") == "const" for s in segs) \
        else ", then ".join(parts)
    if len(segs) == 1 and segs[0].get("kind", "const") == "const":
        out += " throughout"
    warmup = int((plan or {}).get("warmup") or 0)
    if warmup:
        out += f", after a {fmt_step(warmup)}-step warm-up"
    return out


def marks_for_chart(events: list, plan: dict) -> list[dict]:
    """Rate changes as chart annotations: the past from events, the future from
    the plan.

    One mark per breakpoint, and exactly one at the start of a ramp or a cycle --
    never one per step or per cycle, which would be a thicket on a 280k-step run.
    A plan rewritten mid-run would otherwise claim breakpoints that never
    happened, so plan segments are only used from after the last recorded event.
    """
    marks: list[dict] = []
    seen_seg = None
    last_event_step = -1
    for e in events or []:
        try:
            step = int(e.get("step"))
            lr = float(e.get("lr"))
        except (TypeError, ValueError):
            continue
        why = e.get("why") or "plan"
        seg = e.get("seg")
        last_event_step = max(last_event_step, step)
        # Collapse the run of events a ramp produces inside one segment; a live
        # edit is always its own mark.
        if why == "plan" and seg is not None and seg == seen_seg:
            continue
        # A live edit claims its segment too, so the loop's own event confirming
        # the change a step later does not draw a second mark beside it.
        seen_seg = seg
        marks.append({"step": step, "lr": lr, "why": why,
                      "label": (f"edited -> {fmt_lr(lr)}" if why != "plan" else fmt_lr(lr))})

    for s in (plan or {}).get("segments") or []:
        start = int(s.get("from", 0))
        if start <= last_event_step:
            continue
        kind = s.get("kind", "const")
        if kind == "const":
            label = fmt_lr(s["lr"])
        elif kind == "cyclic":
            label = f"{fmt_lr(s['lr'])} <-> {fmt_lr(s['to_lr'])}"
        else:
            label = f"{fmt_lr(s['lr'])} -> {fmt_lr(s['to_lr'])}"
        marks.append({"step": start, "lr": float(s["lr"]), "why": "planned", "label": label})

    marks.sort(key=lambda m: m["step"])
    return marks


# --- the files --------------------------------------------------------------

def path_for(run_dir) -> Path:
    return Path(run_dir) / PLAN_NAME


def events_path(run_dir) -> Path:
    return Path(run_dir) / EVENTS_NAME


def write(run_dir, plan: dict) -> Path:
    """Write the plan atomically: the loop may be reading it at any moment, and a
    half-written file would look like a parse failure."""
    p = path_for(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".lr_plan-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(plan, f, indent=2)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def read(run_dir) -> dict | None:
    p = path_for(run_dir)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def append_event(run_dir, step: int, lr: float, why: str = "plan") -> None:
    """Record what the rate actually became, and when. Append-only: this is the
    one artifact whose loss would be unrecoverable, so it survives the log
    rotation and any number of plan rewrites."""
    rec = {"step": int(step), "lr": float(lr), "why": why,
           "seg": segment_index(read(run_dir) or {}, int(step))}
    try:
        p = events_path(run_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # telemetry, never worth failing a run over


def read_events(run_dir) -> list[dict]:
    out: list[dict] = []
    try:
        text = events_path(run_dir).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


class Watcher:
    """Re-reads the plan when it changes and says what the rate should be.

    The single point of policy: both training loops call ``lr_for`` and nothing
    else. One ``os.stat`` per optimizer step, against a step that takes hundreds
    of milliseconds at the smallest useful size.
    """

    # Report a change when the segment changes, when the plan was rewritten, or
    # when a ramp has moved the rate this far since the last report. Without the
    # last rule a cosine decay would log a line every single step; with it, a
    # full L -> L/20 decay reports about sixteen times.
    REPORT_RATIO = 1.2

    def __init__(self, run_dir, fallback_lr: float):
        self._dir = Path(run_dir)
        self._path = path_for(run_dir)
        self._fallback = float(fallback_lr)
        self._key: tuple | None = None
        self._plan: dict | None = None
        self._last_lr: float | None = None
        self._last_seg: int | None = None
        self._warned = False
        self._reload()

    @property
    def plan(self) -> dict | None:
        return self._plan

    def _stat_key(self):
        try:
            st = os.stat(self._path)
        except OSError:
            return None
        # mtime_ns and size, not mtime: NTFS granularity would let two writes of
        # the same length inside one tick look identical.
        return (st.st_mtime_ns, st.st_size)

    def _reload(self) -> bool:
        key = self._stat_key()
        if key == self._key:
            return False
        self._key = key
        plan = read(self._dir)
        try:
            plan = validate(plan) if plan is not None else None
        except ValidationError as e:
            plan, why = None, str(e)
        else:
            why = "could not be read"
        if plan is None:
            if not self._warned:
                self._warned = True
                print(f"lr plan at {self._path} {why}; {self._holding()}", flush=True)
            return False
        self._plan = plan
        self._warned = False
        return True

    def _holding(self) -> str:
        if self._plan is None:
            return f"holding {fmt_lr(self._fallback)}"
        return "keeping the plan already in force"

    def lr_for(self, step: int) -> tuple[float, bool]:
        """``(lr, changed)`` for this step. ``changed`` means worth logging."""
        reloaded = self._reload()
        if self._plan is None:
            lr, seg = self._fallback, -1
        else:
            lr, seg = lr_at(self._plan, step), segment_index(self._plan, step)

        if self._last_lr is None:
            changed = True
        elif seg != self._last_seg:
            changed = True
        elif reloaded and lr != self._last_lr:
            changed = True
        else:
            ratio = lr / self._last_lr if self._last_lr else float("inf")
            changed = not (1.0 / self.REPORT_RATIO <= ratio <= self.REPORT_RATIO)

        if changed:
            self._last_lr, self._last_seg = lr, seg
        return lr, changed
