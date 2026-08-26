"""Backend registry: turn a model identifier into the engine that owns it.

Resolution order for a plain string, matching the order a user's intent gets
more explicit:

1. an existing xurdif ``.pt`` file            -> xurdif  (today's behaviour)
2. an explicit ``<backend>:<locator>`` prefix -> that backend
3. whichever registered backend ``claims()`` it, in registration order
4. nothing claims it                          -> NotFoundError, listing what was tried

Backends register lazily so a missing optional dependency degrades to "that
backend is unavailable" rather than breaking app startup.
"""
from pathlib import Path

from utils.exceptions import NotFoundError
from utils.logger import get_logger

from .base import Backend, BetaSchedule, Capabilities, ModelDescriptor, ModelRef

log = get_logger("backends")

_REGISTRY: "dict[str, Backend]" = {}
_ORDER: list[str] = []
_loaded = False


def register(backend: Backend) -> Backend:
    if backend.name not in _REGISTRY:
        _ORDER.append(backend.name)
    _REGISTRY[backend.name] = backend
    return backend


def _ensure_loaded():
    """Import the built-in backends once, tolerating optional-dependency failures."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module, attr in (
        ("app.core.backends.xurdif", "XurdifBackend"),
        ("app.core.backends.hfdiffusers", "DiffusersBackend"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            register(getattr(mod, attr)())
        except Exception as e:  # noqa: BLE001
            log.warning("backend from %s unavailable: %s", module, e)


def available() -> list[str]:
    _ensure_loaded()
    return list(_ORDER)


def get(name: str) -> Backend:
    _ensure_loaded()
    b = _REGISTRY.get(name)
    if b is None:
        for cand in _REGISTRY.values():
            if name in cand.aliases:
                return cand
        raise NotFoundError(f"unknown backend: {name}")
    return b


def _split_prefix(s: str) -> tuple[str, str] | None:
    """Split an explicit ``<backend>:<locator>`` selector.

    A Windows drive letter ("C:\models") must never be mistaken for one, so the
    prefix has to be longer than a single character *and* name a backend we
    actually have.
    """
    head, sep, rest = s.partition(":")
    if not sep or len(head) < 2 or not rest:
        return None
    try:
        backend = get(head)
    except NotFoundError:
        return None
    return backend.name, rest


def parse_ref(value) -> ModelRef:
    """Resolve a model identifier to a ModelRef. Idempotent on a ModelRef."""
    if isinstance(value, ModelRef):
        return value
    s = str(value or "").strip()
    if not s:
        raise NotFoundError("no model given")
    _ensure_loaded()

    # 1. an existing .pt is xurdif, always -- checked first so a path can never
    #    be reinterpreted by a later rule.
    try:
        p = Path(s)
        if p.suffix == ".pt" and p.exists():
            return ModelRef("xurdif", s)
    except OSError:
        pass

    # 2. explicit selector
    split = _split_prefix(s)
    if split is not None:
        name, rest = split
        locator, _, revision = rest.partition("@")
        return ModelRef(name, locator, revision or None)

    # 3. first backend that claims it
    for name in _ORDER:
        try:
            if _REGISTRY[name].claims(s):
                return ModelRef(name, s)
        except Exception as e:  # noqa: BLE001
            log.warning("backend %s failed to claim %r: %s", name, s, e)

    tried = ", ".join(_ORDER) or "none registered"
    raise NotFoundError(f"no backend recognises '{s}' (tried: {tried})")


def resolve(value) -> "tuple[Backend, ModelRef]":
    ref = parse_ref(value)
    return get(ref.backend), ref


__all__ = [
    "Backend", "BetaSchedule", "Capabilities", "ModelDescriptor", "ModelRef",
    "available", "get", "parse_ref", "register", "resolve",
]
