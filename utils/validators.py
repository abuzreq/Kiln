"""Input validation helpers."""
import re
from .exceptions import ValidationError

_SAFE_NAME = re.compile(r"^[A-Za-z0-9 _\-.]{1,128}$")


def safe_name(name: str, field: str = "name") -> str:
    """Validate a user-supplied name used to build a folder/file name."""
    if not name or not isinstance(name, str):
        raise ValidationError(f"{field} is required")
    name = name.strip()
    if not _SAFE_NAME.match(name):
        raise ValidationError(
            f"{field} may only contain letters, numbers, spaces, dashes, dots and underscores"
        )
    if name in (".", ".."):
        raise ValidationError(f"invalid {field}")
    return name


def require(payload: dict, *fields):
    """Ensure required fields exist in a request payload; return their values."""
    if not isinstance(payload, dict):
        raise ValidationError("expected a JSON object")
    missing = [f for f in fields if payload.get(f) in (None, "")]
    if missing:
        raise ValidationError("missing required fields: " + ", ".join(missing))
    return [payload[f] for f in fields]


def as_int(value, field: str, lo: int | None = None, hi: int | None = None) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be an integer")
    if lo is not None and v < lo:
        raise ValidationError(f"{field} must be >= {lo}")
    if hi is not None and v > hi:
        raise ValidationError(f"{field} must be <= {hi}")
    return v


def as_float(value, field: str, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number")
    if lo is not None and v < lo:
        raise ValidationError(f"{field} must be >= {lo}")
    if hi is not None and v > hi:
        raise ValidationError(f"{field} must be <= {hi}")
    return v
