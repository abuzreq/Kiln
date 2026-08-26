"""Domain exceptions for Kiln. All carry an HTTP status for the API layer."""


class KilnError(Exception):
    """Base class for expected, user-facing errors."""

    status_code = 400

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(KilnError):
    status_code = 404


class ValidationError(KilnError):
    status_code = 422


class GPUUnavailableError(KilnError):
    """Raised when a CUDA-only operation is attempted without a usable GPU."""

    status_code = 503


class EngineError(KilnError):
    """Raised when the underlying diffusion engine fails."""

    status_code = 500


class IncompatibleModelError(KilnError):
    """Raised when two models cannot be merged / a bend preset does not match."""

    status_code = 409
