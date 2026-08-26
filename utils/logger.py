"""Centralized logging for Kiln."""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "kiln") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("kiln")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    if name == "kiln":
        return logging.getLogger("kiln")
    return logging.getLogger(f"kiln.{name}")
