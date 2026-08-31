"""Content identities for development overlays layered on native wheels."""

from __future__ import annotations

import functools
from hashlib import sha256
from pathlib import Path


_PYTHON_OVERLAY_IDENTITY_PATH = Path(__file__).with_name("_python_overlay_identity.json")


@functools.cache
def get_python_overlay_stamp() -> str | None:
    """Return a stable content hash for an installed Python overlay, if present."""

    try:
        identity = _PYTHON_OVERLAY_IDENTITY_PATH.read_bytes()
    except FileNotFoundError:
        return None
    return sha256(identity).hexdigest()
