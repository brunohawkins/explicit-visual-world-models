"""
Disk-based result cache for expensive WorldAPI calls.

Cache keys are SHA-256 hashes of serialised inputs, so:
  * Identical inputs always hit the cache.
  * Stochastic operations (RANSAC, etc.) are deterministic across runs.
  * The cache survives process restarts and can be shared across pipeline
    iterations on the same image.

Layout::

    <cache_dir>/
      segment/<hex>.pkl
      estimate_3d_points/<hex>.pkl
      estimate_3d_mesh/<hex>.glb
      predict_ground_plane/<hex>.pkl
      fit_3d_primitive/<hex>.pkl
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _arr_bytes(arr) -> bytes:
    """Stable byte representation of a numpy array (values + shape + dtype)."""
    import numpy as np
    arr = np.asarray(arr)
    return arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()


def _hash_inputs(*parts: bytes) -> str:
    """SHA-256 of an ordered sequence of byte strings.

    Each part is length-prefixed so that concatenations of different splits
    cannot collide (e.g. ``b"ab" + b"c"`` vs ``b"a" + b"bc"``).
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class DiskCache:
    """Simple content-addressed disk cache for WorldAPI results.

    Args:
        cache_dir: Root directory.  Created automatically if it does not exist.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path(self, method: str, key: str, suffix: str) -> str:
        method_dir = os.path.join(self.cache_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        return os.path.join(method_dir, f"{key}{suffix}")

    # ------------------------------------------------------------------
    # Pickle (generic Python objects — dicts, tuples, numpy arrays, …)
    # ------------------------------------------------------------------

    def get_pickle(self, method: str, key: str) -> Any | None:
        path = self._path(method, key, ".pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Cache read failed for %s/%s: %s — recomputing.", method, key[:8], exc)
            return None

    def set_pickle(self, method: str, key: str, value: Any) -> None:
        path = self._path(method, key, ".pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("Cache write failed for %s/%s: %s", method, key[:8], exc)

    # ------------------------------------------------------------------
    # GLB (trimesh.Trimesh meshes)
    # ------------------------------------------------------------------

    def get_glb(self, method: str, key: str):
        """Return a trimesh.Trimesh loaded from a cached GLB file, or None."""
        import trimesh

        path = self._path(method, key, ".glb")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return trimesh.load(
                    file_obj=io.BytesIO(f.read()), file_type="glb", force="mesh"
                )
        except Exception as exc:
            logger.warning("Cache read failed for %s/%s: %s — recomputing.", method, key[:8], exc)
            return None

    def set_glb(self, method: str, key: str, mesh) -> None:
        path = self._path(method, key, ".glb")
        try:
            glb_bytes = mesh.export(file_type="glb")
            with open(path, "wb") as f:
                f.write(glb_bytes)
        except Exception as exc:
            logger.warning("Cache write failed for %s/%s: %s", method, key[:8], exc)

    # ------------------------------------------------------------------
    # Log bundles  (metadata.json + binary artefacts per tool call)
    # ------------------------------------------------------------------

    def get_log_bundle(self, method: str, key: str) -> dict[str, bytes] | None:
        """Return a previously-saved ``{filename: bytes}`` log bundle, or None."""
        path = self._path(method + "_log", key, ".pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Log bundle read failed for %s/%s: %s", method, key[:8], exc)
            return None

    def set_log_bundle(self, method: str, key: str, bundle: dict[str, bytes]) -> None:
        """Persist a ``{filename: bytes}`` log bundle alongside the cached result."""
        path = self._path(method + "_log", key, ".pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("Log bundle write failed for %s/%s: %s", method, key[:8], exc)
