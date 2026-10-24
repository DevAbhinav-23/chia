"""Content-addressed, on-disk cache of OpenRAM results.

OpenRAM takes minutes to hours per macro, and the
same SRAM geometries recur across every branch of a design-space loop, so
``run_openram`` consults this cache before launching the compiler and stores
every successful run in it.

Layout on disk (``cache_dir`` lives on the worker node; it is persistent only
if the cluster yaml bind-mounts a host directory there)::

    <cache_dir>/
        <key>/
            meta.json          # key inputs, macro name, timings, openram version
            result.json        # the serialized OpenRAMResult (minus big blobs)
            <macro>.v          # OpenRAM collateral, verbatim
            <macro>.lef
            <macro>.py
            <macro>_<corner>.lib   (one per corner)
            <macro>.html / <macro>.log  (optional)
            <macro>.gds        (only with store_gds)

The key is the SHA-256 of the *deterministic* part of the rendered OpenRAM
config (technology, geometry, ports, write size, corners, every option that
changes the macro) plus the OpenRAM version. Paths, temp dirs and timeouts are
excluded, so entries are portable: a cache populated on one node, or exported
with :func:`export_openram_cache`, is valid anywhere the same OpenRAM version
runs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time

META_NAME = "meta.json"
RESULT_NAME = "result.json"
ENV_CACHE_DIR = "CHIA_OPENRAM_CACHE_DIR"


def default_cache_dir() -> str | None:
    """The cache directory the worker image advertises (``CHIA_OPENRAM_CACHE_DIR``), or None."""
    return os.environ.get(ENV_CACHE_DIR) or None


def cache_key(config_body: str, openram_version: str) -> str:
    """SHA-256 over the deterministic config body and the OpenRAM version."""
    h = hashlib.sha256()
    h.update(openram_version.encode())
    h.update(b"\0")
    h.update(config_body.encode())
    return h.hexdigest()


class OpenRAMCache:
    """Thin wrapper over the on-disk layout above. Safe for concurrent writers
    (entries are built in a temp dir and atomically renamed into place)."""

    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    # -- lookup -----------------------------------------------------------
    def entry_dir(self, key: str) -> str:
        return os.path.join(self.cache_dir, key)

    def has(self, key: str) -> bool:
        d = self.entry_dir(key)
        return os.path.isfile(os.path.join(d, META_NAME)) and os.path.isfile(os.path.join(d, RESULT_NAME))

    def load(self, key: str) -> dict | None:
        """Return the stored result dict (with file contents inlined) or None."""
        if not self.has(key):
            return None
        d = self.entry_dir(key)
        with open(os.path.join(d, RESULT_NAME)) as f:
            result = json.load(f)
        files = {}
        for fname in sorted(os.listdir(d)):
            if fname in (META_NAME, RESULT_NAME) or fname.endswith(".gds"):
                continue
            with open(os.path.join(d, fname), errors="replace") as f:
                files[fname] = f.read()
        result["files"] = files
        return result

    def meta(self, key: str) -> dict | None:
        p = os.path.join(self.entry_dir(key), META_NAME)
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return json.load(f)

    # -- store ------------------------------------------------------------
    def store(self, key: str, result: dict, files: dict[str, str | bytes], meta: dict) -> str:
        """Write an entry atomically. ``files`` maps basename -> content."""
        final = self.entry_dir(key)
        tmp = tempfile.mkdtemp(prefix=f".{key[:12]}_", dir=self.cache_dir)
        try:
            for fname, content in files.items():
                mode = "wb" if isinstance(content, bytes) else "w"
                with open(os.path.join(tmp, fname), mode) as f:
                    f.write(content)
            with open(os.path.join(tmp, RESULT_NAME), "w") as f:
                json.dump(result, f, indent=1, sort_keys=True)
            meta = dict(meta, key=key, stored_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            with open(os.path.join(tmp, META_NAME), "w") as f:
                json.dump(meta, f, indent=1, sort_keys=True)
            if os.path.isdir(final):
                shutil.rmtree(final)
            os.rename(tmp, final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return final

    def remove(self, key: str) -> None:
        shutil.rmtree(self.entry_dir(key), ignore_errors=True)

    # -- inventory / transport -------------------------------------------
    def keys(self) -> list[str]:
        return sorted(k for k in os.listdir(self.cache_dir)
                      if not k.startswith(".") and self.has(k))

    def inventory(self) -> list[dict]:
        """One meta dict per entry (macro name, geometry, version, timings)."""
        out = []
        for k in self.keys():
            m = self.meta(k)
            if m:
                out.append(m)
        return out

    def export_tar(self, keys: list[str] | None = None) -> bytes:
        """Tar+gz the selected entries (default: all) into memory."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for k in (keys if keys is not None else self.keys()):
                if self.has(k):
                    tar.add(self.entry_dir(k), arcname=k)
        return buf.getvalue()

    def import_tar(self, blob: bytes, overwrite: bool = False) -> list[str]:
        """Unpack a tarball made by :meth:`export_tar`; returns imported keys."""
        imported = []
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = tar.getmembers()
            keys = sorted({m.name.split("/", 1)[0] for m in members if m.name and not m.name.startswith(("/", "."))})
            with tempfile.TemporaryDirectory(dir=self.cache_dir, prefix=".import_") as tmp:
                safe = [m for m in members if not (m.name.startswith("/") or ".." in m.name.split("/"))]
                tar.extractall(tmp, members=safe)
                for k in keys:
                    src = os.path.join(tmp, k)
                    if not os.path.isdir(src):
                        continue
                    if self.has(k) and not overwrite:
                        continue
                    dst = self.entry_dir(k)
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
                    imported.append(k)
        return imported
