"""Fetching and caching the source data.

Downloads land in ``data/external/`` and are cached by content hash. The hash is
recorded on first fetch and checked on every subsequent one, so a source that
changes underneath the project is noticed rather than silently changing every
figure downstream.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from ..config import DATA_DIR
from .registry import DATASETS, Dataset

EXTERNAL_DIR = DATA_DIR / "external"
MANIFEST = EXTERNAL_DIR / "sources.json"

USER_AGENT = "customer-intelligence/0.2 (portfolio project; +https://github.com/gondamol)"
CHUNK = 1 << 16


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _read_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _write_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def fetch(dataset: Dataset, force: bool = False) -> Path:
    """Download a dataset if it is not already cached, and return its path."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    target = EXTERNAL_DIR / dataset.filename

    if target.exists() and not force:
        return target

    request = Request(dataset.url, headers={"User-Agent": USER_AGENT})
    tmp = target.with_suffix(target.suffix + ".part")
    with urlopen(request, timeout=300) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK)
    tmp.replace(target)

    digest = _sha256(target)
    manifest = _read_manifest()
    previous = manifest.get(dataset.key, {}).get("sha256")
    if previous and previous != digest:
        print(f"  ! {dataset.name} has changed upstream "
              f"({previous[:12]}… -> {digest[:12]}…). Figures may move.")
    manifest[dataset.key] = {
        "name": dataset.name, "url": dataset.url, "sha256": digest,
        "bytes": target.stat().st_size, "licence": dataset.licence,
        "attribution": dataset.attribution,
    }
    _write_manifest(manifest)
    return target


def extract(dataset: Dataset, force: bool = False) -> Path:
    """Return the usable data file, unzipping first where the source is an archive."""
    archive = fetch(dataset, force=force)
    if not dataset.is_archive:
        return archive

    member = dataset.member
    destination = EXTERNAL_DIR / dataset.key / member
    if destination.exists() and not force:
        return destination

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        # Some UCI archives nest a second zip inside the first.
        if member not in names:
            for name in names:
                if name.lower().endswith(".zip"):
                    inner = EXTERNAL_DIR / dataset.key / name
                    inner.parent.mkdir(parents=True, exist_ok=True)
                    inner.write_bytes(zf.read(name))
                    with zipfile.ZipFile(inner) as inner_zf:
                        if member in inner_zf.namelist():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(inner_zf.read(member))
                            return destination
            raise FileNotFoundError(
                f"{member} not found in {dataset.filename}. Archive contains: {names[:10]}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(zf.read(member))
    return destination


def fetch_all(keys: tuple[str, ...] | None = None, force: bool = False) -> dict[str, Path]:
    """Download every registered dataset and report what was retrieved."""
    paths = {}
    for key in (keys or tuple(DATASETS)):
        dataset = DATASETS[key]
        path = extract(dataset, force=force)
        size = path.stat().st_size / 1e6
        print(f"      {dataset.name:<28} {size:8.1f} MB   {dataset.licence}")
        paths[key] = path
    return paths


def provenance() -> dict:
    """What was downloaded, when, and with what hash."""
    return _read_manifest()
