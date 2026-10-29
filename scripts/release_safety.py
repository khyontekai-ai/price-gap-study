"""Fail-closed manifest verification shared by the two upload entry points."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b"),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".cff", ".tex", ".bib", ".ps1", ".sh", ".csv", ".bbl"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, relative):
    name = PurePosixPath(relative)
    if not name.parts or name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe manifest path: {relative}")
    root = Path(root).resolve()
    path = root.joinpath(*name.parts)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
        raise ValueError(f"Symlink not permitted: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes release: {relative}")
    return path


def scan(path):
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
        return
    content = path.read_text(encoding="utf-8-sig")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Possible credential in {path.name}; inspect locally, not in logs")
    if re.search(r"[A-Za-z]:\\(?:Users|khyontekAI_apps|AgriTraceAI)\\", content):
        raise ValueError(f"Machine-specific path in {path.name}")


def verify_manifest(root, name="release-manifest.json", kind="code", allow_runtime=False):
    root = Path(root).resolve()
    manifest = json.loads((root / name).read_text(encoding="utf-8"))
    if manifest.get("kind") != kind or not manifest.get("files"):
        raise ValueError("Wrong or empty release manifest")
    files = manifest["files"]
    for relative, expected in files.items():
        path = checked_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed release file: {relative}")
        scan(path)
    ignored = {".git"}
    if allow_runtime:
        ignored |= {".venv", "__pycache__", ".pytest_cache", "_reproduced"}
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in ignored for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink not permitted: {relative.as_posix()}")
        if path.is_file():
            actual.add(relative.as_posix())
    if actual != set(files) | {name}:
        raise ValueError(f"Unmanifested/missing files: {sorted(actual ^ (set(files) | {name}))}")
    return manifest


if __name__ == "__main__":
    manifest = verify_manifest(ROOT, allow_runtime=True)
    print(f"Verified {len(manifest['files'])} release files; no matching credential patterns.")
