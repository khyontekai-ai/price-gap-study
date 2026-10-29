"""Create a NEW GitHub repository from the verified code release. Never uploads artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

from scripts.release_safety import ROOT, verify_manifest


def run(*args):
    subprocess.run(list(args), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="OWNER/REPOSITORY; a new repository only")
    parser.add_argument("--public", action="store_true", help="Default is private")
    parser.add_argument("--confirm-publication", action="store_true", help="Required for a public repository")
    parser.add_argument("--execute", action="store_true", help="Without this, validate and print the plan only")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo):
        parser.error("Use OWNER/REPOSITORY, not a URL")
    if args.public and not args.confirm_publication:
        parser.error("Public upload requires --confirm-publication")
    manifest = verify_manifest(ROOT, allow_runtime=True)
    if (ROOT / ".git").exists():
        parser.error("This directory already has Git history. Use a fresh extracted release for this one-time uploader.")
    print(f"Verified {len(manifest['files'])} files. New {'public' if args.public else 'private'} repo: {args.repo}")
    print("Includes code, tests and audited manuscript. Excludes saved predictions and raw prices.")
    if not args.execute:
        print("Dry run only. Add --execute to create and upload.")
        return
    for command in ("git", "gh"):
        if not shutil.which(command):
            parser.error(f"Install {command} first")
    run("gh", "auth", "status")
    # Refuse to adopt any existing repository; gh create also fails closed if it exists.
    found = subprocess.run(["gh", "repo", "view", args.repo], cwd=ROOT, capture_output=True)
    if found.returncode == 0:
        parser.error("Remote repository already exists; no changes made")
    run("git", "init", "--initial-branch=main", str(ROOT))
    for relative in [*sorted(manifest["files"]), "release-manifest.json"]:
        run("git", "-C", str(ROOT), "add", "--", relative)
    run("git", "-C", str(ROOT), "-c", "user.name=Pritam Deka", "-c", "user.email=pritam@khyontekai.com",
        "commit", "-m", "Package v0.1.0 research reproducibility release")
    run("gh", "repo", "create", args.repo, "--public" if args.public else "--private", "--source", str(ROOT),
        "--remote", "origin", "--push", "--description", "Reconstructing missing mandi prices with regional factors: research code and audited manuscript")
    print(f"Uploaded https://github.com/{args.repo}. No tag or DOI was created.")


if __name__ == "__main__":
    main()
