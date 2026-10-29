"""Upload the separate saved-prediction bundle only after confirming data rights."""
import argparse
from pathlib import Path
import re

from scripts.release_safety import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF_USER/DATASET_NAME; new dataset only")
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--confirm-data-rights", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--confirm-publication", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo):
        parser.error("Use HF_USER/DATASET_NAME, not a URL")
    if args.public and not args.confirm_publication:
        parser.error("Public upload requires --confirm-publication")
    manifest = verify_manifest(args.artifacts, "artifact-manifest.json", "saved-predictions")
    print(f"Verified {len(manifest['files'])} artifact files. Target: {args.repo} ({'public' if args.public else 'private'})")
    if not args.execute:
        print("Dry run only. Nothing uploaded. Execution also requires --confirm-data-rights.")
        return
    if not args.confirm_data_rights:
        parser.error("Even private third-party hosting requires --confirm-data-rights")
    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()  # Authentication must already be configured; no token is printed or embedded.
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=False)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.artifacts),
                      allow_patterns=[*manifest["files"], "artifact-manifest.json"],
                      commit_message="Add v0.1.0 saved-prediction replication artifacts")
    print(f"Uploaded https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
