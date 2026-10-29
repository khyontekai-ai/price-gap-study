# Explicit upload commands

Run these from the **extracted code release**, not from the original study or an
unrelated repository. Replace `YOUR_USER` with your actual owner/organisation.
The scripts do not run automatically. GitHub and Hugging Face authentication
stays in their normal credential stores; do not paste tokens into source files.

## GitHub: recommended primary repository

Install Git and GitHub CLI, then authenticate interactively:

```powershell
gh auth login
python -m scripts.upload_github --repo YOUR_USER/price-gap-study
```

The second command is a dry run. Create and push a new private repository:

```powershell
python -m scripts.upload_github --repo YOUR_USER/price-gap-study --execute
```

Or, after reviewing the file manifest, publish directly as a new public repo:

```powershell
python -m scripts.upload_github --repo YOUR_USER/price-gap-study --public --confirm-publication --execute
```

Choose one execution command, not both: this one-time helper refuses existing
local Git history or an existing remote repository. It stages only verified
allowlisted files. Raw prices and the local artifact bundle are not included.
If a remote creation/push fails after a local commit, do not delete work or
force-push: inspect `git status`, `git remote -v` and `gh repo view` before
manually resuming or use another fresh extraction with the appropriate new name.

To make an archived release, first enable this **public** repository in Zenodo's
GitHub integration. After successful upload, in the code directory:

```powershell
git tag -a v0.1.0 -m 'Research reproducibility package v0.1.0'
git push origin v0.1.0
gh release create v0.1.0 --verify-tag --title 'v0.1.0 research reproducibility package' --notes-file RELEASE_NOTES.md
```

Zenodo issues a DOI only after its integration archives a release successfully;
the command does not invent or guarantee one. Add the real repository/DOI to
`CITATION.cff` and the paper's availability statement afterward.

## Optional Hugging Face artifact dataset

The artifact bundle includes actual source prices and derived predictions.
Redistribution rights are **not yet verified**. Do not execute an upload until
you have established permission, including for a private third-party upload.
No data licence is assigned by the code's MIT licence.

```powershell
python -m pip install huggingface_hub
python -c "from huggingface_hub import login; login()"
python -m scripts.upload_hf --repo YOUR_USER/price-gap-study-artifacts --artifacts '..\price-gap-study-v0.1.0-local-artifacts'
```

That last command is a dry run. Once rights are confirmed, a private upload is:

```powershell
python -m scripts.upload_hf --repo YOUR_USER/price-gap-study-artifacts --artifacts '..\price-gap-study-v0.1.0-local-artifacts' --confirm-data-rights --execute
```

For an explicitly approved public release, also add `--public --confirm-publication`.
It creates a new dataset only and will not silently change an existing repo's
visibility or overwrite an existing dataset. Review the dataset card and add the
actual source-specific reuse terms before public release. No execution command
here has been run by the packaging process.

Official references:
- https://cli.github.com/manual/gh_repo_create
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
