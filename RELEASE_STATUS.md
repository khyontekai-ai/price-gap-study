# Release preparation tracker

Version 0.1.0, updated 29 October 2026. The author confirms uploading the code to
https://github.com/khyontekai-ai/price-gap-study. The 17-page paper now cites this
URL and restores both method diagrams and five supporting plots (11 figures
total). A direct arXiv source ZIP with metadata is prepared. No arXiv submission,
follow-up GitHub push, data upload or archive DOI was performed in this revision.

## Completed packaging

- Author approved MIT for original software; manuscript/data scope is separate.
- Allowlisted scientific source and tests copied without changing algorithms.
- Daily/monthly loaders made local; configurable data directory replaces private
  sibling-path injection. Original checksums and attribution are recorded.
- Audited manuscript, bibliography and figures copied from verified source hashes.
- Local Python/package versions pinned; raw snapshot checksums and provenance
  gaps recorded without including raw data in the code repository.
- Separate local-only prediction bundle, dataset card and calendar lookup prepared.
- No-training empirical verification entry point and safe upload scripts added.
- GitHub Actions definition added; hosted CI not yet run.

## Completed local verification

- 60,582 unique scoring/calendar cells checked against reconstructed original
  panels across all seven states; no predictions refitted.
- 45 packaged tests pass; 12 pre-existing sparse-panel empty-slice warnings remain.
- Complete fast reproduction passed in 54.96 seconds on the packaging machine:
  96 displayed main-table values, 64 full-precision matched-control values and
  1,872 corrected-inference values agree with archived expectations.
- The fast reproduction used 2,000 bootstrap draws, seed 0, common Monday anchor
  1970-01-05 and 13-week calendar blocks. Model refits: zero.
- Both GitHub and Hugging Face dry runs pass. Mocked execution tests verify
  private defaults, exact allowlists and mandatory publication/data permissions.
- Code/paper and artifact manifests pass SHA256 and credential-pattern scans.
  This is a targeted scan, not proof that arbitrary secret formats cannot exist.
- Exact file bytes are preserved by `.gitattributes`, even with Git's Windows
  autocrlf conversion enabled. A local commit/clone verifies those checksums.
- Clean ZIP extraction and Git-checkout-style test results are recorded in the
  external `release-verification.json` delivered beside the archives.

All empirical checks used the existing recorded Python 3.10.4 environment, not
a new dependency installation. Publicly hosted CI has not run. Archive integrity
hashes are in the external `release-archives.json`; the local artifact archive
must not be confused with the code-only archive suitable for GitHub.

## Remaining publication decisions

- Review and push the locally updated paper/manifest/citation files to the
  existing GitHub repository. Git Bash is sufficient; the follow-up commands in
  `paper/ARXIV_UPLOAD.md` do not require GitHub CLI.
- Verify source-data redistribution rights before third-party artifact hosting.
- Review dataset terms and card before a Hugging Face upload. No licence is
  invented for prices, and MIT does not grant data rights.
- Add real repository/archive links to citation metadata and paper only after
  those links exist. Connect Zenodo before creating the first archived release.
- A clean dependency installation, hosted CI, full model retraining and the
  external downstream forecasting engine are not verified by packaging alone.

Public code alone is not a complete empirical replication package while
prediction/raw-data access remains unresolved. No expensive model training was
started as part of this release preparation.
