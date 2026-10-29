# Missing mandi prices: regional-factor reconstruction

Research companion to **Reconstructing Missing Mandi Prices in North-East India
with Regional Factors**, Pritam Deka, Khyontek AI (2026). Version 0.1.0.

This is an exploratory retrospective reconstruction study, not a deployed
current-price service. The audited manuscript, with 59 references and 18 tables,
is in `paper/main_audited.pdf`: 17 pages and 11 figures, including the restored
pipeline and masking schematics. Code repository:
https://github.com/khyontekai-ai/price-gap-study. No arXiv ID or archive DOI has
been assigned. The direct arXiv source upload is `paper/arxiv_submission.zip`;
see `paper/ARXIV_UPLOAD.md` and `paper/ARXIV_METADATA.txt` for submission steps.

## What is included

- Self-contained imputation, masking, study and information-set control code.
- Unit tests, pinned local environment records and an integrity manifest.
- Exact audited manuscript source, bibliography, figures and PDF.
- A no-training reproduction command for the 96 main-table display values,
  matched B/C contrasts and corrected calendar-block inference.
- Aggregate expected results for comparison, not a replacement for predictions.

**Row-level predictions and raw price records are deliberately separate.** The
saved-prediction artifact bundle is prepared locally but has not been cleared for
public redistribution. This code-only release cannot reconstruct the empirical
results by itself. No download URL is claimed until the artifacts are actually
released with appropriate permissions.

## Quick start: tests only

Use Python 3.10 (the recorded patch version and package inventory are in
`provenance/environment.json`). From the extracted repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-fast.lock.txt
.\.venv\Scripts\python.exe -m scripts.release_safety
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

On Linux/macOS substitute `.venv/bin/python`. The fast dependency set does not
need PyTorch, PyPOTS, an API key, a sibling project, or a data download. Exact pins
record the local working environment; they are not a cross-platform wheel lock
or a guarantee that every historical wheel remains available. Fresh dependency
installation and hosted GitHub CI have not been verified until actually run.

## Fast empirical reproduction: no retraining

Obtain the matching `price-gap-study-v0.1.0-local-artifacts` folder from the author
or an authorised future archive. Keep it outside the code repository. Then:

```powershell
.\.venv\Scripts\python.exe -m scripts.reproduce --artifacts "..\price-gap-study-v0.1.0-local-artifacts"
```

The command verifies artifact SHA256 hashes, loads saved predictions, verifies
truth against the calendar lookup, and recomputes the main table, matched B/C
contrasts and panel/calendar inference with the original 2,000 draws and seed 0.
The original masks survive in scoring-cell keys; the validated calendar lookup
preserves actual dates instead of inferring dates from panel-relative weeks.
Outputs go to `_reproduced/`; archived inputs and the manuscript are not changed.
The main table's one-decimal values and full-precision control/inference values
must match the archived expectations, or the command fails.

`--skip-inference` is a quicker **partial** check, clearly marked as such. Interval
coverage and scenario summaries are also regenerated on the main common cohort;
do not confuse those with every distinct appendix/control cohort. Original CPU
timings cannot be regenerated without rerunning models. This command does not
refit downstream forecasters or rerun hold/sell simulations.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for scope, input schema and full runs.

## Upload this release

See [UPLOAD.md](UPLOAD.md). The upload scripts default to a dry run and private
visibility, use existing CLI authentication, and never embed credentials. The
GitHub script uploads only the manifest-listed code/paper files, not the saved
prediction bundle or raw prices. Public publication is an explicit action.

## Scientific qualifications

Model development was informed by earlier results; there is no untouched final
confirmation set. Monthly release dates are assumed, raw-price units are not
fully harmonised, interval coverage is inadequate in some small panels, and
calendar-block resampling does not remove shared-helper dependence. The C
pipeline changes inputs and ensemble validation together: its adaptive contrast
is not a pure preprocessing effect. Retrospective filtered-state results are
not strict real-time learned-baseline comparisons. See the separate Limitations
section of the manuscript for the complete interpretation.

## Licence and citation

Original software is MIT-licensed, as approved by the author. Manuscript and
third-party data are not relicensed under MIT; see [THIRD_PARTY.md](THIRD_PARTY.md).
Use `CITATION.cff` and update its repository/DOI only after publication. Contact:
Pritam Deka, <pritam@khyontekai.com>.
