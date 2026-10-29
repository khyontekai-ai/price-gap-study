# Reproduction scope and provenance

## Paths and immutable inputs

The study no longer imports `../crop_price_forecasting`. The required daily and
monthly loaders are included under `gapfill/loaders/`. Set `PRICE_GAP_DATA_DIR`
before importing/running Python to point to your authorised raw snapshot:

```powershell
$env:PRICE_GAP_DATA_DIR = 'D:\datasets\agmarknet-snapshot'
```

The default is `data/raw/agmarknet`, relative to the release, not to the
current shell directory. `provenance/raw-snapshots.json` lists original filenames,
SHA256, byte lengths, observed date ranges, state labels and record envelopes.
Snapshot times are not invented from filesystem modification times: the source
records do not retain independently verified retrieval timestamps. Exact HTTP
response/pagination provenance is incomplete, as recorded in the manifest.

`provenance/source-requests.json` records the historical API request templates
and commodities. These are retrieval instructions, not a promise that live API
responses remain unchanged or reproduce the snapshot. No live download is run
during packaging. Check source terms before retrieving, using or sharing data.

Raw format: daily JSONL envelopes contain `date` and `rows`; records include
`state_name`, `district_name`, `mkt_name`, `mkt_id`, `cmdt_name`,
`cmdt_grp_name`, `modal_price_wt`, `arrival_qty`, `unit_name_arrival`,
`unit_name_price`, `status`. Monthly envelopes contain `commodity`, `year`,
`month`, `rows`, with `prices_<month>_<year>` fields. A monthly record's own
report takes precedence over previous-month/year copies. Reported positive
daily prices are deduplicated by date/market ID/commodity. The preserved study
uses weighted weekly means and retrospective cleaning in its main benchmark;
unit consistency remains a scientific limitation, not silently fixed here.

## Saved evaluation schema

- `outputs/<State>/evaluation.csv.gz`: method, commodity, protocol, run, online,
  series, week, actual, filled, gap, sd, in80 and optional fallback flag.
- `outputs/controls/{strict,causal,noavg}_<State>.csv.gz`: corresponding control
  scoring cells; no `online` field is needed for origin-restricted controls.
- `calendar_cells.csv.gz`: state, commodity, series, week, actual, calendar_week.
  Actual is checked against the reconstructed original panel before this lookup
  is frozen. The fast path independently checks truth on every merge.
- `outputs/inference/inference_calendar.csv`: archived corrected result for
  comparison. The fast command recomputes inference, rather than copying it.
- `artifact-manifest.json`: all artifact file hashes. Masks are represented by
  saved hidden-cell keys. The original RNG rules remain in study/control code.

The main twelve-method common-run table has a different cohort from the
baseline-specific inference rows. The release preserves this distinction.
Calendar blocks share Monday 1970-01-05 and width 13 weeks. Tests reject wrong
truth, missing dates, duplicate cells and unmatched B/C comparisons.

## Full model reruns (expensive, not executed during packaging)

Use a separate extracted checkout/output area to avoid replacing the reference
outputs. Install the CPU build of PyTorch recorded in `provenance/environment.json`
from its official index, then `requirements-full.lock.txt`. This is a snapshot of
the imputation environment, not a separately verified environment for external
Chronos/LightGBM downstream models. CPU/GPU, thread count and library changes
can affect timings and learned-baseline outputs.

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.lock.txt
python -m pip install -r requirements-full.lock.txt
$env:PRICE_GAP_DATA_DIR = 'D:\datasets\agmarknet-snapshot'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$states = 'Assam','Meghalaya','Manipur','Nagaland','Tripura','Bihar','Goa'
foreach ($state in $states) {
    python -m gapfill.study --state $state --seeds 3 --jobs 1
    if ($LASTEXITCODE -ne 0) { throw "Study failed: $state" }
}
python -m gapfill.controls --part A --jobs 1
python -m gapfill.controls --part B --jobs 1
python -m gapfill.controls --part C --jobs 1
python -m gapfill.revision_controls
python -m gapfill.inference
```

The original Assam development selection is retained in the artifact bundle as
`chosen_config.json`/`dev_selection.csv`. The source uses development salt 99991,
three transplant runs, block lengths 4/8/13, 26-week backcast, outages 4/8 and
random missingness sweeps 0.2/0.4/0.6/0.8. Per-run mask RNG seeds are
`1000 * run_id + commodity_salt`; selection/order therefore matters. Heavy
baselines have smaller predeclared run subsets. Configurations and budgets are
defined in `gapfill/study.py` and `impute.py`, not hidden in a notebook.

The legacy `--report-only` rebuilds an individual state report from saved
evaluations; it is not the full fast empirical reproduction command. The audited
manuscript is a curated revision and is not regenerated by the old paper-writing
scripts. Those scripts are deliberately not distributed as an authoritative
manuscript generator. The original downstream training/decision engine depends
on a separate forecasting project and is not included in v0.1.0. Saved downstream
predictions/summaries are retained locally for provenance, but an independent
downstream refit is **not** claimed to be self-contained in this release.

## Paper compilation

From `paper/`, with a TeX distribution containing `acmart`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main_audited.tex
pdflatex -interaction=nonstopmode -halt-on-error main_audited.tex
```

The supplied `.bbl` supports this path. If the bibliography changes, run BibTeX
between the first and later LaTeX passes. Compilation under arXiv's environment
and hosted CI are separate checks, not established by local success.

## Public replication status

Code/tests and audited manuscript are packaged. Public raw data and row-level
predictions are pending rights review, and no public repository URL is claimed.
See `RELEASE_STATUS.md` for the checks actually completed during packaging.
