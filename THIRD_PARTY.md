# Attribution and scope of the MIT licence

The root MIT licence applies to the original Python code, tests and software
documentation in this release. `gapfill/loaders/data.py`, `monthly.py` and
`agmarknet.py` are adapted/copied from the author's neighbouring `cropcast`
prototype; their original checksums are recorded in `provenance/code-origin.json`.
Only the path configuration changes for data loading. Scientific method code
and historical prediction artifacts are not changed during packaging.

The manuscript PDF/TeX, figures, bibliography and publication-supplied bibliography
style output are provided for research inspection, not relicensed under MIT.
Copyright in the original manuscript remains with its author. Third-party
papers cited in the bibliography are not included. TeX styles/classes are supplied
by the user's TeX distribution under their own licences, not bundled here.

Agmarknet prices and derivatives are NOT licensed under MIT. No raw prices or
row-level predictions are in this code repository. Redistribution permissions
for the separate saved-prediction bundle remain unverified; confirm them before
any third-party upload, even to a private repository. Accessibility of an API
does not itself establish redistribution rights. Dependency packages retain their
own licences and are not vendored.
