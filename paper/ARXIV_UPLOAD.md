# arXiv upload package

Prepared 29 October 2026. **Not submitted to arXiv.**

## Upload this file

`arxiv_submission.zip` is the direct source upload. Its files are at the archive
root, with **`main.tex`** as the only document entry point and **`main.bbl`** as
the prebuilt BibTeX bibliography. It includes all referenced table fragments,
two TikZ schematic sources, nine plot images and the bibliography database.
No raw data, predictions, Git history, credentials, review notes, build logs,
auxiliary files or precompiled manuscript PDF are inside this ZIP.

Use **pdfLaTeX**. Standard TeX Live packages/classes are required, including
`acmart`, TikZ/PGF and `multicol`; no private style file or external code checkout
is needed. The extracted upload source compiles locally to **17 pages**, with
**11 figures**, **18 tables**, **59 references**, and the repository hyperlink.
The no-training results and mathematics have not changed. There are no unresolved
references/citations or overfull boxes in the final local log. Some underfull-box
diagnostics remain; they are not missing content or compilation failures.

## Author submission steps

1. Sign in to your arXiv author account and start a new submission.
2. Choose the subject category that best fits the applied statistical study.
   Complete endorsement if the platform requests it. No category is silently
   selected by this package.
3. Choose the manuscript distribution licence after checking intended-journal
   and funder requirements. **MIT applies to the original software, not the
   manuscript or source price data.** No paper licence has been assigned for you.
4. Upload `arxiv_submission.zip`; if asked for the main file, select `main.tex`.
5. Use `ARXIV_METADATA.txt` for the exact title, author, affiliation and abstract.
   Leave journal reference and DOI empty; neither has been assigned.
6. Inspect arXiv's own compiled PDF. Check both diagrams, every figure/table,
   bibliography/appendix ordering, author email and repository link. Local
   compilation does not guarantee identical arXiv rendering or acceptance.
7. Approve metadata and submit only after final author/rights review. Do not
   upload the saved-price artifact bundle as ancillary material without rights.

This is a source-based paper; do not replace the source ZIP with a PDF-only
submission. The PDF is provided separately for your final preview.

## What changed in this revision

- Added https://github.com/khyontekai-ai/price-gap-study to Data and Code Availability.
- Updated the adjacent reproducibility limitation to reflect the self-contained
  code release while preserving unverified data rights and missing archive DOI.
- Restored the pipeline and outer/inner masking diagrams from the original
  manuscript, with captions consistent with normalized NNLS and exploratory design.
- Restored five supporting plots, including the selected long-gap example.
  The gain plot retains its sixteen values but replaces the misleading nowcasting
  legend with retrospective filtering. No learner has been rerun.
- Balanced the final reference columns and retained separate Limitations,
  three contribution bullets, and references before all appendices.

## Refresh the already-created GitHub repository

The local Git checkout has been updated but **no follow-up commit or push was
performed**. From its existing directory in Git Bash:

```bash
python -m scripts.release_safety &&
git add -- paper README.md CITATION.cff RELEASE_STATUS.md RELEASE_NOTES.md release-manifest.json &&
git commit -m "Add repository link and restore manuscript diagrams" &&
git push origin main
```

The original full-release ZIP and manifests have been refreshed locally; the
source-price artifact bundle has not changed. Repository visibility was not
independently verified through arXiv or GitHub's public web service during this
revision; the code link is included following the author's confirmation of upload.

## Official guidance

- https://info.arxiv.org/help/submit/index.html
- https://info.arxiv.org/help/submit_tex.html
- https://info.arxiv.org/help/license/index.html
