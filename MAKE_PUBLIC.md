# Before GitHub publication

1. Run `python -m pytest -q` on a clean Python 3.10 virtual environment with `requirements-core.txt`.
2. Run `python scripts/verify_model.py --online` to check official pinned `config.json` (requires network) and keep its output privately or publish the public architecture verification report.
3. Confirm any `data/raw`, `data/processed`, `.env`, manuscript DOCX, private spreadsheet, audio and exported participant data are NOT committed (see `.gitignore`).
4. Do not use repository demo tests as classroom study outcome evidence. No `results/` are checked in.
5. Complete the real ethical approval / waiver statement, participant consent scope, rights to publish any datasets, teacher qualification checks, and data access/licensing records before scientific claims are published.
6. Publish a tagged repository release and include the final commit SHA and URL in the manuscript. Do not assert the repo alone validates all reported effects.
7. Keep exact source dataset release/access date and the processing manifest hashes in an appropriate data availability appendix.

Useful upload workflow:

```bash
git init
git add .
git status --short
git commit -m "Add research reproducibility toolkit"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/PEL-Framework-Repro.git
git push -u origin main
```

Repository name suggestion: `PEL-Framework-Repro`.
