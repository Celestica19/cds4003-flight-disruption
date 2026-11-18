# Running and deploying the project

## Before a real-data run

1. Use Python 3.12 and install `requirements.txt`. The completed analysis used Python 3.12.14; the supplied trained model can be served without rerunning preparation or training.
2. To reproduce the analysis, preserve the included results first and use a fresh working copy. The training script protects existing test evaluations.
3. Download the original public dataset using the README instructions, or import the original archive already obtained by the project owner. Full source data is excluded from this package.
4. Plan sufficient RAM and disk space. Final training is uncapped by default; any change to sampling must be documented in the run configuration.
5. Run `python -m unittest discover -s tests -v`. These software tests use isolated fixtures; real-data validation is recorded separately.
6. Run preparation and inspect audit counters, source shape and row reconciliation. Investigate conflicting business keys if any occur; the supplied run found none.
7. Run modelling and figure generation. Read `training_warnings.json`; the supplied run recorded no training warnings.

The full-data fit may take a substantial time. For CPU control set `OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2` in your shell when appropriate; `parallel_jobs` controls cross-validation workers. Do not change environment variables unrelated to this project. Every script stops on errors and supports a portable project directory.

## Reproducibility and reruns

Record the source hash, configuration, Python and library versions, observed sample sizes and run time. Preserve the original downloaded file. Test evaluation intentionally refuses to overwrite an existing `reports/test_metrics.json`. If a software bug requires a corrected run, archive the entire prior run outside the repository, document why it was invalid, and rerun `train_models.py --overwrite`. Do not use this flag to repeatedly improve a model based on the same test results.

## GitHub handoff

Use the existing public repository at https://github.com/Celestica19/cds4003-flight-disruption. Place the project contents directly at the repository root, with `app/` and `requirements.txt` visible there. Keep source archives, full data, processed data and SQLite files out of Git. Commit the small real sample, source instructions, code, technical documentation and observed output tables/figures. Check file sizes before pushing; the browser-upload edition is designed to keep individual files below 25 MiB. Inspect `model.joblib` size too. If any required artifact is too large, use an appropriate download mechanism and documented checksum rather than committing it as an ordinary file. Do not publish secrets.

Complete contributor names, actual responsibilities and the live application URL. Use meaningful commit messages and attribute changes to the person who made them. Preserve the recorded metrics unless a documented new run supersedes them.

## Streamlit Community Cloud

1. Run the local app after real training: `python -m streamlit run app/streamlit_app.py`.
2. Test valid inputs, same-airport rejection, numeric range checks and prediction output. Confirm the displayed model metrics match generated test results.
3. Push `app/`, `src/`, `requirements.txt`, the trusted `models/model.joblib`, `models/metadata.json`, `models/lookups.json`, and the small metric/importance files used by the app to the same repository.
4. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the account that can access that repository.
5. Create the app, select the repository and branch, set the entry point to `app/streamlit_app.py`, and choose a compatible Python version matching training. The serialized estimator should use the exact pinned scikit-learn version.
6. Deploy and resolve any build/runtime errors. Confirm public access in a signed-out browser.
7. Capture genuine screenshots of the full app, a successful prediction, validation feedback, and held-out performance. Add the verified live URL to the README.

Account connection and publication were not available in the authoring environment. This package contains app source code and a real trained model with verified inference; it is not evidence of a live deployment. Streamlit itself was not installed here, so UI execution still needs verification.

## Monitoring and future operational evaluation

Before any real operational use, replace the historical demonstration with a recent-data model and agree on acceptance thresholds. Monitor incoming schema, missingness, category novelty, score distribution, review workload and delayed-label performance. Define owners and retraining criteria. Explain that cost ratios in this project are hypothetical and that feature importance is not a causal explanation.

## Completed run supplied with this package

Real-data preparation, tuning, final training, holdout evaluation and inference checks are complete. All six notebooks contain executed outputs. Review the included evidence before rerunning expensive stages. A sample input for the batch review interface is in `data/sample/app_example.csv`; its ranking function has been tested on real data independently of Streamlit. The remaining deployment steps still require a working Streamlit environment and your repository/account access.
