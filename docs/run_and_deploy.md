# Running and deploying the project

## Before a real-data run

1. Confirm the CDS 4003 Python brief and whether the separate RapidMiner rubric applies.
2. Obtain the required instructor approvals. The proposed problem and dataset have not been approved in this package. The brief requires problem approval before dataset selection, followed by milestone approvals before later stages.
3. Download the original public dataset using the README instructions. The authoring environment blocked direct downloads; the source data is not packaged.
4. Install dependencies in Python 3.11 or 3.12. On a memory-constrained machine use a cloud notebook with sufficient RAM or reduce the training cap only with a clearly documented sampling rationale. The code still requires >300,000 final training rows.
5. Run `python -m unittest discover -s tests -v`. These tests do not use or validate the real data.
6. Run preparation. Inspect all nonzero audit counters, quarantine reasons, source shape, and reconciliation. Investigate the flagged non-identical business-key repeats before deciding they are legitimate. A counter is not a substitute for reviewing conflicting records.
7. Run modelling and then figure generation. Read `training_warnings.json`; convergence warnings require investigation and a fresh development run, not automatic acceptance.

The full-data fit may take a substantial time. For CPU control set `OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2` in your shell when appropriate; `parallel_jobs` controls cross-validation workers. Do not change environment variables unrelated to this project. Every script stops on errors and supports a portable project directory.

## Reproducibility and reruns

Record the source hash, configuration, Python and library versions, observed sample sizes and run time. Preserve the original downloaded file. Test evaluation intentionally refuses to overwrite an existing `reports/test_metrics.json`. If a software bug requires a corrected run, archive the entire prior run outside the repository, document why it was invalid, and rerun `train_models.py --overwrite`. Do not use this flag to repeatedly improve a model based on the same test results.

## GitHub handoff

Create a repository in the team's account and add this project. Keep source archives, full data, processed data and SQLite files out of Git. Commit the small real sample, source instructions, code, technical documentation and observed output tables/figures. Check file sizes before pushing; ordinary GitHub files must remain below the limit specified in the brief. Inspect `model.joblib` size too. If any required artifact is too large, use an appropriate download mechanism and documented checksum rather than committing it as an ordinary file. Do not publish secrets.

Replace placeholder team details and results with your own verified entries. Each student must commit their own actual changes and explain their contribution. No commit history has been fabricated or backdated here. Add your instructor if the repository is private.

## Streamlit Community Cloud

1. Run the local app after real training: `python -m streamlit run app/streamlit_app.py`.
2. Test valid inputs, same-airport rejection, numeric range checks and prediction output. Confirm the displayed model metrics match generated test results.
3. Push `app/`, `src/`, `requirements.txt`, the trusted `models/model.joblib`, `models/metadata.json`, `models/lookups.json`, and the small metric/importance files used by the app to the same repository.
4. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the account that can access that repository.
5. Create the app, select the repository and branch, set the entry point to `app/streamlit_app.py`, and choose a compatible Python version matching training. The serialized estimator should use the exact pinned scikit-learn version.
6. Deploy and resolve any build/runtime errors. Confirm public access in a signed-out browser.
7. Capture genuine screenshots of the full app, a successful prediction, validation feedback, and held-out performance. Include the live URL and repository URL on the first page of the group report.

Account connection and publication were not available in the authoring environment. This package contains app source code and a real trained model with verified inference; it is not evidence of a live deployment. Streamlit itself was not installed here, so UI execution still needs verification.

## Monitoring proposal for the student's evaluation

Before any real operational use, replace the historical demonstration with a recent-data model and agree on acceptance thresholds. Monitor incoming schema, missingness, category novelty, score distribution, review workload and delayed-label performance. Define owners and retraining criteria. Explain that cost ratios in this project are hypothetical and that feature importance is not a causal explanation.

## Completed run supplied with this package

Real-data preparation, tuning, final training, holdout evaluation and inference checks are complete. All six notebooks contain executed outputs. Review the included evidence before rerunning expensive stages. A sample input for the batch review interface is in `data/sample/app_example.csv`; its ranking function has been tested on real data independently of Streamlit. The remaining deployment steps still require a working Streamlit environment and your repository/account access.
