# Historical flight disruption prediction

Python implementation for the CDS 4003 Data Science for Business Application final project. The proposed use is to prioritize passenger-support review before scheduled departure using a flight's planned schedule.

**Current status: real-data preparation, model training, evaluation and inference checks completed.** The uploaded original archive contains 5,819,079 flight records and 31 columns. All 5,819,079 records were audited and retained. The selected model was fitted on all 3,888,115 eligible training records. Public deployment, instructor approvals, student-authored assessments and the separate RapidMiner requirement remain outstanding.

## Business problem and proposed evaluation

The target is whether a scheduled US domestic flight is cancelled, diverted, or arrives at least 15 minutes late. The intended user is a passenger-support planning team, not a flight-safety or dispatch system. Predictors are restricted to information available before departure. Actual operational times, delays, causes, tail assignments and outcomes are excluded from model inputs.

Proposed measurable targets, subject to instructor approval: demonstrate higher chronological cross-validation average precision than a prevalence baseline, and at least 1.25 times random-ranking lift when prioritizing 10% of flights. The observed holdout ranking lift was **1.604×**, exceeding the proposed 1.25× target. Chronological CV also exceeded the baseline. These are analytical acceptance checks, not evidence of realized financial benefit or instructor approval. Measure the number of disruptions captured and false alerts at that capacity. The 10% capacity and 1:5 false-positive/false-negative cost ratio are assumptions for sensitivity analysis, not documented airline policies or currency amounts.

## Dataset source and licence

- Source: [US DOT 2015 Flight Delays and Cancellations on Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays).
- Producer identified by the dataset page: US Department of Transportation, Bureau of Transportation Statistics.
- Kaggle licence: **CC0 Public Domain**, checked on 18 November 2026.
- Verified original size: **5,819,079 rows and 31 columns** in `flights.csv`, plus airline and airport lookups. The full-file independent count agrees with the preparation audit and original-file SHA256. Preparation records actual shape and hashes and rejects fewer than 300,001 raw or eligible rows. Training also rejects fewer than 300,001 training rows.
- Keep the original files unmodified. The source archive is too large for a normal GitHub commit. Obtain it using the script or browser download; only a 1,000-row real-data sample is generated for the repository after successful preparation.
- The dataset is historical and US-specific. It does not establish current airline reliability or applicability to the UAE. Confirm it has not been used in your previous coursework.
- See `docs/dataset_search.csv`, `docs/data_dictionary.csv`, `docs/references.md` and `docs/workflow.md`.

## Team and approvals

Student-authored entries required: team of two or three; names; IDs; GitHub usernames; actual roles; instructor approval evidence for all five milestones. Do not claim generated work as independent human contribution. Each member must make their own regular meaningful commits; no fabricated history is supplied.

## Environment

Use **Python 3.11 or 3.12** and a virtual environment. The real run used Python 3.12.14 and the pinned analysis versions in `requirements.txt`. Streamlit is a runtime dependency that was unavailable in the authoring environment, so live UI execution remains unverified. After installing, record its exact version in your run log. Plan for approximately 16 GB RAM and at least 10 GB free disk space for the full uncapped run; actual requirements depend on hardware. Final model training is uncapped by default. Tuning uses a reproducible 120,000-row sample from training only.

```bash
python -m venv .venv
```

Activate `.venv` using the command for your operating system, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/verify_trained_artifact.py
python scripts/build_visuals.py
python scripts/build_evidence.py
python -m streamlit run app/streamlit_app.py
```

If Kaggle requires a browser login, download the original ZIP from the source page and run:

```bash
python scripts/download_data.py --archive /path/to/flight-delays.zip
```

If the three original CSVs are already in `data/raw/`, run:

```bash
python scripts/run_pipeline.py --skip-download
```

For Jupyter notebooks, also install `requirements-dev.txt`. Run the numbered notebooks from first to last. The notebooks invoke the same modules as the command-line pipeline, avoiding duplicate implementations. All six notebooks include genuine outputs captured from execution against the original data and its verified run artifacts. Training intentionally refuses to overwrite an existing final test evaluation; see `docs/run_and_deploy.md` before an intentional rerun.

## Structure

| Path | Purpose |
|---|---|
| `config.json` | Fixed seed, date partitions, tuning budget and evaluation assumptions |
| `scripts/download_data.py` | Original data download, safe extraction and source checksums |
| `src/cleaning.py`, `scripts/prepare_data.py` | Full-data quality checks, global deduplication, quarantine and split export |
| `src/features.py` | Explicit pre-departure allowlist and deterministic feature engineering |
| `src/modeling.py`, `scripts/train_models.py` | Baseline, two model families, temporal tuning, ablation and final evaluation |
| `scripts/build_visuals.py` | Real-data EDA and evaluation plots with captions |
| `app/streamlit_app.py` | Historical decision-support application |
| `tests/` | Unit and integration tests with isolated artificial test fixtures |
| `notebooks/` | Numbered notebook entry points |
| `docs/` | Technical decision records, dictionaries, checklist and student-writing prompts |
| `data/raw/` | Original files, excluded from Git |
| `data/processed/` | Cleaned splits, quarantined rows and disk-backed duplicate index, excluded from Git |
| `data/sample/` | Real sample generated only after original data is supplied |
| `models/` | Real fitted pipeline and metadata generated after training |
| `reports/` | Machine-generated audits, metrics, search results and figures |

## Evaluation design

Training uses dates before 30 August 2015; validation uses 1 September through 29 October; testing uses 1 November through 31 December. The two days before each new partition are embargoed and preserved separately. Within training, expanding folds validate on April, June and August with the same two-day gap. The gap reduces immediate cross-boundary dependence; it does not guarantee elimination of all long-range temporal dependence.

The competitors are a prevalence baseline, logistic regression fitted by stochastic gradient descent, and histogram gradient-boosted trees. Imputation, scaling and categorical encoding are fitted inside each fold. Both nonbaseline models receive randomized hyperparameter search on the same folds. Winner selection uses mean CV average precision. Validation scores set the alert threshold. Test data is accessed by the training script only after selection, feature evaluation and threshold freezing.

Generated evidence includes CV before/after tuning, exact parameter search results, feature ablation, final AP and ROC AUC, precision/recall/F1, Brier score and log loss, confusion counts, precision/recall/lift at the review budget, day-cluster bootstrap intervals, subgroup performance, validation threshold sensitivity and scenario-cost sensitivity. Feature improvements, calibration or business benefit are not assumed in advance.

## Results and deployment status

The selected histogram gradient-boosting model uses the base feature set. The engineered set did not improve chronological CV average precision; the observed mean difference was only −0.000386, so this is a very small observed difference rather than strong statistical evidence of superiority.

| Held-out measure | Selected model | Prevalence baseline |
|---|---:|---:|
| Test flights | 947,202 | 947,202 |
| Average precision | 0.2663 | 0.1926 |
| ROC AUC | 0.6171 | 0.5000 |
| Brier score | 0.1526 | 0.1563 |
| Precision among highest-ranked 10% | 30.89% | 19.20% |
| Recall among highest-ranked 10% | 16.04% | 9.97% |
| Lift at 10% capacity | 1.604× | 0.997× |

The fixed validation threshold flagged 12.19% of test flights, exceeding the intended 10% validation workload. The app therefore also includes a batch-ranking function that enforces the selected fraction of a supplied schedule. That function was verified on 1,000 real flights, selecting exactly 100 at a 10% capacity.

Predictive discrimination is modest and top-10% review captures only 16.04% of disruptions. The model is not a reliable operational forecast. Calibration is imperfect, numerical airport IDs remain unresolved in October, and cost calculations use hypothetical relative units. Review `reports/results_evidence.html`, all generated tables, `models/model_card.md` and the limitations before writing your own interpretation.

**Repository URL: not created. Live Streamlit URL: not deployed.** Deployment must use this same repository and a real trained model. The app rejects fixture-trained artifacts. Follow `docs/run_and_deploy.md`, verify the public URL while signed out, and place both links on the first page of your group PDF.

## AI assistance disclosure

ChatGPT/Codex assisted with dataset-source research, coding, software tests and technical usage documentation for this package. Real-data analysis, model fitting, chart generation and inference checks ran in the authoring environment. Streamlit itself could not be installed, so the actual UI and public deployment remain unverified. The assessment brief restricts AI to coding support and requires disclosure in the README and report. Discuss the source-research assistance with your instructor if needed; do not conceal it. Students must verify, understand and adapt the code and write all submitted group-report and individual-assessment prose themselves. No submission-ready group report, personal reflection or fabricated contribution statement is provided.

## Assessment scope

This implements the Python/GitHub/Streamlit requirements of the uploaded **CDS 4003** brief. The separate 12-mark **RapidMiner** rubric is tracked in `docs/assessment_checklist.md`; no RapidMiner `.rmp` process or genuine RapidMiner execution screenshot is claimed. Confirm with the instructor whether that separate rubric applies. A Python implementation cannot by itself establish compliance with a RapidMiner-specific workflow criterion.

## Included real-run evidence

The archive includes the trained model, all source code, six executed notebooks, ten labelled figures, source and run manifests, all search results, a real source sample and an app input example. Raw and processed full datasets are excluded because they are reproducible from the original archive. Keep the original uploaded archive available and follow the source instructions. The HTML evidence page is machine-generated technical output, not the student-authored report required for Blackboard.

## Browser upload edition

This edition keeps all project source code, the trained model, executed notebooks, charts, assessment documentation and summary evidence. The 50 MiB row-level prediction export and routine console logs are omitted for browser upload. Full prediction records remain in the complete project archive and can be regenerated by the pipeline. `verify_trained_artifact.py` requires that export; restore it from the complete archive before running that verification script. The app does not need it. This edition contains fewer than 100 files, each below GitHub’s 25 MiB browser limit.
