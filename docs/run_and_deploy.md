# Running, reproducing and deploying the project

Choose the route that matches your goal. Reading evidence and serving the existing model do not require a full retraining run.

| Route | Full dataset needed? | Main action |
|---|---|---|
| Inspect the completed work | No | Read notebooks, metrics, figures and the technical walkthrough |
| Run the included application | No | Install dependencies and launch Streamlit |
| Reproduce the full analysis | Yes | Preserve previous outputs, then run acquisition and the pipeline |
| Publish the application | No | Deploy the included model from the verified GitHub root |

## 1. Prepare the Python environment

Use Python 3.12, matching the completed analysis environment. Open a terminal in the project directory containing `requirements.txt`.

```bash
python --version
python -m venv .venv
```

Activate the environment in Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

For Windows PowerShell, the corresponding activation path is `.venv\Scripts\Activate.ps1`. If local policy prevents activation, use Command Prompt or invoke the environment's Python directly. Activation is convenient but not required when using the interpreter's full path.

For macOS/Linux:

```bash
source .venv/bin/activate
```

Install the recorded dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The saved run passed 18 software tests. They cover software behaviour using isolated fixtures; real-source audits and saved-model verification are separate evidence. The Streamlit dependency was unavailable in the analysis environment, so its runtime still needs verification.

Core analysis packages are pinned to maintain compatibility with the serialized estimator. Streamlit uses a bounded version range. Record the actual installed deployment version rather than assuming it is identical across machines.

## 2. Launch the included model

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local address printed by Streamlit. The application uses:

| Required artifact | Purpose |
|---|---|
| `app/streamlit_app.py` | Interface and input processing |
| `src/` | Shared feature and review logic |
| `models/model.joblib` | Fitted preprocessing and classifier |
| `models/metadata.json` | Feature variant, threshold, checksum, vocabulary and numeric ranges |
| `models/lookups.json` | Airline and airport display names |
| `reports/test_metrics.json` | Recorded holdout performance |
| `reports/permutation_importance.csv` | Global validation feature importance |

The app requires real-data model status and verifies the model checksum. It does not train a new model on startup. No Kaggle account or original raw dataset is needed to serve the included artifact.

### Manual acceptance checks

These are **checks to perform**, not a record that the UI has already passed.

| Check | Action | Expected behaviour |
|---|---|---|
| Initial load | Start the app with supplied artifacts | Title, input form and performance sections load without error |
| Valid flight | Choose two different airports, valid 2015 date and in-range schedule inputs | A finite probability between 0 and 1 and a priority message appear |
| Same-airport rejection | Set origin equal to destination | Clear validation feedback; no accepted prediction |
| Numeric-range check | Try a value beyond the training range | Explicit rejection rather than silent extrapolation |
| Batch example | Upload `data/sample/app_example.csv` at 10% capacity | Two of the 20 valid rows are selected |
| Batch columns | Remove a required column in a temporary copy | Message identifies missing required input |
| Historical date | Change a temporary batch date outside 2015 | Batch is rejected with a year/date message |
| CSV export | Download the ranked list | File includes input row, probability, rank and review flag |
| Performance display | Compare displayed values with test metrics | AP about 0.266; queue precision about 30.9%; recall about 16.0% |
| Cold reload | Restart the app and reload | Model loads with the same metadata and results |

Input examples are historical demonstrations. A valid form submission does not establish that an arbitrary manually entered route/duration corresponds to a real scheduled flight. Use the real example schedule when checking realistic inputs.

## 3. Reproduce the full analysis separately

Keep the supplied completed version intact. Make a separate working copy for reproduction. Archive the copy's existing `reports/` and `models/` folders outside that working project, then allow the scripts to create new outputs. Keep the source code, configuration and original input files. This prevents old cached outputs from being mistaken for a new run.

The training script refuses to overwrite `reports/test_metrics.json`. Do not remove that protection by default. If correcting a verified implementation defect, retain the prior run and document the reason before intentionally invoking `train_models.py --overwrite`.

Plan approximately 16 GB RAM and 10 GB free disk space for a full run. Resource needs vary by hardware. Tuning uses 120,000 training-only records; the final fit and holdouts are uncapped by default.

### Acquire the source

```bash
python scripts/download_data.py
```

Or import a browser-downloaded archive:

```bash
python scripts/download_data.py --archive /path/to/flight-delays.zip
```

### Run all stages

With `flights.csv`, `airlines.csv` and `airports.csv` in `data/raw/`:

```bash
python scripts/run_pipeline.py --skip-download
```

The orchestrator stops if any stage fails. Its order is:

| Order | Script | Main output or check |
|---|---|---|
| 1 | `prepare_data.py` | Source audit, date partitions, lookups and training EDA tables |
| 2 | `train_models.py` | Candidate comparison, tuned model, threshold, evaluation and prediction export |
| 3 | `verify_trained_artifact.py` | Saved-model and numerical-input consistency |
| 4 | `build_visuals.py` | Ten analysis figures and captions |
| 5 | `build_evidence.py` | HTML results evidence |

The supplemental workflow, folder-layout and timeline visuals are documentation artifacts built from the code structure and recorded partition manifest. They are not additional model evaluations.

Inspect `quality_audit.json` for row reconciliation, `training_warnings.json` for warnings, `run_manifest.json` for actual fit sizes, and `trained_artifact_validation.json` for consistency checks. Preserve source hashes, configuration, versions and run logs.

The standalone verification script requires `reports/test_predictions.csv.gz`. This approximately 50 MiB export is omitted from the browser-upload edition and included in the complete archive; a full pipeline run regenerates it.

### Use the notebooks

Install `requirements-dev.txt` and open the six numbered notebooks in order. Markdown explains purpose, execution sequence, findings and limitations. The notebooks invoke shared modules rather than maintaining separate implementations.

Their captured outputs belong to the supplied run. Rerunning raw-data or training-preview cells requires the corresponding full files. A notebook rendering successfully on GitHub is not proof that its cells have just been rerun on your machine.

## 4. Prepare the GitHub repository

Follow [github_update_guide.md](github_update_guide.md). Ensure `README.md`, `requirements.txt`, `app/`, `src/`, `models/` and `reports/` are at the root.

The existing repository is [Celestica19/cds4003-flight-disruption](https://github.com/Celestica19/cds4003-flight-disruption). Upload the latest project contents and verify them before removing the obsolete nested directory. Preserve meaningful commit history and credit each person's actual changes.

Full source data, processed partitions and the large prediction export should remain outside ordinary browser uploads. The provided browser edition contains fewer than 100 files with each below 25 MiB.

## 5. Deploy through Streamlit Community Cloud

Open [Streamlit Community Cloud](https://share.streamlit.io/) with an account that can access the repository. Create an app from the existing repository and enter:

| Setting | Value |
|---|---|
| Repository | `Celestica19/cds4003-flight-disruption` |
| Branch | `main` |
| Main file | `app/streamlit_app.py` |
| Python in advanced settings | 3.12 |

Launch deployment and inspect the build/runtime logs. Dependency installation and model loading must both succeed. The app serves an existing artifact, so a cloud service does not need the memory used for full training.

Run the manual acceptance checks above on the deployed URL. Then open that URL in a signed-out/private browser window. Record the deployed dependency versions, the commit used and the date of verification. Add the live URL to the README only after public access succeeds.

The official [deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) describes repository selection, entry point, Python settings and launch logs.

## 6. Diagnose common failures

| Symptom | What to inspect | Resolution |
|---|---|---|
| `No module named streamlit` | Active Python environment and installation output | Install requirements using the same interpreter that launches the app |
| `No module named src` | Repository layout and entry point | Place the full project at the root and use `app/streamlit_app.py` |
| Missing model message | `models/model.joblib` and metadata paths | Upload the supplied trusted artifact and matching metadata |
| Checksum mismatch | Whether model and metadata came from different versions | Restore both from the same verified package |
| Serialized-estimator compatibility error | Python and scikit-learn versions | Match the pinned analysis environment; inspect the exact traceback |
| Existing test-results error | `reports/test_metrics.json` | Preserve the prior run; use a separate reproduction copy |
| Artifact verifier cannot find predictions | Browser edition omits row-level export | Restore the export from the complete archive or reproduce the full pipeline |
| Memory exhaustion during training | Final fit size and available RAM | Use a suitable machine; document any sampling change rather than silently reducing the run |
| App available only to owner | Sharing settings and signed-out test | Configure public access and recheck without owner authentication |
| Cloud app shows an old version | Deployed branch and latest commit | Confirm `main` contains the revision; inspect deployment status |

A traceback identifies the failing operation. Keep it available for diagnosis instead of rerunning expensive preparation or training when the issue is only a missing dependency or nested folder.

## 7. Evidence and operational follow-up

Capture genuine app evidence after successful verification: the full interface, a valid prediction, validation feedback, a batch selection/download and the performance section. Exclude private account details and secrets. UI screenshots are evidence of execution; schematic diagrams in the documentation explain structure and do not substitute for them.

Before operational use, obtain recent data and agree on review costs and capacity. Monitor schema changes, missing values, unseen categories, score shifts, workload and delayed-label performance. Assign monitoring owners and define retraining criteria. These are future operational proposals.

**Current status:** the data pipeline, trained artifact and pure-Python inference checks are complete. The repository exists. Final root-layout verification, Streamlit UI execution and public deployment remain pending.

