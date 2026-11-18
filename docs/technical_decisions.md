# Implementation decisions and evidence

This record explains the implementation choices and their evidence. Read it alongside the [step-by-step technical walkthrough](project_walkthrough.md), which connects each choice to execution order and observed results.

| Issue or decision | Implementation | Evidence produced |
|---|---|---|
| Source preservation | Original CSV remains untouched; SHA256 and byte sizes recorded | `download_manifest.json`, `quality_audit.json` |
| Realistic scale | Full original file processed in chunks; require >300,000 raw, eligible and training rows | Audit and run manifests |
| Missing values | Normalize missing tokens; retain structurally absent outcomes; median imputation for numerical predictors inside training folds | Missingness table; serialized fitted pipeline |
| Categorical absence | Explicit `UNKNOWN` category; fold-fitted encoding | Pipeline and feature code |
| Missing labels | Never infer an on-time label from absent arrival delay; cancellation/diversion can define a known positive label; otherwise quarantine | `quarantine.csv.gz`, reason counters |
| Exact duplicates | SHA256 of every original field; SQLite index across chunks | Removed-row counter and reconciliation |
| Non-identical repeats | Retain and flag repeated business keys; no arbitrary first-record rule for conflicting facts | Audit counter; duplicate index for investigation |
| Mixed airport identifiers | Normalize whitespace/case, distinguish IATA codes from `BTS_` numeric IDs; optional verified crosswalk | Per-column mapping/unresolved counters |
| Invalid types and domains | Invalid nonnumeric values, nonfinite values, invalid HHMM, nonpositive distance/duration and negative cause durations become missing | Detailed change counters |
| Clock interpretation | Accept 2400 as midnight; clock encoding becomes 0; departure calendar features roll to the following day | Clock tests and counters |
| Dates | Invalid dates/unexpected years quarantined; weekday recomputed from service date | Quarantine counters and weekday corrections |
| Legitimate extremes | Retain negative delays and positive extreme delays; descriptive train-only IQR review does not automatically remove records | `outlier_review.csv`, retained-extreme counters |
| Outcome inconsistency | Cancelled and diverted simultaneously is quarantined; cancellation reason on a noncancelled flight is cleared | Audit counters |
| Structural missing cause fields | Preserve absent delay-cause values; no unsupported zero imputation | Before/after missingness |
| Data leakage | Allowlist schedule fields only; preprocessing fitted in each fold; no actual departure, arrival, delay cause or tail assignment predictors | Leakage invariance tests and feature code |
| Temporal validation | Fixed calendar holdouts, expanding folds and two-day embargo | `cv_folds.json`, split counts |
| Tuning efficiency | Uniform fixed-seed training sample; final fit and holdouts uncapped by default | Run manifest distinguishes raw, sample and final sizes |
| Class imbalance | AP and workload metrics primary; tune class weighting for the linear model; avoid synthetic oversampling | Search space, comparison and final metrics |
| Feature value | Same-fold, fixed-parameter comparison with/without engineered features | `feature_ablation.json` |
| Alerts | Validation-score threshold under a 10% assumed capacity; ties excluded when they exceed capacity | Threshold sensitivity; actual validation/test alert rates |
| Costs | Illustrative relative error costs and multiple ratios; never label these as measured money saved | Cost sensitivity CSV |
| Model explanation | Validation permutation importance of the selected feature set, with correlation and causality limitations | Importance CSV and figure |
| Test uncertainty | Date-cluster bootstrap of precision, recall and cost, conditional on fixed fitted model | Final evaluation JSON |

## Airport identifiers requiring review

The dataset mixes five-digit BTS airport IDs with IATA codes in October. The full original-file check verified 486,165 numeric-ID origins and 486,165 numeric-ID destinations, matching the October record count. Do not fabricate a lookup. If a verified, historically valid authoritative crosswalk is available, place `airport_id_to_iata.csv` in `data/raw/` with columns `airport_id,iata_code`. Record its source, retrieval date and historical applicability; the pipeline validates one mapping per ID and records its checksum. Without a mapping the namespace is preserved and unmatched categories are handled explicitly. This is a documented limitation, not a claim that all identities have been reconciled. Examine validation performance for September versus October before discussing robustness.

## Model mechanics and trade-offs

The baseline predicts the training disruption prevalence for every flight. It tests whether more complex models add predictive information. A tied-score baseline has no useful ranking; deterministic random tie-breaking is independent of outcomes.

The linear model fits logistic loss by stochastic gradient descent. A sigmoid converts a linear feature score into a class probability estimate. Standardization helps numerical optimization; one-hot encoding represents categories without inventing numerical order. It scales efficiently but depends on specified features to express nonlinear relationships. Regularization strength, penalty and class weighting are tuned. Class weighting may change probability calibration, so calibration metrics are reported rather than assumed.

Histogram gradient boosting adds decision trees to reduce prediction loss. Binned numerical values and native categorical splits model nonlinear effects and interactions. Ordinal encoding is used only to communicate categorical IDs to native categorical handling; those IDs are not treated as continuous rank. Unknown categories become missing at prediction. Category grouping is learned separately in each fold. Leaf count, learning rate, regularization and minimum leaf size are tuned. Early stopping is disabled to avoid a hidden random validation split.

Median imputation is a defensible simple baseline, not universally optimal. Students should compare alternatives if the real missingness pattern warrants it. Domain-invalid predictor values become missing rather than deleting otherwise usable outcomes. No claims about imputation improving accuracy can be made until results are observed.

## Feature definitions

Baseline inputs are planned departure and arrival minutes, distance, scheduled duration, departure calendar month and weekday, airline and origin/destination categories. Engineered inputs add sine/cosine representations of cyclic times and dates, weekend, late-evening departure, log distance, scheduled miles per hour, numeric-ID indicators and route. The speed proxy uses scheduled gate-to-gate duration, not measured airborne speed. No holiday or weather variable is invented. Local origin and destination clock times are not directly subtracted because of time-zone differences.

## Limitations and next investigations

The data covers 2015 US reporting carriers. Carrier mergers, airport networks, weather, disruption causes and operating practices have changed. The composite target combines events with different severity and costs. There are no measured intervention effects, staff-capacity records or passenger outcomes; predictive lift alone does not establish ROI. The two-day embargo does not eliminate all seasonal or route dependence. Infrequent-category grouping can weaken rare-airport signals. Permutation importance is global and noncausal. Small subgroups require caution. Report weak models or negative ablations honestly; do not repeatedly optimize against the held-out test period.

For a practical extension, obtain recent data, use multiple rolling backtests, assess calibrated probabilities on a separate temporal calibration period, verify error costs with a stakeholder, monitor feature drift and observed outcome rates, and define a retraining trigger before operational use. These are future work, not completed features.

## Decision boundaries and alternatives

| Choice made | Alternative considered | Why this implementation uses the stated choice | Remaining limitation |
|---|---|---|---|
| Preserve outcome-related absence | Fill every missing value with zero | Zero can create a false on-time or no-cause interpretation | Structural patterns need review when changing source |
| Training-fold median imputation | Drop incomplete rows or use a learned multivariate imputer | Only six planned-duration training values are missing; simple fold-fitted handling is reproducible | No comparative accuracy benefit from the imputer is claimed |
| Retain valid extremes | Remove every IQR-flagged observation | Rare severe disruption belongs to the business outcome | Extremes should still be checked against source provenance |
| Explicit BTS namespace | Guess an airport mapping or remove October | Avoid false identities and preserve records | Airport identity remains unresolved |
| Calendar validation | Random row split | Reflect later-period prediction and reduce mixing of time conditions | A single year cannot establish future-year robustness |
| AP plus queue metrics | Select on raw accuracy | A majority-class predictor can appear accurate without helping the queue | AP does not directly establish intervention value |
| Same-fold feature ablation | Assume every engineered feature helps | Measures whether the added set changes the selected family's CV score | Small difference; variants not separately retuned |
| Capacity-based batch queue | Rely only on the frozen threshold | Workload remains controlled when score distributions shift | Review membership depends on relative scores and ties |

The planned crosswalk, recent-data evaluation, calibration study and monitoring service are future work. They are not represented as completed components.
