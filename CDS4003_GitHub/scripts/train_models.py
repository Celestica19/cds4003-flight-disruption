"""Compare and tune models, select without test access, then evaluate once."""
import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, config, load_split, save_json, sha256_file
from src.features import make_features
from src.schema import MODEL_RAW
from src.modeling import model_pipeline, parameter_space, temporal_folds
from src.evaluation import budget_threshold, metrics, day_bootstrap

def train_project(root, cfg, fixture_mode=False, overwrite=False):
    root = Path(root)
    reports, models = root / "reports", root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    if (reports / "test_metrics.json").exists() and not overwrite:
        raise ValueError("Test results already exist. Do not repeatedly tune against test data. Archive the prior run before an intentional rerun using --overwrite.")
    audit = json.loads((reports / "quality_audit.json").read_text())
    if not fixture_mode and audit["status"] != "REAL_DATA_PREPARED":
        raise ValueError("Full real dataset has not passed preparation checks")
    seed = cfg["seed"]
    train_path = root / "data/processed/train.csv.gz"
    model_columns = MODEL_RAW + ["record_id", "disrupted"]
    train = load_split(train_path, cfg["max_train_rows"], seed, usecols=model_columns)
    val = load_split(root / "data/processed/validation.csv.gz", cfg["max_validation_rows"], seed, usecols=model_columns)
    if not fixture_mode and len(train) <= cfg["minimum_raw_rows_exclusive"]:
        raise ValueError("Training requires more than 300,000 eligible rows; remove or increase the training cap")
    if pd.to_datetime(train.flight_date).max() >= pd.to_datetime(val.flight_date).min() - pd.Timedelta(days=cfg["embargo_days"]):
        raise ValueError("Training/validation embargo violated")
    tune = train.sample(min(cfg["tuning_rows"], len(train)), random_state=seed).sort_values(["flight_date", "record_id"]).reset_index(drop=True)
    folds = temporal_folds(tune.flight_date, cfg["embargo_days"])
    y_tune = tune.disrupted.astype(int)
    for a, b in folds:
        if y_tune.iloc[a].nunique() != 2 or y_tune.iloc[b].nunique() != 2:
            raise ValueError("Each temporal fold must contain both target classes")
    fold_manifest = [{"fold": i + 1, "train_rows": len(a), "validation_rows": len(b),
                      "train_start": tune.flight_date.iloc[a].min(), "train_end": tune.flight_date.iloc[a].max(),
                      "validation_start": tune.flight_date.iloc[b].min(), "validation_end": tune.flight_date.iloc[b].max()}
                     for i, (a, b) in enumerate(folds)]
    save_json(fold_manifest, reports / "cv_folds.json")
    X_tune, X_train, X_val = make_features(tune), make_features(train), make_features(val)
    candidates = {}
    comparison, warning_log = [], []
    search_spaces = {}
    start = time.monotonic()
    for name in ["baseline", "logistic_sgd", "hist_gradient_boosting"]:
        print(f"Comparing {name} on {len(tune):,} training-only tuning rows", flush=True)
        estimator = model_pipeline(name, seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            before = cross_val_score(estimator, X_tune, y_tune, cv=folds, scoring="average_precision", n_jobs=cfg["parallel_jobs"], error_score="raise")
            if name == "baseline":
                tuned, after, params = estimator, before, {}
            else:
                space = parameter_space(name)
                search_spaces[name] = space
                search = RandomizedSearchCV(estimator, space, n_iter=cfg["search_iterations"],
                    scoring="average_precision", cv=folds, n_jobs=cfg["parallel_jobs"],
                    random_state=seed, refit=True, error_score="raise", return_train_score=True)
                search.fit(X_tune, y_tune)
                pd.DataFrame(search.cv_results_).to_csv(reports / f"search_{name}.csv", index=False)
                tuned, params = clone(search.best_estimator_), search.best_params_
                after = np.array([search.cv_results_[f"split{i}_test_score"][search.best_index_] for i in range(len(folds))])
            tuned.fit(X_train, train.disrupted.astype(int))
            warning_log.extend([{"model": name, "category": w.category.__name__, "message": str(w.message)} for w in captured])
        candidates[name] = tuned
        p_val = tuned.predict_proba(X_val)[:, 1]
        threshold = budget_threshold(p_val, cfg["alert_budget"])
        row = {"model": name, "cv_ap_before": float(before.mean()), "cv_ap_after": float(after.mean()),
               "cv_ap_std_after": float(after.std()), "parameters": json.dumps(params),
               **{f"validation_{k}": v for k, v in metrics(val.disrupted, p_val, threshold, cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed).items()}}
        comparison.append(row)
    save_json(search_spaces, reports / "search_spaces.json")
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(reports / "model_comparison.csv", index=False)
    # Winner chosen using only CV within January-August, never the test set.
    winner = comparison_frame.sort_values("cv_ap_after", ascending=False, kind="stable").iloc[0]["model"]
    selected = candidates[winner]
    p_val = selected.predict_proba(X_val)[:, 1]
    threshold = budget_threshold(p_val, cfg["alert_budget"])
    # Feature-value check: same tuned classifier settings and same chronological folds.
    base_model = model_pipeline(winner, seed, engineered=False)
    selected_params = {k: v for k, v in selected.get_params().items() if k.startswith("classifier__") and k != "classifier__categorical_features"}
    base_model.set_params(**selected_params)
    base_scores = cross_val_score(base_model, make_features(tune, False), y_tune, cv=folds,
        scoring="average_precision", n_jobs=cfg["parallel_jobs"], error_score="raise")
    full_scores = cross_val_score(clone(selected), X_tune, y_tune, cv=folds,
        scoring="average_precision", n_jobs=cfg["parallel_jobs"], error_score="raise")
    engineered_selected = bool(full_scores.mean() >= base_scores.mean())
    save_json({"model": winner, "base_fold_ap": base_scores.tolist(), "engineered_fold_ap": full_scores.tolist(),
               "mean_ap_difference": float(full_scores.mean() - base_scores.mean()),
               "selected_variant": "engineered" if engineered_selected else "base",
               "interpretation": "The higher-CV-AP feature variant is selected; ties retain the engineered variant. This fixed-parameter ablation is not a separately tuned model competition."}, reports / "feature_ablation.json")
    if not engineered_selected:
        selected = base_model.fit(make_features(train, False), train.disrupted.astype(int))
        X_val = make_features(val, False)
        p_val = selected.predict_proba(X_val)[:, 1]
        threshold = budget_threshold(p_val, cfg["alert_budget"])
    save_json(metrics(val.disrupted, p_val, threshold, cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed), reports / "final_validation_metrics.json")
    validation_groups = []
    for field, labels in [("month", pd.to_datetime(val.flight_date).dt.month.astype(str)),
                          ("airport_id_format", val.ORIGIN_AIRPORT.fillna("").str.startswith("BTS_").map({True: "BTS_ID", False: "IATA_OR_MISSING"}))]:
        for value in labels.unique():
            mask = labels.eq(value).to_numpy()
            validation_groups.append({"group_field": field, "group": str(value), **metrics(val.disrupted.to_numpy()[mask], p_val[mask], threshold, cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed)})
    pd.DataFrame(validation_groups).to_csv(reports / "validation_subgroup_metrics.csv", index=False)
    # Training-derived empirical supports for validation in the app.
    vocabulary = {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]}
    numeric_ranges = {c: {"min": float(train[c].min()), "max": float(train[c].max())}
                      for c in ["DISTANCE", "SCHEDULED_TIME"] if train[c].notna().any()}
    metadata = {"status": "TEST_FIXTURE_ONLY" if fixture_mode else "TRAINED_REAL_DATA",
        "winner": winner, "selection_rule": "Highest mean chronological CV average precision including baseline, then same-fold fixed-parameter feature ablation",
        "engineered_features": engineered_selected,
        "selected_feature_variant_cv_ap": float(full_scores.mean() if engineered_selected else base_scores.mean()),
        "threshold": threshold, "alert_budget": cfg["alert_budget"],
        "training_rows": len(train), "tuning_rows": len(tune), "validation_rows": len(val),
        "train_date_min": train.flight_date.min(), "train_date_max": train.flight_date.max(),
        "historical_year": cfg["expected_year"], "vocabulary": vocabulary, "numeric_ranges": numeric_ranges,
        "target": "Cancelled OR diverted OR arrival delay at least 15 minutes",
        "source_sha256": audit["source_sha256"], "configuration": cfg,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": ["Historical US 2015 demonstration, not validated for current operations or UAE routes.",
            "Probability calibration is measured, not guaranteed.", "Costs are illustrative units, not verified monetary savings.",
            "The fixed threshold may exceed the validation alert budget after distribution shift.",
            "If the baseline wins or lift is weak, do not claim business improvement."]}
    joblib.dump(selected, models / "model.joblib", compress=3)
    metadata["model_sha256"] = sha256_file(models / "model.joblib")
    save_json(metadata, models / "metadata.json")
    # Global, non-causal explanations measured on validation data only.
    explanation = X_val.sample(min(3000, len(X_val)), random_state=seed)
    importance = permutation_importance(selected, explanation, val.disrupted.loc[explanation.index],
        scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=cfg["parallel_jobs"])
    pd.DataFrame({"feature": explanation.columns, "mean_ap_drop": importance.importances_mean,
                  "std_ap_drop": importance.importances_std}).sort_values("mean_ap_drop", ascending=False).to_csv(reports / "permutation_importance.csv", index=False)
    # Threshold sensitivity is validation-only; no retuning after test outcomes.
    sensitivity = [metrics(val.disrupted, p_val, float(t), cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed)
                   for t in sorted(set(np.linspace(0.05, 0.95, 19).tolist() + [threshold]))]
    pd.DataFrame(sensitivity).to_csv(reports / "validation_threshold_sensitivity.csv", index=False)
    del X_train, X_tune, tune
    # First test access occurs AFTER model choice, feature decisions and threshold are frozen.
    test = load_split(root / "data/processed/test.csv.gz", cfg["max_test_rows"], seed, usecols=model_columns)
    if pd.to_datetime(val.flight_date).max() >= pd.to_datetime(test.flight_date).min() - pd.Timedelta(days=cfg["embargo_days"]):
        raise ValueError("Validation/test embargo violated")
    X_test = make_features(test, engineered_selected)
    p_test = selected.predict_proba(X_test)[:, 1]
    result = metrics(test.disrupted, p_test, threshold, cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed)
    result["day_cluster_bootstrap_95_percent_intervals"] = day_bootstrap(test.disrupted, p_test, test.flight_date, threshold,
        cfg["bootstrap_repetitions"], seed, cfg["false_positive_cost"], cfg["false_negative_cost"])
    baseline_p = np.full(len(test), float(train.disrupted.mean()))
    baseline_threshold = budget_threshold(np.full(len(val), float(train.disrupted.mean())), cfg["alert_budget"])
    baseline_results = metrics(test.disrupted, baseline_p, baseline_threshold, cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed)
    save_json({"status": metadata["status"], "selected_model": winner, "selected": result,
               "baseline": baseline_results, "test_date_min": test.flight_date.min(), "test_date_max": test.flight_date.max(),
               "notes": ["Test set used once for final assessment, not model selection.",
                         "Bootstrap resamples dates, conditional on the fitted model; it does not include training uncertainty."]}, reports / "test_metrics.json")
    prediction = test[["record_id", "flight_date", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "disrupted"]].copy()
    prediction["probability"], prediction["alert"] = p_test, p_test >= threshold
    prediction.to_csv(reports / "test_predictions.csv.gz", index=False, compression="gzip")
    groups = []
    for field, labels in [("airline", test.AIRLINE.fillna("UNKNOWN")),
                          ("month", pd.to_datetime(test.flight_date).dt.month.astype(str)),
                          ("airport_id_format", test.ORIGIN_AIRPORT.fillna("").str.startswith("BTS_").map({True: "BTS_ID", False: "IATA_OR_MISSING"}))]:
        for value in labels.unique():
            mask = labels.eq(value).to_numpy()
            groups.append({"group_field": field, "group": str(value), **metrics(test.disrupted.to_numpy()[mask], p_test[mask], threshold, cfg["alert_budget"], cfg["false_positive_cost"], cfg["false_negative_cost"], seed)})
    pd.DataFrame(groups).to_csv(reports / "test_subgroup_metrics.csv", index=False)
    # Scenario sensitivity changes cost assumptions, never the frozen predictions.
    cost_rows = [metrics(test.disrupted, p_test, threshold, cfg["alert_budget"], 1, ratio, seed) | {"fn_to_fp_cost_ratio": ratio}
                 for ratio in [1, 2, 5, 10, 20]]
    pd.DataFrame(cost_rows).to_csv(reports / "test_cost_sensitivity.csv", index=False)
    save_json(warning_log, reports / "training_warnings.json")
    save_json({"status": metadata["status"], "elapsed_seconds": round(time.monotonic() - start, 2),
               "raw_rows": audit["counts"]["raw_rows"], "full_train_rows": audit["split_counts"]["train"],
               "used_train_rows": len(train), "used_validation_rows": len(val), "used_test_rows": len(test),
               "tuning_rows": cfg["tuning_rows"], "note": "Tuning is sampled for tractability. Final fitting and holdout scoring are uncapped by default."}, reports / "run_manifest.json")
    if fixture_mode:
        print("Fixture integration completed. Its metrics are not real-data results.", flush=True)
    else:
        print(f"Finished. Selected model: {winner}; test AP: {result['average_precision']:.4f}", flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_project(ROOT, config(args.config), overwrite=args.overwrite)

if __name__ == "__main__":
    main()
