"""Probability, classification, workload and scenario-cost evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                            recall_score, f1_score, brier_score_loss, log_loss, confusion_matrix)

def budget_threshold(probabilities, budget=0.10):
    """Choose an unlabeled validation score cutoff with alert rate <= budget.

    Ties are excluded if including them would exceed capacity. In the constant-
    score case the result alerts nobody; precision@k still tests ranking ties.
    """
    p = np.asarray(probabilities, dtype=float)
    if not len(p) or not 0 < budget < 1 or not np.isfinite(p).all():
        raise ValueError("Invalid probabilities or alert budget")
    k = int(np.floor(len(p) * budget))
    if k < 1:
        return float(np.nextafter(p.max(), np.inf))
    cutoff = np.sort(p)[-k]
    if np.count_nonzero(p >= cutoff) > k:
        cutoff = np.nextafter(cutoff, np.inf)
    return float(cutoff)

def metrics(y, p, threshold, budget=0.10, fp_cost=1, fn_cost=5, seed=42):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # Random tie-break is deterministic and independent of labels/date ordering.
    tie = np.random.default_rng(seed).random(len(p))
    ranking = np.lexsort((tie, -p))
    k = max(1, int(np.floor(len(y) * budget)))
    prevalence = float(y.mean())
    precision_k = float(y[ranking[:k]].mean())
    return {
        "n": len(y), "prevalence": prevalence,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "threshold": float(threshold), "alert_rate": float(pred.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "rank_budget": budget, "precision_at_budget": precision_k,
        "recall_at_budget": float(y[ranking[:k]].sum() / y.sum()) if y.sum() else 0.0,
        "lift_at_budget": precision_k / prevalence if prevalence else None,
        "scenario_cost_per_1000": float(1000 * (fp_cost * fp + fn_cost * fn) / len(y))
    }

def day_bootstrap(y, p, dates, threshold, repetitions=200, seed=42, fp_cost=1, fn_cost=5):
    y, pred = np.asarray(y, dtype=int), np.asarray(p) >= threshold
    daily = pd.DataFrame({"date": np.asarray(dates), "tp": (y == 1) & pred,
                          "fp": (y == 0) & pred, "fn": (y == 1) & ~pred,
                          "n": np.ones(len(y), dtype=int)}).groupby("date").sum()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sums = daily.iloc[rng.integers(0, len(daily), len(daily))].sum()
        estimates.append([sums.tp / max(1, sums.tp + sums.fp), sums.tp / max(1, sums.tp + sums.fn),
                          1000 * (fp_cost * sums.fp + fn_cost * sums.fn) / sums.n])
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {name: {"lower": float(intervals[0, j]), "upper": float(intervals[1, j])}
            for j, name in enumerate(["precision", "recall", "scenario_cost_per_1000"])}

def validate_input(flight_date, airline, origin, destination, departure_hhmm, arrival_hhmm, distance, duration):
    """App validation in plain Python so it is testable without Streamlit."""
    from .cleaning import valid_hhmm
    date = pd.to_datetime(flight_date, errors="coerce")
    if pd.isna(date):
        raise ValueError("Choose a valid service date.")
    if not airline or not origin or not destination:
        raise ValueError("Choose an airline and both airports.")
    if origin == destination:
        raise ValueError("Origin and destination must be different.")
    if not valid_hhmm(pd.Series([departure_hhmm, arrival_hhmm])).all():
        raise ValueError("Use valid local clock times, such as 0830 or 1745.")
    if not np.isfinite([distance, duration]).all() or distance <= 0 or duration <= 0:
        raise ValueError("Distance and scheduled duration must be positive finite numbers.")
    return pd.DataFrame([{"flight_date": date.strftime("%Y-%m-%d"), "AIRLINE": airline,
        "ORIGIN_AIRPORT": origin, "DESTINATION_AIRPORT": destination,
        "SCHEDULED_DEPARTURE": departure_hhmm, "SCHEDULED_ARRIVAL": arrival_hhmm,
        "DISTANCE": distance, "SCHEDULED_TIME": duration}])
