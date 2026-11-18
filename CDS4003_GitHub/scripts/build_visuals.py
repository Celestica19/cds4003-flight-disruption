"""Generate labeled figures from observed data, never illustrative results."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, config, load_split, save_json
from src.features import make_features

plt.rcParams.update({"figure.figsize": (9, 5), "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 160})

def build_visuals(root, cfg):
    root = Path(root)
    out = root / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    captions = []
    def save(name, caption):
        plt.tight_layout()
        plt.savefig(out / name, bbox_inches="tight")
        plt.close()
        captions.append({"filename": name, "caption": caption})
    quality = pd.read_csv(root / "reports/missingness_before_after.csv")
    top = quality.sort_values("raw_missing_pct").tail(12)
    y = np.arange(len(top))
    plt.figure(figsize=(10, 6))
    plt.barh(y - .18, top.raw_missing_pct, .36, label="Raw rows")
    plt.barh(y + .18, top.clean_missing_pct, .36, label="Eligible unique rows")
    plt.yticks(y, top.column)
    plt.xlabel("Missing values (%)")
    plt.title("Figure 1  Missingness before and after deterministic cleaning")
    plt.legend()
    save("01_missingness.png", "All rows profiled. Raw and eligible-row denominators differ. Structural missing values are retained; predictor imputation occurs later inside training folds.")
    # Exact training aggregates; no EDA on holdout outcomes before modelling.
    summaries = {"AIRLINE": [], "flight_date": [], "departure_hour": []}
    for chunk in pd.read_csv(root / "data/processed/train.csv.gz", chunksize=100000):
        f = make_features(chunk)
        chunk["departure_hour"] = (f.dep_minute // 60).fillna(-1).astype(int)
        for field in summaries:
            summaries[field].append(chunk.groupby(field, dropna=False).disrupted.agg(["count", "sum"]))
    aggregates = {}
    for field, parts in summaries.items():
        table = pd.concat(parts).groupby(level=0).sum()
        table["disruption_rate"] = table["sum"] / table["count"]
        table.to_csv(root / "reports" / f"eda_{field}.csv")
        aggregates[field] = table
    carrier = aggregates["AIRLINE"].sort_values("disruption_rate")
    plt.figure()
    plt.barh(carrier.index.astype(str), carrier.disruption_rate * 100)
    plt.xlabel("Disrupted flights (%)")
    plt.title("Figure 2  Historical disruption rate by airline")
    save("02_airline_rates.png", "All training rows; cancellations, diversions and arrivals at least 15 minutes late form the outcome. Differences are descriptive and not causal airline rankings.")
    hour = aggregates["departure_hour"].loc[lambda d: d.index >= 0]
    plt.figure()
    plt.plot(hour.index, hour.disruption_rate * 100, marker="o")
    plt.xlabel("Scheduled local departure hour")
    plt.ylabel("Disrupted flights (%)")
    plt.xticks(range(0, 24, 2))
    plt.title("Figure 3  Disruption rates by scheduled departure hour")
    save("03_departure_hour.png", "All training rows with valid scheduled departure time. A time association may also reflect route mix, weather or accumulated delays.")
    daily = aggregates["flight_date"]
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(daily.index), daily.disruption_rate * 100, alpha=.45, label="Daily")
    plt.plot(pd.to_datetime(daily.index), daily.disruption_rate.rolling(7, min_periods=7).mean() * 100, label="Trailing 7-day mean")
    plt.xlabel("Service date")
    plt.ylabel("Disrupted flights (%)")
    plt.title("Figure 4  Training-period daily disruption rates")
    plt.legend()
    save("04_daily_pattern.png", "Exact training-period aggregates. The rolling line uses only the preceding seven dates and is descriptive, not a model input.")
    sample = load_split(root / "data/processed/train.csv.gz", cfg["eda_sample_rows"], cfg["seed"])
    plt.figure()
    plt.hist(sample.ARRIVAL_DELAY.dropna(), bins=60, range=(-60, 300), color="#245e8e")
    plt.xlabel("Arrival delay in minutes")
    plt.ylabel("Flight count in EDA sample")
    plt.title("Figure 5  Arrival delays including valid early arrivals")
    outside = int(((sample.ARRIVAL_DELAY < -60) | (sample.ARRIVAL_DELAY > 300)).sum())
    save("05_arrival_delay.png", f"Uniform hash sample of {len(sample):,} training rows. Display restricted to −60 through 300 minutes; {outside:,} nonmissing observations outside that range remain in the data. Missing arrival outcomes are omitted from this histogram.")
    # IQR flags describe extremes, without automatically deleting valid flights.
    outlier_rows = []
    for c in ["DISTANCE", "SCHEDULED_TIME", "ARRIVAL_DELAY", "DEPARTURE_DELAY"]:
        s = sample[c].dropna()
        if not len(s):
            continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_rows.append({"column": c, "sample_n": len(s), "q1": q1, "q3": q3,
                             "lower_fence": low, "upper_fence": high,
                             "flagged_n": int(((s < low) | (s > high)).sum()),
                             "decision": "Retain domain-valid extremes; review IQR flags, do not automatically remove"})
    pd.DataFrame(outlier_rows).to_csv(root / "reports/outlier_review.csv", index=False)
    sample[["DISTANCE", "SCHEDULED_TIME", "ARRIVAL_DELAY", "DEPARTURE_DELAY"]].describe().to_csv(root / "reports/training_sample_statistics.csv")
    prediction_file = root / "reports/test_predictions.csv.gz"
    if prediction_file.exists():
        p = pd.read_csv(prediction_file)
        PrecisionRecallDisplay.from_predictions(p.disrupted, p.probability, plot_chance_level=True)
        plt.title("Figure 6  Held-out precision recall curve")
        save("06_precision_recall.png", "November–December holdout, after model and threshold freezing. Average precision is reported separately; it is not trapezoidal PR area.")
        RocCurveDisplay.from_predictions(p.disrupted, p.probability, plot_chance_level=True)
        plt.title("Figure 7  Held-out receiver operating characteristic")
        save("07_roc.png", "Read alongside precision–recall and the observed class prevalence.")
        ConfusionMatrixDisplay.from_predictions(p.disrupted, p.alert.astype(int), display_labels=["Not disrupted", "Disrupted"], cmap="Blues", colorbar=False)
        plt.title("Figure 8  Frozen-threshold confusion matrix")
        save("08_confusion_matrix.png", "Threshold chosen from validation scores to respect the validation workload budget; test workload can differ.")
        fraction, average = calibration_curve(p.disrupted, p.probability, n_bins=10, strategy="quantile")
        plt.figure()
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        plt.plot(average, fraction, "o-", label="Selected model")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed disruption fraction")
        plt.title("Figure 9  Held-out probability calibration")
        plt.legend()
        save("09_calibration.png", "Quantile-bin calibration diagnostic; few unique scores may produce fewer than ten points. No claim of calibrated probabilities is assumed.")
        importance = pd.read_csv(root / "reports/permutation_importance.csv").head(12).sort_values("mean_ap_drop")
        plt.figure(figsize=(10, 6))
        plt.barh(importance.feature, importance.mean_ap_drop, xerr=importance.std_ap_drop)
        plt.xlabel("Validation average-precision decrease after permutation")
        plt.title("Figure 10  Global predictive feature importance")
        save("10_feature_importance.png", "Validation sample, three repeats. Correlated features may share importance. This is a global predictive association, not a causal or flight-specific explanation.")
    save_json(captions, root / "reports/figure_captions.json")
    print(f"Generated {len(captions)} observed-data figures", flush=True)
    return captions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    build_visuals(ROOT, config(args.config))
