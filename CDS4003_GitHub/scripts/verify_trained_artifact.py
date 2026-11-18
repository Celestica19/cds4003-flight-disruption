"""Check persisted real-data model predictions and actual predictor imputation."""
import json
import sys
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, save_json, sha256_file
from src.schema import MODEL_RAW
from src.features import make_features, feature_columns
from src.evaluation import validate_input

def verify(root=ROOT):
    root = Path(root)
    meta = json.loads((root / "models/metadata.json").read_text())
    assert meta["status"] == "TRAINED_REAL_DATA"
    model_path = root / "models/model.joblib"
    assert sha256_file(model_path) == meta["model_sha256"]
    model = joblib.load(model_path)
    numeric, categorical = feature_columns(meta["engineered_features"])
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    before, after = Counter(), Counter()
    training_rows = 0
    for d in pd.read_csv(root / "data/processed/train.csv.gz", usecols=MODEL_RAW, chunksize=100000):
        X = make_features(d, meta["engineered_features"])
        before.update(X[numeric].isna().sum().to_dict())
        Z = imputer.transform(X[numeric])
        after.update(dict(zip(numeric, np.isnan(Z).sum(axis=0))))
        training_rows += len(d)
    pd.DataFrame([{"feature": c, "training_rows": training_rows, "missing_before": before[c],
                   "missing_after_fitted_imputer": after[c], "training_median": float(imputer.statistics_[i])}
                  for i, c in enumerate(numeric)]).to_csv(root / "reports/model_input_imputation.csv", index=False)
    assert not sum(after.values())
    heldout = pd.read_csv(root / "data/processed/test.csv.gz", nrows=1000)
    actual = model.predict_proba(make_features(heldout, meta["engineered_features"]))[:, 1]
    expected = pd.read_csv(root / "reports/test_predictions.csv.gz", usecols=["record_id", "probability"])
    expected = expected.set_index("record_id").loc[heldout.record_id, "probability"].to_numpy()
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
    row = heldout.iloc[0]
    inputs = validate_input(row.flight_date, row.AIRLINE, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT,
                            row.SCHEDULED_DEPARTURE, row.SCHEDULED_ARRIVAL, row.DISTANCE, row.SCHEDULED_TIME)
    app_probability = float(model.predict_proba(make_features(inputs, meta["engineered_features"]))[0, 1])
    assert abs(app_probability - actual[0]) < 1e-12
    unknown = inputs.copy(); unknown["ORIGIN_AIRPORT"] = "BTS_99999"
    unknown_probability = float(model.predict_proba(make_features(unknown, meta["engineered_features"]))[0, 1])
    assert 0 <= unknown_probability <= 1
    results = {"status": "passed", "training_rows_imputation_checked": training_rows,
               "missing_numeric_inputs_after_imputation": int(sum(after.values())),
               "persisted_prediction_rows_compared": 1000, "predictions_match_saved_evaluation": True,
               "app_input_to_feature_to_prediction_path": "passed", "unseen_airport_prediction": "passed",
               "streamlit_ui_runtime": "not tested; dependency unavailable",
               "public_deployment": "not deployed", "model_sha256_matched": True}
    save_json(results, root / "reports/trained_artifact_validation.json")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    verify()
