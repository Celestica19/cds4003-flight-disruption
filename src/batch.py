"""Capacity-constrained batch review using a frozen model, without labels."""
import numpy as np
import pandas as pd
from .cleaning import valid_hhmm
from .features import make_features
from .schema import MODEL_RAW

def prioritize_batch(frame, model, metadata, review_fraction=0.10):
    if not 0 < review_fraction <= 1:
        raise ValueError("Review fraction must be greater than zero and no more than one.")
    if frame.empty or len(frame) > 50000:
        raise ValueError("Upload between 1 and 50,000 flights per batch.")
    absent = set(MODEL_RAW) - set(frame.columns)
    if absent:
        raise ValueError("Missing required columns: " + ", ".join(sorted(absent)))
    d = frame[MODEL_RAW].copy().reset_index(drop=True)
    dates = pd.to_datetime(d.flight_date, errors="coerce")
    if dates.isna().any() or dates.dt.year.ne(metadata["historical_year"]).any():
        raise ValueError("Use valid service dates from the model's historical year.")
    for c in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        d[c] = d[c].astype("string").str.strip().str.upper()
        pattern = r"[A-Z0-9]{2}" if c == "AIRLINE" else r"[A-Z]{3}|BTS_\d{5}"
        if not d[c].str.fullmatch(pattern).fillna(False).all():
            raise ValueError(f"Invalid or missing codes in {c}.")
    if d.ORIGIN_AIRPORT.eq(d.DESTINATION_AIRPORT).any():
        raise ValueError("Origin and destination must differ for each flight.")
    for c in ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"]:
        if not valid_hhmm(d[c]).all():
            raise ValueError(f"Invalid or missing HHMM values in {c}.")
    for c in ["DISTANCE", "SCHEDULED_TIME"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        if not (np.isfinite(d[c]) & d[c].gt(0)).all():
            raise ValueError(f"{c} must contain positive finite numbers.")
        bounds = metadata.get("numeric_ranges", {}).get(c)
        if bounds and not d[c].between(bounds["min"], bounds["max"]).all():
            raise ValueError(f"{c} contains values outside the training range.")
    probabilities = model.predict_proba(make_features(d, metadata["engineered_features"]))[:, 1]
    result = d.copy()
    result.insert(0, "input_row", np.arange(1, len(d) + 1))
    result["disruption_probability"] = probabilities
    # Stable row-order tie breaking uses no outcome information.
    result = result.sort_values("disruption_probability", ascending=False, kind="stable").reset_index(drop=True)
    result["review_rank"] = np.arange(1, len(result) + 1)
    capacity = int(np.floor(len(result) * review_fraction))
    result["prioritize_for_review"] = result.review_rank.le(capacity)
    result["unseen_code"] = False
    for c in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        result["unseen_code"] |= ~result[c].isin(metadata.get("vocabulary", {}).get(c, []))
    return result
