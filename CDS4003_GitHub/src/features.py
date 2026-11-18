"""Stateless features restricted to the pre-departure information set."""
import numpy as np
import pandas as pd
from .cleaning import hhmm_minutes
from .schema import MODEL_RAW

BASE_NUMERIC = ["dep_minute", "arr_minute", "distance", "scheduled_minutes", "month", "weekday"]
BASE_CATEGORICAL = ["airline", "origin", "destination"]
EXTRA_NUMERIC = ["dep_sin", "dep_cos", "arr_sin", "arr_cos", "weekday_sin", "weekday_cos",
                 "month_sin", "month_cos", "weekend", "late_departure", "log_distance",
                 "scheduled_miles_per_hour", "origin_bts_id", "destination_bts_id"]

def feature_columns(engineered=True):
    return (BASE_NUMERIC + (EXTRA_NUMERIC if engineered else []),
            BASE_CATEGORICAL + (["route"] if engineered else []))

def make_features(frame, engineered=True):
    absent = set(MODEL_RAW) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing prediction inputs: {sorted(absent)}")
    # Deliberate allowlist: changing ANY outcome column cannot affect X.
    d = frame[MODEL_RAW].copy()
    date = pd.to_datetime(d.flight_date, errors="coerce")
    if date.isna().any():
        raise ValueError("Prediction inputs contain invalid flight dates")
    departure = hhmm_minutes(d.SCHEDULED_DEPARTURE)
    arrival = hhmm_minutes(d.SCHEDULED_ARRIVAL)
    calendar_date = date + pd.to_timedelta(pd.to_numeric(d.SCHEDULED_DEPARTURE, errors="coerce").eq(2400).astype(int), unit="D")
    out = pd.DataFrame(index=frame.index)
    out["dep_minute"], out["arr_minute"] = departure, arrival
    out["distance"] = pd.to_numeric(d.DISTANCE, errors="coerce").where(lambda x: (x > 0) & np.isfinite(x))
    out["scheduled_minutes"] = pd.to_numeric(d.SCHEDULED_TIME, errors="coerce").where(lambda x: (x > 0) & np.isfinite(x))
    out["month"], out["weekday"] = calendar_date.dt.month, calendar_date.dt.dayofweek
    for src, dest in [("AIRLINE", "airline"), ("ORIGIN_AIRPORT", "origin"), ("DESTINATION_AIRPORT", "destination")]:
        out[dest] = d[src].astype("string").str.strip().str.upper().fillna("UNKNOWN").replace("", "UNKNOWN").astype(object)
    if engineered:
        for name, values, period in [("dep", departure, 1440), ("arr", arrival, 1440), ("weekday", out.weekday, 7), ("month", out.month - 1, 12)]:
            out[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
            out[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
        out["weekend"] = out.weekday.ge(5).astype(float)
        out["late_departure"] = departure.ge(20 * 60).astype(float).where(departure.notna())
        out["log_distance"] = np.log1p(out.distance)
        # A schedule-derived proxy, not true airspeed; scheduled duration includes taxi time.
        out["scheduled_miles_per_hour"] = 60 * out.distance / out.scheduled_minutes
        out["origin_bts_id"] = out.origin.str.startswith("BTS_").astype(float)
        out["destination_bts_id"] = out.destination.str.startswith("BTS_").astype(float)
        out["route"] = out.origin + "_" + out.destination
    nums, cats = feature_columns(engineered)
    out[nums] = out[nums].replace([np.inf, -np.inf], np.nan).astype(float)
    return out[nums + cats]
