"""Historical flight disruption decision-support demonstration."""
import json
import sys
from datetime import date, time
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import sha256_file
from src.evaluation import validate_input
from src.features import make_features
from src.batch import prioritize_batch

st.set_page_config(page_title="Flight disruption planner", page_icon="✈", layout="wide")
st.title("Flight disruption planner")
st.write("Estimate historical disruption risk from a flight's planned schedule to help prioritize passenger-support review.")
st.caption("A disruption means cancellation, diversion, or arrival at least 15 minutes late. This is a historical US 2015 demonstration.")

@st.cache_resource
def load_model():
    meta_path = ROOT / "models/metadata.json"
    model_path = ROOT / "models/model.joblib"
    if not meta_path.exists() or not model_path.exists():
        return None, None
    metadata = json.loads(meta_path.read_text())
    if metadata.get("status") != "TRAINED_REAL_DATA":
        raise ValueError("Only a model trained on the verified real dataset may be served.")
    if sha256_file(model_path) != metadata["model_sha256"]:
        raise ValueError("The model file does not match its recorded checksum.")
    # Only load the team's trusted artifact. Never accept uploaded pickle/joblib files.
    return joblib.load(model_path), metadata

try:
    model, meta = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()
if model is None:
    st.info("Predictions will become available after the real dataset has been processed and the model has been trained.")
    st.stop()

lookup_path = ROOT / "models/lookups.json"
lookups = json.loads(lookup_path.read_text()) if lookup_path.exists() else {"airlines": {}, "airports": {}}
def display_code(kind, code):
    name = lookups[kind].get(code)
    return f"{code} · {name}" if name else code

with st.form("flight"):
    left, middle, right = st.columns(3)
    with left:
        service_date = st.date_input("Historical service date", date(2015, 11, 15), min_value=date(2015, 1, 1), max_value=date(2015, 12, 31))
        airline = st.selectbox("Airline", meta["vocabulary"]["AIRLINE"], format_func=lambda x: display_code("airlines", x))
        origin = st.selectbox("Origin airport", meta["vocabulary"]["ORIGIN_AIRPORT"], format_func=lambda x: display_code("airports", x))
    with middle:
        destination = st.selectbox("Destination airport", meta["vocabulary"]["DESTINATION_AIRPORT"], format_func=lambda x: display_code("airports", x))
        departure = st.time_input("Scheduled departure (origin local time)", time(9, 0))
        arrival = st.time_input("Scheduled arrival (destination local time)", time(11, 0))
    with right:
        distance = st.number_input("Route distance (miles)", min_value=1.0, value=500.0, step=10.0)
        duration = st.number_input("Scheduled gate-to-gate duration (minutes)", min_value=1.0, value=120.0, step=5.0)
        st.caption("Enter scheduled duration separately: local arrival and departure clocks may use different time zones.")
    submitted = st.form_submit_button("Estimate disruption risk", type="primary")

if submitted:
    try:
        inputs = validate_input(service_date, airline, origin, destination,
            departure.hour * 100 + departure.minute, arrival.hour * 100 + arrival.minute, distance, duration)
        for field, value in [("DISTANCE", distance), ("SCHEDULED_TIME", duration)]:
            bounds = meta["numeric_ranges"].get(field)
            if bounds and not bounds["min"] <= value <= bounds["max"]:
                raise ValueError(f"{field.replace('_', ' ').title()} is outside the observed training range ({bounds['min']:g}–{bounds['max']:g}).")
        probability = float(model.predict_proba(make_features(inputs, meta["engineered_features"]))[0, 1])
        alert = probability >= meta["threshold"]
        st.metric("Estimated disruption probability", f"{probability:.1%}")
        if alert:
            st.warning("Prioritize for passenger-support review under the selected historical workload policy.")
        else:
            st.success("Standard review priority under the selected historical workload policy.")
        st.write("Use this estimate to plan review capacity; a low score does not guarantee an on-time flight.")
        st.caption(f"Historical validation alert budget: {meta['alert_budget']:.0%}. A fixed cutoff does not guarantee the same workload on new data.")
    except ValueError as error:
        st.error(str(error))

with st.expander("Plan a batch review with a fixed capacity"):
    st.write("Upload historical flights to rank them and select exactly the available review capacity. This avoids relying on a fixed score cutoff when the score distribution changes.")
    st.caption("CSV columns: flight_date, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, SCHEDULED_DEPARTURE, SCHEDULED_ARRIVAL, SCHEDULED_TIME, DISTANCE. Dates must be in 2015; times use local HHMM. Maximum 50,000 rows.")
    budget = st.slider("Flights to prioritize (%)", min_value=1, max_value=50, value=10)
    upload = st.file_uploader("Flight schedule CSV", type=["csv"])
    if upload is not None:
        try:
            schedule = pd.read_csv(upload, nrows=50001)
            ranked = prioritize_batch(schedule, model, meta, budget / 100)
            st.write(f"{int(ranked.prioritize_for_review.sum()):,} of {len(ranked):,} flights selected for review. Capacity is rounded down to whole flights.")
            if ranked.unseen_code.any():
                st.warning("Some codes were not seen during training. Those rows are flagged; their predictions need extra caution.")
            st.dataframe(ranked, hide_index=True)
            st.download_button("Download ranked review list", ranked.to_csv(index=False), file_name="ranked_flight_review.csv", mime="text/csv")
        except Exception as error:
            st.error(f"Cannot process this batch: {error}")

tab1, tab2, tab3 = st.tabs(["Performance", "What the model uses", "Scope and limitations"])
with tab1:
    path = ROOT / "reports/test_metrics.json"
    if path.exists():
        result = json.loads(path.read_text())["selected"]
        a, b, c = st.columns(3)
        a.metric("Test average precision", f"{result['average_precision']:.3f}")
        b.metric("Precision at review budget", f"{result['precision_at_budget']:.1%}")
        c.metric("Disruptions captured at budget", f"{result['recall_at_budget']:.1%}")
        st.write(f"Evaluated on {result['n']:,} held-out flights. The observed disruption rate was {result['prevalence']:.1%}.")
        st.caption("Ranking at a fixed review budget differs from applying one score threshold to each flight.")
    else:
        st.info("Held-out evaluation is not available yet.")
with tab2:
    st.write("Inputs include airline, route, calendar patterns, scheduled local times, distance and planned duration. Actual delay outcomes and delay causes are excluded.")
    path = ROOT / "reports/permutation_importance.csv"
    if path.exists():
        importance = pd.read_csv(path).head(10).set_index("feature")
        st.bar_chart(importance["mean_ap_drop"])
        st.caption("Global validation importance, measured by the decrease in average precision after shuffling a feature. It does not explain the cause of an individual flight's outcome.")
with tab3:
    for limitation in meta["limitations"]:
        st.write("• " + limitation)
    st.write("Support planning only; not for flight safety, dispatch, regulatory decisions, or automated passenger entitlements.")
    st.markdown("[Dataset source and CC0 licence](https://www.kaggle.com/datasets/usdot/flight-delays)")
