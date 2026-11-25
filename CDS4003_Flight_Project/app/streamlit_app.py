from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.title('Flight disruption prediction')
st.write('A simple classroom example using US flight data from 2015.')
st.caption('Disruption means cancellation, diversion, or arrival at least 15 minutes late.')
model = joblib.load(Path(__file__).resolve().parents[1] / 'models/model.joblib')
airlines = ['AA', 'AS', 'B6', 'DL', 'EV', 'F9', 'HA', 'MQ', 'NK', 'OO', 'UA', 'US', 'VX', 'WN']

with st.form('flight'):
    airline = st.selectbox('Airline code', airlines)
    month = st.selectbox('Month in 2015', range(1, 13))
    weekday = st.selectbox('Day of week (1 = Monday, 7 = Sunday)', range(1, 8))
    hour = st.slider('Scheduled departure hour (local time)', 0, 23, 9)
    distance = st.number_input('Distance (miles)', min_value=1, max_value=6000, value=500)
    duration = st.number_input('Scheduled duration (minutes)', min_value=1, max_value=1000, value=120)
    submitted = st.form_submit_button('Predict')

if submitted:
    flight = pd.DataFrame([{'MONTH': month, 'DAY_OF_WEEK': weekday,
        'DEPARTURE_HOUR': hour, 'IS_WEEKEND': int(weekday >= 6),
        'DISTANCE': distance, 'SCHEDULED_TIME': duration, 'AIRLINE': airline}])
    if model.predict(flight)[0] == 1:
        st.warning('Predicted: disruption. This flight may need staff attention.')
    else:
        st.success('Predicted: no disruption. A disruption is still possible.')
    st.caption('This historical model does not use current weather or live flight information.')
