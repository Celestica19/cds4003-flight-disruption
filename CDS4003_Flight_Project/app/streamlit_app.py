from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.title('Flight disruption prediction')
st.write('This app predicts if a flight will be disrupted (cancelled, diverted, or 15+ minutes late). '
         'Airline support staff can use it to see which flights may need attention.')
st.caption('Model: decision tree trained on US flight data from 2015 (CDS 4003 class project).')

# Load the saved model (the path works locally and on Streamlit Cloud)
model = joblib.load(Path(__file__).resolve().parents[1] / 'models' / 'model.joblib')
airlines = ['AA', 'AS', 'B6', 'DL', 'EV', 'F9', 'HA', 'MQ', 'NK', 'OO', 'UA', 'US', 'VX', 'WN']

with st.form('flight'):
    airline = st.selectbox('Airline code', airlines)
    month = st.selectbox('Month', range(1, 13))
    weekday = st.selectbox('Day of week (1 = Monday, 7 = Sunday)', range(1, 8))
    hour = st.slider('Planned departure hour', 0, 23, 9)
    distance = st.number_input('Distance (miles)', min_value=30, max_value=5000, value=500)
    duration = st.number_input('Planned flight time (minutes)', min_value=15, max_value=720, value=120)
    submitted = st.form_submit_button('Predict')

if submitted:
    # Simple input check: a plane cannot fly faster than about 650 miles per hour
    if distance / duration * 60 > 650:
        st.error('The distance is too long for this flight time. Please check the inputs.')
    else:
        flight = pd.DataFrame([{'MONTH': month, 'DAY_OF_WEEK': weekday,
                                'DEPARTURE_HOUR': hour, 'IS_WEEKEND': int(weekday >= 6),
                                'DISTANCE': distance, 'SCHEDULED_TIME': duration, 'AIRLINE': airline}])
        chance = model.predict_proba(flight)[0][1]
        if model.predict(flight)[0] == 1:
            st.warning(f'Prediction: DISRUPTED (model score {chance:.0%}). This flight may need staff attention.')
        else:
            st.success(f'Prediction: NOT disrupted (model score {chance:.0%}).')
        st.caption('The model does not use weather or live flight data, so a prediction can be wrong.')
