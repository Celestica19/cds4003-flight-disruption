import joblib
import pandas as pd
import streamlit as st

st.title('Flight disruption prediction')
st.write('This app predicts if a flight will be disrupted (cancelled, diverted, or 15+ minutes late). '
         'Airline support staff can use it to see which flights may need attention.')
st.write('Model: decision tree trained on US flight data from 2015.')

# load the saved model
model = joblib.load('models/model.joblib')

airlines = ['AA', 'AS', 'B6', 'DL', 'EV', 'F9', 'HA', 'MQ', 'NK', 'OO', 'UA', 'US', 'VX', 'WN']

airline = st.selectbox('Airline', airlines)
month = st.selectbox('Month', range(1, 13))
weekday = st.selectbox('Day of week (1 = Monday, 7 = Sunday)', range(1, 8))
hour = st.slider('Departure hour', 0, 23, 9)
distance = st.number_input('Distance (miles)', min_value=30, max_value=5000, value=500)
duration = st.number_input('Flight time (minutes)', min_value=15, max_value=720, value=120)

if st.button('Predict'):
    # check the input: a plane cannot fly faster than about 650 miles per hour
    if distance / duration * 60 > 650:
        st.error('The distance is too long for this flight time. Please check the inputs.')
    else:
        weekend = 0
        if weekday >= 6:
            weekend = 1

        # make one row with the same columns as the training data
        row = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
        row['MONTH'] = month
        row['DAY_OF_WEEK'] = weekday
        row['DEPARTURE_HOUR'] = hour
        row['IS_WEEKEND'] = weekend
        row['DISTANCE'] = distance
        row['SCHEDULED_TIME'] = duration
        row['AIRLINE_' + airline] = 1

        result = model.predict(row)[0]
        if result == 1:
            st.warning('Prediction: DISRUPTED. This flight may need staff attention.')
        else:
            st.success('Prediction: NOT disrupted.')
        st.write('Note: the model does not use weather, so the prediction can be wrong.')
