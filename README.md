# Flight Disruption Prediction (CDS 4003)

**Live app:** https://cds4003-flight-disruption-cpyrlyk54gga4pgmrcwvcj.streamlit.app/
**GitHub:** https://github.com/Celestica19/cds4003-flight-disruption

## Team and roles
| Name | ID | GitHub | Role |
|---|---|---|---|
| Salem Marwan Alebri | H00472169 | Celestica19 | Data cleaning, EDA, models |
| Ebrahim Abdulaziz Mohamed Ibrahim Almarzouqi | H00442189 | brhowm2will-jpg | Streamlit app, deployment | Auditing

## Business problem
Flight delays and cancellations give airline support staff extra work. We predict from the planned schedule if a flight will be **disrupted** (cancelled, diverted or 15+ minutes late), so staff can check risky flights early. We compare two models and choose the one with the better F1 score.

## Dataset
- **Source:** [2015 Flight Delays and Cancellations, US DOT (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Licence:** CC0 Public Domain
- **Size:** full `flights.csv` has 5,819,079 rows and 31 columns. It is too big for GitHub (590 MB), so `data/flights.csv.gz` is a random sample of 250,000 rows with all 31 columns (code for the sample is in the notebook).
- **How to get the full file:** download the zip from the Kaggle link above.
- **Limitations:** the data is only from US flights in 2015 and has no weather, so it may not match other countries or today. Some airlines have more flights than others, so the model may be less accurate for small airlines.

**Dataset search:**
| Dataset | Site | Decision |
|---|---|---|
| 2015 Flight Delays and Cancellations | Kaggle | Chosen: large, 31 columns, clear target |
| Online Retail II | UCI | Rejected: only 8 columns |
| NYC 311 Service Requests | NYC Open Data | Rejected: no clear target for our problem |

**Main columns used:**
| Column | Meaning |
|---|---|
| AIRLINE | Airline code |
| MONTH, DAY_OF_WEEK | Date of the flight |
| SCHEDULED_DEPARTURE | Planned departure time (HHMM) |
| DISTANCE | Distance in miles |
| SCHEDULED_TIME | Planned flight time in minutes |
| ARRIVAL_DELAY, CANCELLED, DIVERTED | Used only to make the target `DISRUPTED` |

## Folder structure
```
Flight_Project.ipynb   all the work in one notebook (cleaning, EDA, models, tuning)
data/                  250,000-row sample of the Kaggle data
models/                saved model (model.joblib)
app/                   Streamlit app (streamlit_app.py)
requirements.txt       Python packages
```

## How to run
```
pip install -r requirements.txt
pip install notebook
jupyter notebook Flight_Project.ipynb      (Run All)
streamlit run app/streamlit_app.py
```

## Results
| Model | Best setting | Cross-validation F1 |
|---|---|---|
| Logistic regression | C = 0.1 | 0.368 |
| Decision tree | max_depth = 8 | **0.378** |

Baseline (always "not disrupted"): accuracy 0.797 but F1 0.0. Tuning improved the tree F1 from 0.259 to 0.384. The decision tree was chosen. On the 20% test set: accuracy 0.587, precision 0.275, recall 0.637, F1 0.384. It finds about 64 of every 100 disrupted flights, but about 3 of every 4 alerts are false. The model uses 2015 data and no weather, so it is a class example only. The most important inputs are departure hour and month.

## AI use
AI tools (ChatGPT, Claude) were used to help with the code.
