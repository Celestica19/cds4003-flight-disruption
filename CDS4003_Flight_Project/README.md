# Flight disruption prediction | CDS 4003

Salem Marwan Alebri - H00472169 (Celestica19)  
Ebrahim Abdulaziz Mohamed Ibrahim Almarzouqi - H00442189 (brhowm2will-jpg)

[GitHub repository](https://github.com/Celestica19/cds4003-flight-disruption) | [Existing app](https://cds4003-flight-disruption-npatkr4rrtvz5rg79napp9q.streamlit.app/)

**1. Business problem.** Predict whether a planned flight will be cancelled, diverted or arrive at least 15 minutes late. This can help airline support staff identify flights that may need attention. Compare two models and choose the better validation F1 score, which balances false alerts and missed disruptions.

**2. Data.** US DOT 2015 Flight Delays and Cancellations (Kaggle), CC0 Public Domain. The supplied random sample has 250,000 real rows and all 31 source columns. The original has 5,819,079 rows. Flights was selected for its schedules and outcomes; Online Retail II had too few columns, and NYC 311 needed a different outcome definition. Search notes are in the notebook.

**3. Cleaning.** Checked missing values, duplicates, clock formats and unusual values. There were no exact duplicates and all 250,000 outcomes were known. Invalid distances, durations and clocks become missing; model-input gaps are filled from training data. Early arrivals and unusually long but valid flights are retained. Actual delays and other outcomes are excluded from the predictors.

**4. Exploration and features.** Two charts compare disruption rates by airline and departure hour using training rows. Airline rates range from 11.9% to 30.8%; route differences may help explain this. Departure hour and a weekend indicator are created from the schedule. Seven planned inputs are used. The data is split into 200,000 training and 50,000 evaluation rows.

**5. Models.** Logistic regression learns weighted relationships; a decision tree makes a sequence of simple decisions. Both use balanced class weights because only 20.05% of flights are disrupted. Three-fold validation F1 is 0.370 for logistic regression and 0.381 for the tree. The tree is selected.

**6. Tuning and results.** GridSearchCV compares logistic C values 0.1, 1 and 10, and tree depths 3, 5 and 8. The chosen tree has depth 8. Final accuracy is 59.0%, precision 27.2%, recall 62.2% and F1 0.378. About 27 of 100 alerts are correct; about 62 of 100 disruptions are found. Always predicting normal gives 80.0% accuracy but finds no disruptions. False alerts remain common. These historical results do not establish future forecasting performance.

**7. App and running the project.** The basic Streamlit form takes airline, month, weekday, departure hour, distance and duration, then shows a prediction. The existing live link still shows the earlier app; this simpler version needs publishing. Use Python 3.11 or 3.12. Run: pip install -r requirements.txt. Open Flight_Project.ipynb with Jupyter and run all cells; launch the app with: streamlit run app/streamlit_app.py. Deploy that same app path from the team repository.

**Files and roles.** Flight_Project.ipynb contains the analysis; data/ contains the sample; models/ contains the saved model; app/ contains the form. requirements.txt lists packages. Assigned roles: Salem - data, charts and model explanation; Ebrahim - app, setup and deployment. These are assignments, not verified completed contributions.

**Sources and assistance.** Dataset: https://www.kaggle.com/datasets/usdot/flight-delays. References: scikit-learn.org/stable/user_guide.html and docs.streamlit.io. ChatGPT assisted with code and explanations; the team reported instructor approval for AI-assisted writing. Data is from 2015 and excludes live weather. This is a classroom demonstration.

