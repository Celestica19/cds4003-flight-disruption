# Source and documentation links

Checked on 18 November 2026. These links support the implementation and dataset-selection record; source-reported dimensions must still be verified from downloaded files.

1. US Department of Transportation dataset on Kaggle. *2015 Flight Delays and Cancellations*. Source, producer description and CC0 Public Domain licence: https://www.kaggle.com/datasets/usdot/flight-delays
2. Bureau of Transportation Statistics. *Data Dictionary*. On-time arrival means less than 15 minutes after the published arrival time: https://transtats.bts.gov/printglossary.asp
3. Bureau of Transportation Statistics. *Number 22 Technical Directive, effective January 1, 2013*. Historical reporting and on-time definitions relevant to the dataset period: https://www.bts.gov/topics/airlines-and-airports/number-22-technical-directive-time-reporting-effective-jan-1-2013
4. US Department of Transportation. *Air Travel Consumer Reports*. Provenance and reporting context: https://www.transportation.gov/individuals/aviation-consumer-protection/air-travel-consumer-reports
5. scikit-learn. *Common pitfalls and recommended practices*. Leakage and consistent preprocessing: https://scikit-learn.org/stable/common_pitfalls.html
6. scikit-learn. *TimeSeriesSplit*. Time-ordered validation concepts. This implementation uses explicit calendar folds to keep whole dates together rather than equal row-count folds: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
7. Streamlit. *Prep and deploy your app on Community Cloud*: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
8. UCI Machine Learning Repository. *Online Retail II*. Rejected candidate; 1,067,371 records but eight source variables: https://archive.ics.uci.edu/dataset/502/online+retail+ii
9. NYC Open Data. *311 Service Requests from 2020 to Present*. Alternative large public dataset considered: https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9
10. Uploaded course brief. *CDS4003 Student Project 202610*. Assignment requirements, milestone approvals, assessment rubric and AI-use restrictions.

The uploaded original release was locally counted: 5,819,079 rows and 31 columns. The audit measured numeric airport identifiers in 486,165 origin and 486,165 destination fields. Both findings are supported by the original-file audit and an independent read of the source.

11. Bureau of Transportation Statistics. Master Coordinate field definitions: AirportID is a persistent airport identifier; Airport codes can change or be reused. https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FLL

The original archive was supplied by the user on 18 November 2026. An independent full-file count verified 5,819,079 flight records. A direct request for the official historical airport-reference archive also returned HTTP 403, so no guessed crosswalk is supplied.
