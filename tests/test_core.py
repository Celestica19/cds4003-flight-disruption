"""Small artificial test fixtures only; these are NOT the project dataset."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.cleaning import DuplicateIndex, clean_chunk, hhmm_minutes, row_hashes, split_labels
from src.common import config
from src.evaluation import budget_threshold, metrics, validate_input
from src.features import make_features
from src.modeling import model_pipeline, temporal_folds
from src.schema import COLUMNS, POST_EVENT
from scripts.prepare_data import prepare

def fixture_frame(n_per_month=40):
    """Deterministic, deliberately small records for software tests, not research."""
    rows = []
    for month in range(1, 13):
        for i in range(n_per_month):
            row = {c: "0" for c in COLUMNS}
            row.update(YEAR="2015", MONTH=str(month), DAY=str(1 + i % 20), DAY_OF_WEEK="1",
                AIRLINE="AA" if i % 2 else "DL", FLIGHT_NUMBER=str(month * 1000 + i),
                TAIL_NUMBER="N123AB", ORIGIN_AIRPORT="JFK" if i % 3 else "SFO",
                DESTINATION_AIRPORT="LAX", SCHEDULED_DEPARTURE=str(800 + (i % 10) * 100),
                SCHEDULED_ARRIVAL="1400", DEPARTURE_TIME="815", ARRIVAL_TIME="1415",
                WHEELS_OFF="830", WHEELS_ON="1400", DEPARTURE_DELAY="15",
                TAXI_OUT="15", TAXI_IN="15", SCHEDULED_TIME="360", ELAPSED_TIME="360",
                AIR_TIME="330", DISTANCE=str(2500 + i), ARRIVAL_DELAY="25" if i % 4 == 0 else "-5",
                CANCELLED="0", DIVERTED="0", CANCELLATION_REASON="")
            for c in ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]:
                row[c] = ""
            if i % 13 == 0:
                row.update(CANCELLED="1", ARRIVAL_DELAY="", ARRIVAL_TIME="", CANCELLATION_REASON="B")
            rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]

class CleaningTests(unittest.TestCase):
    def test_cancellation_is_not_imputed_arrival(self):
        raw = fixture_frame(2).head(2)
        good, bad, counts = clean_chunk(raw)
        self.assertTrue(bad.empty)
        self.assertEqual(good.iloc[0].disrupted, 1)
        self.assertTrue(pd.isna(good.iloc[0].ARRIVAL_DELAY))
        self.assertEqual(good.iloc[1].ARRIVAL_DELAY, -5)
        self.assertEqual(good.iloc[1].disrupted, 0)

    def test_unknown_target_and_contradictory_flags_are_quarantined(self):
        raw = fixture_frame(4).head(4)
        raw.loc[1, "ARRIVAL_DELAY"] = ""
        raw.loc[2, "CANCELLED"] = "yes"
        raw.loc[3, ["CANCELLED", "DIVERTED"]] = ["1", "1"]
        good, bad, _ = clean_chunk(raw)
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 3)
        self.assertEqual(set(bad.quarantine_reason), {"unknown_target", "contradictory_outcome_flags"})

    def test_bad_dates_and_domain_values(self):
        raw = fixture_frame(4).head(4)
        raw.loc[0, "DAY"] = "32"
        raw.loc[1, "DISTANCE"] = "-20"
        raw.loc[2, "SCHEDULED_DEPARTURE"] = "1260"
        raw.loc[3, "SCHEDULED_TIME"] = "inf"
        good, bad, _ = clean_chunk(raw)
        self.assertEqual(len(bad), 1)
        self.assertTrue(pd.isna(good.loc[1, "DISTANCE"]))
        self.assertTrue(pd.isna(good.loc[2, "SCHEDULED_DEPARTURE"]))
        self.assertTrue(pd.isna(good.loc[3, "SCHEDULED_TIME"]))

    def test_airport_namespace_and_verified_mapping(self):
        raw = fixture_frame(2).head(2)
        raw.loc[0, "ORIGIN_AIRPORT"] = "10397"
        raw.loc[1, "ORIGIN_AIRPORT"] = " jfk "
        good, _, _ = clean_chunk(raw)
        self.assertEqual(good.loc[0, "ORIGIN_AIRPORT"], "BTS_10397")
        self.assertEqual(good.loc[1, "ORIGIN_AIRPORT"], "JFK")
        mapped, _, _ = clean_chunk(raw, airport_map={"10397": "ATL"})
        self.assertEqual(mapped.loc[0, "ORIGIN_AIRPORT"], "ATL")

    def test_duplicate_detection_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = DuplicateIndex(Path(tmp) / "d.sqlite")
            self.assertEqual(idx.keep(["a", "b"], ["key", "key"]).tolist(), [True, True])
            self.assertEqual(idx.keep(["b", "c"], ["key", "other"]).tolist(), [False, True])
            self.assertEqual(idx.conflicting_business_keys(), 1)
            idx.close()

    def test_clock_parsing_and_midnight(self):
        values = hhmm_minutes(pd.Series([5, 930, 2359, 2400, 1260, -1]))
        np.testing.assert_allclose(values.iloc[:4], [5, 570, 1439, 0])
        self.assertTrue(values.iloc[4:].isna().all())

    def test_strict_more_than_300k_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw_dir = root / "raw"; raw_dir.mkdir()
            fixture_frame(4).to_csv(raw_dir / "flights.csv", index=False)
            cfg = config(); cfg["chunk_size"] = 11
            with self.assertRaisesRegex(ValueError, "300,000"):
                prepare(raw_dir, root, cfg)
            audit = json.loads((root / "reports/quality_audit.json").read_text())
            self.assertEqual(audit["status"], "REQUIREMENT_FAILED")

class LeakageAndModelTests(unittest.TestCase):
    def setUp(self):
        self.good, _, _ = clean_chunk(fixture_frame())

    def test_outcomes_cannot_change_features(self):
        modified = self.good.copy()
        for c in POST_EVENT:
            modified[c] = "UNAVAILABLE_AT_PREDICTION"
        modified["disrupted"] = 1 - modified.disrupted
        assert_frame_equal(make_features(self.good), make_features(modified))

    def test_preprocessing_is_fit_on_training_only(self):
        features = make_features(self.good.head(20))
        features.loc[features.index[0], "distance"] = np.nan
        model = model_pipeline("logistic_sgd")
        model.fit(features, self.good.disrupted.head(20))
        expected = features.distance.median()
        stats = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
        self.assertEqual(stats[2], expected)
        unseen = features.tail(2).copy(); unseen["airline"] = "ZZ"; unseen["distance"] = 1000000
        prediction = model.predict_proba(unseen)
        self.assertEqual(prediction.shape, (2, 2))
        np.testing.assert_equal(stats, model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_)

    def test_temporal_split_and_embargo(self):
        dates = pd.Series(["2015-08-29", "2015-08-30", "2015-09-01", "2015-10-30", "2015-11-01"])
        self.assertEqual(split_labels(dates, config()).tolist(), ["train", "embargo", "validation", "embargo", "test"])
        for train, val in temporal_folds(self.good.flight_date):
            self.assertLess(pd.to_datetime(self.good.flight_date.iloc[train]).max(), pd.to_datetime(self.good.flight_date.iloc[val]).min() - pd.Timedelta(days=2))

    def test_native_categorical_tree_accepts_unseen_airport(self):
        X = make_features(self.good)
        model = model_pipeline("hist_gradient_boosting")
        model.set_params(classifier__max_iter=5)
        model.fit(X, self.good.disrupted)
        unseen = X.head(1).copy(); unseen["origin"] = "BTS_99999"; unseen["route"] = "NEW_ROUTE"
        probability = model.predict_proba(unseen)[0, 1]
        self.assertTrue(0 <= probability <= 1)

    def test_budget_handles_constant_scores(self):
        p = np.full(100, 0.2)
        t = budget_threshold(p, .1)
        self.assertEqual(int((p >= t).sum()), 0)
        p = np.linspace(0, 1, 100)
        self.assertLessEqual((p >= budget_threshold(p, .1)).mean(), .1)

    def test_metrics_counts_and_cost(self):
        m = metrics([0, 0, 1, 1], [.1, .7, .2, .9], .5, budget=.5)
        self.assertEqual([m[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertEqual(m["scenario_cost_per_1000"], 1500)

    def test_app_input_validation(self):
        with self.assertRaises(ValueError):
            validate_input("2015-11-01", "AA", "JFK", "JFK", 900, 1200, 500, 120)
        with self.assertRaises(ValueError):
            validate_input("2015-11-01", "AA", "JFK", "LAX", 1260, 1200, 500, 120)
        with self.assertRaises(ValueError):
            validate_input("2015-11-01", "AA", "JFK", "LAX", 900, 1200, -1, 120)

if __name__ == "__main__":
    unittest.main()
