"""Batch capacity checks on small software fixtures."""
import unittest
from src.batch import prioritize_batch
from src.cleaning import clean_chunk
from src.features import make_features
from src.modeling import model_pipeline
from tests.test_core import fixture_frame

class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _, _ = clean_chunk(fixture_frame())
        cls.model = model_pipeline("logistic_sgd")
        cls.model.fit(make_features(cls.rows), cls.rows.disrupted)
        cls.metadata = {"historical_year": 2015, "engineered_features": True,
            "vocabulary": {c: cls.rows[c].unique().tolist() for c in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]}}

    def test_review_capacity_is_exact_and_ranking_is_monotone(self):
        result = prioritize_batch(self.rows.head(103), self.model, self.metadata, .1)
        self.assertEqual(result.prioritize_for_review.sum(), 10)
        self.assertTrue(result.disruption_probability.is_monotonic_decreasing)
        self.assertEqual(set(result.input_row), set(range(1, 104)))

    def test_invalid_date_and_missing_column_are_rejected(self):
        invalid = self.rows.head(5).copy(); invalid["flight_date"] = "2026-11-18"
        with self.assertRaises(ValueError):
            prioritize_batch(invalid, self.model, self.metadata)
        with self.assertRaises(ValueError):
            prioritize_batch(self.rows.drop(columns="DISTANCE"), self.model, self.metadata)

    def test_outcome_columns_do_not_change_ranking(self):
        changed = self.rows.copy()
        changed["disrupted"] = 1 - changed.disrupted
        changed["ARRIVAL_DELAY"] = 999999
        original = prioritize_batch(self.rows, self.model, self.metadata)
        modified = prioritize_batch(changed, self.model, self.metadata)
        self.assertEqual(original.input_row.tolist(), modified.input_row.tolist())
        self.assertEqual(original.disruption_probability.tolist(), modified.disruption_probability.tolist())

if __name__ == "__main__":
    unittest.main()
