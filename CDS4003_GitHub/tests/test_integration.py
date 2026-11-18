"""Runs all computation stages on temporary small test fixtures only."""
import tempfile
import unittest
from pathlib import Path
import json
from src.common import config
from tests.test_core import fixture_frame
from scripts.prepare_data import prepare
from scripts.train_models import train_project
from scripts.build_visuals import build_visuals

class IntegrationTests(unittest.TestCase):
    def test_end_to_end_fixture_is_marked_and_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "data/raw"; raw.mkdir(parents=True)
            frame = fixture_frame(40)
            # Add an exact duplicate crossing the chunk boundary.
            import pandas as pd
            frame = pd.concat([frame, frame.head(1)], ignore_index=True)
            frame.to_csv(raw / "flights.csv", index=False)
            cfg = config(); cfg.update(chunk_size=75, tuning_rows=320, search_iterations=1,
                parallel_jobs=1, eda_sample_rows=100, bootstrap_repetitions=20)
            audit = prepare(raw, root, cfg, fixture_mode=True)
            self.assertEqual(audit["status"], "TEST_FIXTURE_ONLY")
            self.assertEqual(audit["counts"]["exact_duplicates_removed"], 1)
            train_project(root, cfg, fixture_mode=True)
            metadata = json.loads((root / "models/metadata.json").read_text())
            self.assertEqual(metadata["status"], "TEST_FIXTURE_ONLY")
            figures = build_visuals(root, cfg)
            self.assertEqual(len(figures), 10)
            self.assertTrue((root / "reports/test_subgroup_metrics.csv").is_file())
            with self.assertRaises(ValueError):
                train_project(root, cfg, fixture_mode=True)

if __name__ == "__main__":
    unittest.main()
