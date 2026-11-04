from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_engine import decide
from simulator import generate_sequence, extract_features, feature_vector


class SmokeTests(unittest.TestCase):
    def test_normal_sequence_shape_and_features(self):
        pressure, imu = generate_sequence("normal", n_frames=40)
        self.assertEqual(pressure.shape, (40, 8, 8))
        self.assertEqual(imu.shape, (40, 3))
        vector = feature_vector(extract_features(pressure))
        self.assertEqual(vector.shape, (10,))
        self.assertTrue(np.isfinite(vector).all())

    def test_sudden_fall_decision_is_high_risk(self):
        action = decide(
            pred_label=3,
            proba=np.array([0.01, 0.01, 0.02, 0.95, 0.01]),
            vitals={"hr_alert": False, "bp_alert": False},
        )
        self.assertEqual(action.level, "红")
        self.assertEqual(action.scenario, "突发跌倒")
        self.assertTrue(action.notify_family)


if __name__ == "__main__":
    unittest.main()
