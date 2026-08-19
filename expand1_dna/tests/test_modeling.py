from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prime_dna.modeling import (  # noqa: E402
    _stratified_training_families,
    classification_metrics,
    expected_calibration_error,
    select_fusion_weight,
    select_macro_f1_threshold,
)


class ModelingTests(unittest.TestCase):
    def test_fusion_weight_never_exceeds_cap(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        baseline = np.asarray([0.45, 0.45, 0.55, 0.55])
        lineage = np.asarray([0.05, 0.10, 0.90, 0.95])
        weight, scan = select_fusion_weight(y, baseline, lineage, maximum_weight=0.5)
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 0.5)
        self.assertAlmostEqual(scan["weight"].max(), 0.5)

    def test_validation_threshold_improves_macro_f1(self):
        y = np.asarray([0, 0, 0, 1, 1], dtype=np.uint8)
        probability = np.asarray([0.1, 0.2, 0.3, 0.35, 0.4])
        threshold = select_macro_f1_threshold(y, probability)
        self.assertLessEqual(threshold, 0.4)
        self.assertGreater(classification_metrics(y, probability, threshold)["macro_f1"], 0.7)

    def test_single_class_family_metrics_are_defined_where_possible(self):
        y = np.zeros(4, dtype=np.uint8)
        probability = np.asarray([0.1, 0.2, 0.3, 0.4])
        metrics = classification_metrics(y, probability, 0.5)
        self.assertTrue(np.isnan(metrics["AUROC"]))
        self.assertTrue(np.isnan(metrics["AUPRC"]))
        self.assertTrue(np.isnan(metrics["balanced_accuracy"]))
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertTrue(np.isfinite(metrics["log_loss"]))

    def test_calibration_error_distinguishes_good_and_bad_probabilities(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        good = np.asarray([0.01, 0.05, 0.95, 0.99])
        bad = np.asarray([0.45, 0.45, 0.55, 0.55])
        self.assertLess(
            expected_calibration_error(y, good, n_bins=4),
            expected_calibration_error(y, bad, n_bins=4),
        )

    def test_extended_classification_metrics_match_confusion_counts(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        probability = np.asarray([0.1, 0.8, 0.9, 0.2])
        metrics = classification_metrics(y, probability, 0.5)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["precision_duplication"], 0.5)
        self.assertAlmostEqual(metrics["recall_duplication"], 0.5)

    def test_stratified_learning_subset_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "family_id": [f"F{i:03d}" for i in range(40)],
                "split": ["train"] * 30 + ["validation"] * 5 + ["test"] * 5,
                "size_bin": np.arange(40) % 5,
                "duplication_bin": (np.arange(40) // 5) % 5,
            }
        )
        first = _stratified_training_families(frame, 0.5, 42)
        second = _stratified_training_families(frame, 0.5, 42)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(len(first), 0)
        self.assertTrue(set(first).issubset(set(frame.index[:30])))

if __name__ == "__main__":
    unittest.main()
