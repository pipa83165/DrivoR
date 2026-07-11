import unittest

import numpy as np
import pandas as pd

from analysis.compare_scene_scores import pair_exports, summarize_delta


class SceneScoreComparisonTest(unittest.TestCase):
    def test_generation_ranking_decomposition_and_missing_tokens(self):
        baseline = pd.DataFrame(
            {
                "token": ["a", "b", "baseline_only"],
                "selected_score": [0.4, 0.7, 0.1],
                "oracle_score": [0.6, 0.8, 0.2],
                "ranking_regret": [0.2, 0.1, 0.1],
            }
        )
        variant = pd.DataFrame(
            {
                "token": ["a", "b", "variant_only"],
                "selected_score": [0.7, 0.6, 0.3],
                "oracle_score": [0.8, 0.9, 0.4],
                "ranking_regret": [0.1, 0.3, 0.1],
            }
        )

        paired, missing = pair_exports(
            baseline,
            variant,
            ["selected_score", "oracle_score", "ranking_regret"],
            "selected_score",
        )

        np.testing.assert_allclose(
            paired["delta_score"],
            paired["delta_oracle_score"] + paired["delta_ranking_ability"],
        )
        self.assertEqual(set(paired["token"]), {"a", "b"})
        self.assertEqual(set(missing["token"]), {"baseline_only", "variant_only"})

    def test_delta_summary_reports_win_tie_loss(self):
        summary = summarize_delta(
            pd.Series([0.2, 0.0, -0.1]),
            "delta_score",
            tolerance=1e-9,
            bootstrap_samples=20,
            rng=np.random.default_rng(0),
        )
        self.assertAlmostEqual(summary["win_rate"], 1 / 3)
        self.assertAlmostEqual(summary["tie_rate"], 1 / 3)
        self.assertAlmostEqual(summary["loss_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
