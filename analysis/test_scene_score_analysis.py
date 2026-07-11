import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.compare_scene_scores import pair_exports, summarize_delta
from navsim.common.dataclasses import PDMResults

ANALYSIS_DIR = str(Path(__file__).resolve().parent)
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)
from export_proposal_diagnostics import score_proposals


class SceneScoreComparisonTest(unittest.TestCase):
    def test_proposal_scoring_does_not_require_cached_pdm_progress(self):
        metric_cache = SimpleNamespace()
        loader = SimpleNamespace(
            metric_cache_paths={"scene": "unused"},
            get_from_token=lambda _token: metric_cache,
        )
        simulator = SimpleNamespace(proposal_sampling=SimpleNamespace())
        proposals = torch.zeros(1, 2, 8, 3)

        def fake_pdm_score(**kwargs):
            self.assertIs(kwargs["metric_cache"], metric_cache)
            self.assertFalse(hasattr(kwargs["metric_cache"], "pdm_progress"))
            return PDMResults(1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.75)

        with patch("export_proposal_diagnostics.pdm_score", side_effect=fake_pdm_score):
            scores = score_proposals(["scene"], proposals, loader, simulator, object())

        self.assertEqual(scores.shape, (1, 2, 7))

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
