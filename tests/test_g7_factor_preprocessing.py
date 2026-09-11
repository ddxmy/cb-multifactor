from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_g7_fixed_horizon_portfolio import _materialize_scores


def test_g7_scores_reuse_materialized_g3c_standardization() -> None:
    context = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")] * 4,
            "ts_code": ["CB1", "CB2", "CB3", "CB4"],
            "is_eligible": True,
            "relative_premium_to_parity_median": [0.1, 0.2, 0.3, 0.4],
            "winsorized__relative_premium_to_parity_median": [0.1, 0.2, 0.3, 0.4],
            "standardized__relative_premium_to_parity_median": [
                1.5,
                0.5,
                -0.5,
                -1.5,
            ],
        }
    )

    scores, metadata = _materialize_scores(
        context,
        factor_name="relative_premium_to_parity_median",
        mad_multiplier=3.0,
    )

    assert scores["processed_factor"].tolist() == pytest.approx(
        [-1.5, -0.5, 0.5, 1.5]
    )
    assert metadata["preprocessing_source"] == "materialized_g3c"
