from __future__ import annotations

import pandas as pd
import pytest

from cb_quant.factors import FactorDataBundle
from cb_quant.factors.double_low import FACTOR_SPEC, calculate_factor


def make_bundle() -> FactorDataBundle:
    return FactorDataBundle(
        signal=pd.DataFrame(
            {
                "signal_date": ["2024-01-05", "2024-01-05"],
                "ts_code": ["110001.SH", "110002.SH"],
                "close": [108.63, 120.00],
                "conversion_premium": [0.248383, float("nan")],
            }
        )
    )


def test_double_low_matches_researcher_hand_calculation() -> None:
    result = calculate_factor(make_bundle(), {}).set_index("ts_code")

    assert result.loc["110001.SH", "raw_factor"] == pytest.approx(133.4683)
    assert result.loc["110001.SH", "available_date"] == pd.Timestamp("2024-01-05")
    assert pd.isna(result.loc["110002.SH", "raw_factor"])
    assert pd.isna(result.loc["110002.SH", "available_date"])


def test_double_low_metadata_and_parameter_contract_are_explicit() -> None:
    assert FACTOR_SPEC.name == "double_low"
    assert FACTOR_SPEC.direction == -1
    assert FACTOR_SPEC.required_fields == ("close", "conversion_premium")

    with pytest.raises(ValueError, match="does not accept parameters"):
        calculate_factor(make_bundle(), {"weight": 0.5})
