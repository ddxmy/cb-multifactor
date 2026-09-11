from __future__ import annotations

import json

from scripts.list_factors import main


def test_factor_inventory_lists_only_explicitly_registered_plugins(capsys) -> None:
    assert main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in payload] == [
        "cb_abnormal_turnover_20d",
        "cb_return_20d",
        "cb_stock_return_spread",
        "conversion_premium",
        "double_low",
        "relative_premium_to_parity_median",
        "stock_return_20d",
    ]
    spread = next(row for row in payload if row["name"] == "cb_stock_return_spread")
    assert spread["default_parameters"] == {"window": 20}
    assert spread["direction"] == -1
    assert "factor_template" not in {row["name"] for row in payload}


def test_factor_inventory_supports_compact_table_output(capsys) -> None:
    assert main(["--format", "table"]) == 0

    output = capsys.readouterr().out
    assert "cb_stock_return_spread" in output
    assert "conversion_premium" in output
    assert "double_low" in output
    assert "stock_bond_linkage" in output
