"""Run G7 continuous-holding rank-buffer sensitivity for one factor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cb_quant.config import load_backtest_config
from cb_quant.portfolio import run_rank_buffer_factor_portfolio
from run_g7_fixed_horizon_portfolio import _materialize_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="double_low")
    parser.add_argument(
        "--sample-partition",
        choices=("replication_2018_2023",),
        default="replication_2018_2023",
    )
    parser.add_argument("--exit-ranks", type=int, nargs="+", default=[20, 25, 30, 40])
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "backtest_v1.json",
    )
    parser.add_argument(
        "--g4-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G4",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "G7" / "rank_buffer_sensitivity",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _write_charts(
    nav_by_rank: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    initial_cash: float,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 5.5))
    for exit_rank, nav in sorted(nav_by_rank.items()):
        ordered = nav.sort_values("trade_date")
        axis.plot(
            ordered["trade_date"],
            ordered["nav"] / initial_cash,
            label=f"Exit below Top {exit_rank}",
            linewidth=1.2,
        )
    axis.set_title("Double Low: Continuous-Holding Rank-Buffer Net NAV")
    axis.set_xlabel("Trade Date")
    axis.set_ylabel("Net NAV")
    axis.legend(loc="upper left", ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rank_buffer_net_nav.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    ordered = summary.sort_values("exit_rank")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(ordered["exit_rank"].astype(str), ordered["annualized_turnover"])
    axes[0].set_title("Annualized One-Way Turnover")
    axes[0].set_xlabel("Exit Rank")
    axes[0].set_ylabel("Turnover")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        ordered["exit_rank"].astype(str),
        ordered["total_transaction_cost"] / initial_cash,
    )
    axes[1].set_title("Cumulative Cost / Initial Cash")
    axes[1].set_xlabel("Exit Rank")
    axes[1].set_ylabel("Cost Ratio")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rank_buffer_turnover_and_cost.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_backtest_config(args.config)
    portfolio_config = config["portfolio"]
    execution_config = config["execution"]
    if any(rank < int(portfolio_config["portfolio_size"]) for rank in args.exit_ranks):
        raise ValueError("every exit rank must be at least the portfolio size")
    if len(set(args.exit_ranks)) != len(args.exit_ranks):
        raise ValueError("exit ranks must be unique")

    partition_root = args.g4_root / args.sample_partition
    paths = {
        "factor_context": partition_root / "factor_context_v1.parquet",
        "execution_schedule": partition_root / "execution_schedule_v1.parquet",
        "execution_market": partition_root / "execution_market_v1.parquet",
        "lifecycle_panel": partition_root / "lifecycle_panel_v1.parquet",
    }
    if missing := [path for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing G4 inputs: " + ", ".join(map(str, missing)))

    context = pd.read_parquet(paths["factor_context"])
    schedule = pd.read_parquet(paths["execution_schedule"])
    market = pd.read_parquet(paths["execution_market"])
    lifecycle = pd.read_parquet(paths["lifecycle_panel"])
    market_end = pd.Timestamp(market["trade_date"].max()).normalize()
    schedule = schedule.loc[
        pd.to_datetime(schedule["entry_date"]).dt.normalize().le(market_end)
    ].copy()
    scores, factor_metadata = _materialize_scores(
        context,
        factor_name=args.factor,
        mad_multiplier=float(config["preprocessing"]["mad_multiplier"]),
    )

    payload = {
        "schema_version": "1.0",
        "stage": "G7_rank_buffer_sensitivity",
        "factor": factor_metadata,
        "sample_partition": args.sample_partition,
        "entry_rank": int(portfolio_config["portfolio_size"]),
        "exit_ranks": sorted(args.exit_ranks),
        "holding_rule": "continuous_until_next_rebalance_or_rank_exit",
        "terminal_liquidation_date": market_end.strftime("%Y-%m-%d"),
        "preprocessing": config["preprocessing"],
        "portfolio": portfolio_config,
        "execution": execution_config,
        "inputs": {name: _sha256(path) for name, path in paths.items()},
        "implementation": {
            "runner": _sha256(Path(__file__)),
            "rank_buffer": _sha256(
                PROJECT_ROOT / "src" / "cb_quant" / "portfolio" / "buffer.py"
            ),
            "execution_engine": _sha256(
                PROJECT_ROOT / "src" / "cb_quant" / "execution" / "engine.py"
            ),
        },
    }
    run_id = _run_id(payload)
    run_dir = args.output_root / args.factor / args.sample_partition / run_id
    if run_dir.exists():
        print((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        print(pd.read_csv(run_dir / "sensitivity_summary.csv").to_string(index=False))
        return 0

    common = {
        "portfolio_size": int(portfolio_config["portfolio_size"]),
        "cash_reserve": float(portfolio_config["cash_reserve"]),
        "max_weight": float(portfolio_config["maximum_single_name_weight"]),
        "liquidation_date": market_end,
        "initial_cash": float(portfolio_config["initial_cash_yuan"]),
        "board_lot": int(execution_config["board_lot"]),
    }
    net_runs = {}
    summary_rows = []
    one_way_cost = float(execution_config["one_way_cost_bps"]) / 10_000.0
    for exit_rank in sorted(args.exit_ranks):
        net = run_rank_buffer_factor_portfolio(
            scores,
            schedule,
            market,
            lifecycle,
            exit_rank=exit_rank,
            cost_rate=one_way_cost,
            **common,
        )
        gross = run_rank_buffer_factor_portfolio(
            scores,
            schedule,
            market,
            lifecycle,
            exit_rank=exit_rank,
            cost_rate=0.0,
            **common,
        )
        net_runs[exit_rank] = net
        audit = net.plan.rebalance_audit
        summary_rows.append(
            {
                "exit_rank": exit_rank,
                **net.summary,
                "gross_total_return": gross.summary["total_return"],
                "gross_annualized_return": gross.summary["annualized_return"],
                "gross_sharpe_ratio": gross.summary["sharpe_ratio"],
                "gross_max_drawdown": gross.summary["max_drawdown"],
                "average_retained_count": audit["retained_count"].iloc[1:].mean(),
                "average_entered_count": audit["entered_count"].iloc[1:].mean(),
                "average_exited_count": audit["exited_count"].iloc[1:].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("exit_rank").reset_index(drop=True)
    manifest = {
        **payload,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "result_scope": "G7 continuous-holding rank-buffer sensitivity",
    }

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g7_rank_buffer_", dir=run_dir.parent) as temp:
        destination = Path(temp)
        summary.to_csv(destination / "sensitivity_summary.csv", index=False)
        for exit_rank, run in net_runs.items():
            rank_dir = destination / f"exit_rank_{exit_rank}"
            rank_dir.mkdir()
            run.plan.selections.to_parquet(rank_dir / "selections.parquet", index=False)
            run.plan.rebalance_audit.to_csv(rank_dir / "rebalance_audit.csv", index=False)
            run.plan.targets.to_parquet(rank_dir / "targets.parquet", index=False)
            run.execution.nav.to_parquet(rank_dir / "nav.parquet", index=False)
            run.execution.positions.to_parquet(rank_dir / "positions.parquet", index=False)
            run.execution.orders.to_parquet(rank_dir / "orders.parquet", index=False)
            run.execution.fills.to_parquet(rank_dir / "fills.parquet", index=False)
            run.execution.attribution.to_parquet(rank_dir / "attribution.parquet", index=False)
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        _write_charts(
            {rank: run.execution.nav for rank, run in net_runs.items()},
            summary,
            destination,
            initial_cash=float(portfolio_config["initial_cash_yuan"]),
        )
        os.replace(destination, run_dir)

    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    display_columns = [
        "exit_rank",
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_turnover",
        "total_transaction_cost",
        "gross_annualized_return",
        "average_retained_count",
    ]
    print(summary[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
