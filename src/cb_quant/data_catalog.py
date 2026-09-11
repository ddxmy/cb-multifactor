"""Central, read-only catalog for local convertible bond data sources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .legacy_schema import (
    LEGACY_2025_DAILY_FILENAME,
    LEGACY_A_SHARE_DAILY_DIRECTORY,
    LEGACY_CB_INTRADAY_DIRECTORY,
    LEGACY_CB_MARKET_DIRECTORY,
    LEGACY_REDEMPTION_FILENAME,
)


SUPPORTED_INTRADAY_FREQUENCIES = ("1m", "5m", "15m", "30m", "60m", "day")


@dataclass(frozen=True)
class DataCatalog:
    """Resolve external source paths without copying or modifying raw data."""

    research_root: Path
    legacy_project_root: Path
    intraday_root: Path
    database_root: Path | None = None

    @classmethod
    def from_environment(cls) -> "DataCatalog":
        default_research_root = Path(__file__).resolve().parents[2]
        research_root = Path(
            os.environ.get("CB_MULTIFACTOR_ROOT", default_research_root)
        ).expanduser()
        legacy_project_root = Path(
            os.environ.get("CB_LEGACY_PROJECT_ROOT", research_root.parent)
        ).expanduser()
        database_root_value = os.environ.get("CB_DATABASE_ROOT")
        if not database_root_value:
            raise EnvironmentError(
                "CB_DATABASE_ROOT must point to the local database directory"
            )
        database_root = Path(database_root_value).expanduser()
        intraday_override = os.environ.get("CB_INTRADAY_ROOT")
        if intraday_override:
            intraday_root = Path(intraday_override).expanduser()
        else:
            intraday_candidates = [
                database_root.joinpath(*LEGACY_CB_INTRADAY_DIRECTORY),
                database_root.joinpath(*LEGACY_CB_MARKET_DIRECTORY),
            ]
            intraday_root = next(
                (path for path in intraday_candidates if path.exists()),
                intraday_candidates[0],
            )
        return cls(
            research_root=research_root.resolve(),
            legacy_project_root=legacy_project_root.resolve(),
            intraday_root=intraday_root.resolve(),
            database_root=database_root.resolve(),
        )

    @property
    def local_database_root(self) -> Path:
        """Root that holds the local DuckDB datasets.

        ``database_root`` is optional so manually constructed catalogs in tests
        remain concise. The fallback follows the current intraday directory layout.
        """
        if self.database_root is not None:
            return self.database_root
        return self.intraday_root.parents[1]

    @property
    def tushare_raw_root(self) -> Path:
        return self.legacy_project_root / "data" / "raw"

    @property
    def legacy_processed_root(self) -> Path:
        return self.legacy_project_root / "data" / "processed"

    @property
    def tushare_history_root(self) -> Path:
        """Project-local cache for point-in-time Tushare event tables."""
        return self.research_root / "data" / "interim" / "tushare_history"

    @property
    def cb_basic_path(self) -> Path:
        return self.tushare_raw_root / "cb_basic_since_20180101_20260615.csv"

    @property
    def cb_daily_path(self) -> Path:
        return self.tushare_raw_root / "cb_daily_market_20180101_20260707.csv"

    @property
    def stock_daily_path(self) -> Path:
        return self.tushare_raw_root / "stock_daily_for_cb_underlyings_20180102_20260707.csv"

    @property
    def industry_membership_path(self) -> Path:
        return self.tushare_raw_root / "sw2021_l1_industry_membership_20260717.csv"

    @property
    def industry_index_daily_path(self) -> Path:
        return self.tushare_raw_root / "sw2021_l1_index_daily_20180101_20260707.csv"

    @property
    def legacy_feature_panel_path(self) -> Path:
        return self.legacy_processed_root / "cb_feature_panel_20180102_20260707.csv"

    @property
    def cb_terms_path(self) -> Path:
        return self.intraday_root / "cb_list.duckdb"

    @property
    def redemption_path(self) -> Path:
        return self.intraday_root / LEGACY_REDEMPTION_FILENAME

    @property
    def ci_l1_daily_path(self) -> Path:
        """Point-in-time CITIC level-1 industry index and constituent snapshots."""
        return self.local_database_root / "ci_l1_daily.duckdb"

    @property
    def ci_l2_daily_path(self) -> Path:
        """Point-in-time CITIC level-2 industry index and constituent snapshots."""
        return self.local_database_root / "ci_l2_daily.duckdb"

    @property
    def a_share_daily_path(self) -> Path:
        """Local A-share daily OHLCV database used for underlying-stock factors."""
        override = os.environ.get("CB_A_SHARE_DAILY_PATH")
        if override:
            return Path(override).expanduser().resolve()

        daily_root = self.local_database_root / LEGACY_A_SHARE_DAILY_DIRECTORY
        candidates = sorted(daily_root.glob("daily_adj_*.duckdb"))
        if candidates:
            return candidates[-1].resolve()
        return daily_root / "daily_adj_19901219_20260722.duckdb"

    def intraday_db_path(self, year: int, frequency: str = "1m") -> Path:
        """Return the annual DuckDB path, handling source naming changes."""
        if frequency not in SUPPORTED_INTRADAY_FREQUENCIES:
            supported = ", ".join(SUPPORTED_INTRADAY_FREQUENCIES)
            raise ValueError(f"Unsupported frequency {frequency!r}; choose from {supported}")
        if year < 2016 or year > 2026:
            raise ValueError("Local intraday data currently supports years 2016 through 2026")

        year_root = self.intraday_root / str(year)
        if year <= 2024:
            return year_root / f"{year}_{frequency}_7z.duckdb"
        if year == 2025:
            names = {
                "1m": "1min_7z.duckdb",
                "5m": "5min_7z.duckdb",
                "15m": "15min_7z.duckdb",
                "30m": "30min_7z.duckdb",
                "60m": "60min_7z.duckdb",
                "day": LEGACY_2025_DAILY_FILENAME,
            }
            return year_root / names[frequency]
        return year_root / f"2026_{frequency}_zip.duckdb"

    def source_manifest(self, start_year: int = 2018, end_year: int = 2026) -> list[dict]:
        """Describe all external inputs used by the first research phase."""
        sources = [
            ("cb_basic", "csv", self.cb_basic_path, True),
            ("cb_daily", "csv", self.cb_daily_path, True),
            ("stock_daily_legacy", "csv", self.stock_daily_path, False),
            ("industry_membership", "csv", self.industry_membership_path, True),
            ("industry_index_daily", "csv", self.industry_index_daily_path, False),
            ("legacy_feature_panel_reference", "csv", self.legacy_feature_panel_path, False),
            ("cb_terms", "duckdb", self.cb_terms_path, True),
            ("redemption_events", "duckdb", self.redemption_path, False),
            ("ci_l1_daily", "duckdb", self.ci_l1_daily_path, True),
            ("ci_l2_daily", "duckdb", self.ci_l2_daily_path, False),
            ("a_share_daily", "duckdb", self.a_share_daily_path, True),
        ]
        for year in range(start_year, end_year + 1):
            sources.append(
                (
                    f"cb_intraday_1m_{year}",
                    "duckdb",
                    self.intraday_db_path(year, "1m"),
                    True,
                )
            )
        return [
            {
                "source_id": source_id,
                "source_type": source_type,
                "path": str(path),
                "required": required,
                "exists": path.exists(),
                "size_mb": path.stat().st_size / 1024**2 if path.exists() else None,
            }
            for source_id, source_type, path, required in sources
        ]

    def validate_required_sources(self, start_year: int = 2018, end_year: int = 2026) -> None:
        missing = [
            item["path"]
            for item in self.source_manifest(start_year, end_year)
            if item["required"] and not item["exists"]
        ]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Required data sources are missing:\n{formatted}")
