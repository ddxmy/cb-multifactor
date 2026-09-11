"""Resumable, point-in-time downloads for convertible-bond event histories."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import duckdb


class TushareConvertibleBondClient(Protocol):
    """Subset of the Tushare client used by the historical event downloader."""

    def cb_rating(self, *, ts_code: str) -> pd.DataFrame: ...

    def cb_share(self, *, start_date: str, end_date: str) -> pd.DataFrame: ...


RATING_COLUMNS = [
    "ts_code",
    "ann_date",
    "rating_date",
    "rating_com_name",
    "rating_way",
    "rating_type",
    "rating",
    "rating_outlook",
]
SHARE_COLUMNS = [
    "ts_code",
    "bond_short_name",
    "publish_date",
    "end_date",
    "issue_size",
    "convert_price_initial",
    "convert_price",
    "convert_val",
    "convert_vol",
    "convert_ratio",
    "acc_convert_val",
    "acc_convert_vol",
    "acc_convert_ratio",
    "remain_size",
    "total_shares",
]


@dataclass(frozen=True)
class DownloadSummary:
    """Audit information returned after a resumable download."""

    requested: int
    completed: int
    fetched_now: int
    failed: int
    output_path: Path


def create_tushare_client(token: str | None = None) -> TushareConvertibleBondClient:
    """Create a Tushare Pro client without ever persisting the token."""
    resolved_token = token or os.environ.get("TUSHARE_TOKEN")
    if not resolved_token:
        raise EnvironmentError("Set TUSHARE_TOKEN before downloading Tushare history")

    try:
        import tushare as ts
    except ImportError as error:
        raise ImportError("Install the optional Tushare dependency before downloading history") from error
    return ts.pro_api(resolved_token)


def load_cb_codes(cb_basic_path: Path) -> list[str]:
    """Load ordinary convertible-bond codes, excluding exchangeable bonds."""
    basic = pd.read_csv(cb_basic_path, usecols=["ts_code", "cb_type"], low_memory=False)
    return sorted(
        basic.loc[basic["cb_type"].eq("CB"), "ts_code"].dropna().astype(str).unique().tolist()
    )


def load_cb_codes_from_terms(cb_terms_path: Path) -> list[str]:
    """Load the full historical ordinary-CB universe from the local terms database."""
    if not cb_terms_path.exists():
        raise FileNotFoundError(f"CB terms database does not exist: {cb_terms_path}")
    with duckdb.connect(str(cb_terms_path), read_only=True) as connection:
        codes = connection.execute(
            """
            SELECT DISTINCT ts_code
            FROM default_table
            WHERE cb_type = 'CB' AND ts_code IS NOT NULL
            ORDER BY ts_code
            """
        ).fetchdf()
    return codes["ts_code"].astype(str).tolist()


def month_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split a date range into calendar-month API windows, inclusive at both ends."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must not be earlier than start_date")

    windows = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        window_start = max(start, period.start_time)
        window_end = min(end, period.end_time.normalize())
        windows.append((window_start.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
    return windows


def download_cb_rating_history(
    client: TushareConvertibleBondClient,
    ts_codes: Iterable[str],
    output_root: Path,
    *,
    throttle_seconds: float = 0.25,
    checkpoint_every: int = 25,
) -> DownloadSummary:
    """Download rating histories once per bond and resume safely after interruption.

    ``cb_rating`` requires ``ts_code``. A separate completion manifest records
    successful empty responses too, so bonds without rating history are not
    queried repeatedly on every rerun.
    """
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if throttle_seconds < 0:
        raise ValueError("throttle_seconds must be non-negative")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "cb_rating_history.parquet"
    completed_path = output_root / "cb_rating_completed_codes.csv"
    requested_codes = sorted(set(map(str, ts_codes)))

    existing = _read_parquet_or_empty(output_path, RATING_COLUMNS)
    completed_codes = _read_completed_codes(completed_path)
    pending_codes = [code for code in requested_codes if code not in completed_codes]
    fetched_frames: list[pd.DataFrame] = []
    successful_codes: list[str] = []
    failures: list[dict[str, str]] = []
    consecutive_failures = 0

    for index, ts_code in enumerate(pending_codes, start=1):
        try:
            frame = client.cb_rating(ts_code=ts_code)
            fetched_frames.append(_normalize_columns(frame, RATING_COLUMNS))
            successful_codes.append(ts_code)
            consecutive_failures = 0
        except Exception as error:  # Network and entitlement errors must remain visible to the caller.
            failures.append({"ts_code": ts_code, "error": str(error)})
            consecutive_failures += 1
            if consecutive_failures >= 5:
                _persist_rating_checkpoint(
                    existing, fetched_frames, completed_codes | set(successful_codes), output_path, completed_path
                )
                raise RuntimeError("Stopped after five consecutive cb_rating failures") from error

        if index % checkpoint_every == 0:
            _persist_rating_checkpoint(
                existing, fetched_frames, completed_codes | set(successful_codes), output_path, completed_path
            )
        if throttle_seconds:
            time.sleep(throttle_seconds)

    _persist_rating_checkpoint(
        existing, fetched_frames, completed_codes | set(successful_codes), output_path, completed_path
    )
    _write_failures(output_root / "cb_rating_failed_requests.csv", failures)
    return DownloadSummary(
        requested=len(requested_codes),
        completed=len(completed_codes | set(successful_codes)),
        fetched_now=len(successful_codes),
        failed=len(failures),
        output_path=output_path,
    )


def download_cb_share_history(
    client: TushareConvertibleBondClient,
    output_root: Path,
    *,
    start_date: str,
    end_date: str,
    throttle_seconds: float = 0.25,
    checkpoint_every: int = 20,
) -> DownloadSummary:
    """Download published conversion/remaining-size events in resumable monthly batches."""
    if throttle_seconds < 0:
        raise ValueError("throttle_seconds must be non-negative")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "cb_share_history.parquet"
    completed_path = output_root / "cb_share_completed_windows.csv"
    existing = _read_parquet_or_empty(output_path, SHARE_COLUMNS)
    completed_windows = _read_completed_windows(completed_path)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    windows = month_windows(start_date, end_date)
    pending_windows = [window for window in windows if window not in completed_windows]
    successful_windows: list[tuple[str, str]] = []

    for index, (window_start, window_end) in enumerate(pending_windows, start=1):
        try:
            frames.append(
                _normalize_columns(
                    client.cb_share(start_date=window_start, end_date=window_end), SHARE_COLUMNS
                )
            )
            successful_windows.append((window_start, window_end))
        except Exception as error:
            failures.append({"start_date": window_start, "end_date": window_end, "error": str(error)})
        if index % checkpoint_every == 0:
            _persist_share_checkpoint(
                existing,
                frames,
                completed_windows | set(successful_windows),
                output_path,
                completed_path,
            )
        if throttle_seconds:
            time.sleep(throttle_seconds)

    _persist_share_checkpoint(
        existing,
        frames,
        completed_windows | set(successful_windows),
        output_path,
        completed_path,
    )
    if failures:
        _write_failures(output_root / "cb_share_failed_requests.csv", failures)
        raise RuntimeError(f"cb_share failed for {len(failures)} monthly windows; completed windows were saved")

    return DownloadSummary(
        requested=len(windows),
        completed=len(completed_windows | set(successful_windows)),
        fetched_now=len(successful_windows),
        failed=0,
        output_path=output_path,
    )


def _normalize_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns).difference(frame.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(f"Tushare response is missing expected columns: {formatted}")
    normalized = frame.loc[:, columns].copy()
    for column in ("ann_date", "rating_date", "publish_date", "end_date"):
        if column in normalized:
            normalized[column] = normalized[column].astype("string")
    return normalized


def _read_parquet_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)


def _read_completed_codes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, dtype={"ts_code": "string"})["ts_code"].dropna().astype(str))


def _read_completed_windows(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed = pd.read_csv(path, dtype={"start_date": "string", "end_date": "string"})
    return set(zip(completed["start_date"].astype(str), completed["end_date"].astype(str)))


def _deduplicate_events(
    frame: pd.DataFrame,
    columns: list[str],
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.drop_duplicates().sort_values(sort_columns or columns[:2], ignore_index=True)


def _persist_rating_checkpoint(
    existing: pd.DataFrame,
    fetched_frames: list[pd.DataFrame],
    completed_codes: set[str],
    output_path: Path,
    completed_path: Path,
) -> None:
    combined = _deduplicate_events(pd.concat([existing, *fetched_frames], ignore_index=True), RATING_COLUMNS)
    combined.to_parquet(output_path, index=False)
    pd.DataFrame({"ts_code": sorted(completed_codes)}).to_csv(completed_path, index=False)


def _persist_share_checkpoint(
    existing: pd.DataFrame,
    fetched_frames: list[pd.DataFrame],
    completed_windows: set[tuple[str, str]],
    output_path: Path,
    completed_path: Path,
) -> None:
    combined = _deduplicate_events(
        pd.concat([existing, *fetched_frames], ignore_index=True),
        SHARE_COLUMNS,
        sort_columns=["ts_code", "publish_date", "end_date"],
    )
    combined.to_parquet(output_path, index=False)
    pd.DataFrame(sorted(completed_windows), columns=["start_date", "end_date"]).to_csv(
        completed_path, index=False
    )


def _write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    if failures:
        pd.DataFrame(failures).to_csv(path, index=False, encoding="utf-8-sig")
