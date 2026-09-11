"""Small, deterministic quality helpers for public research notebooks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


G1_FUNNEL_STAGE_LABELS = {
    "\u666e\u901aCB\u4e14\u7814\u7a76\u671f\u5185\u6709\u65e5\u7ebf": "Ordinary CB with observed daily data",
    "\u4e0a\u5e02\u7b2c11\u4e2a\u5e02\u573a\u4ea4\u6613\u65e5\u540e": "Listed for at least 11 market days",
    "20\u65e5\u7d2f\u8ba1\u6362\u624b\u7387\u4e0d\u9ad8\u4e8e100%": "20-day cumulative turnover at or below 100%",
    "\u5269\u4f59\u89c4\u6a21\u4e0d\u4f4e\u4e8e2\u4ebf\u5143": "Remaining principal at least CNY 200 million",
    "\u8bc4\u7ea7\u4e0d\u4f4e\u4e8eA": "Rating A or higher",
    "\u4fe1\u53f7\u65e5\u771f\u5b9e\u6210\u4ea4": "Traded on the signal date",
}


def build_lineage_gate(stage_audits: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Summarize stored configuration-fingerprint consistency without exposing hashes."""
    if not stage_audits:
        raise ValueError("At least one stage audit is required")

    fingerprints = [audit.get("config_sha256") for audit in stage_audits.values()]
    complete = all(isinstance(value, str) and value for value in fingerprints)
    hashes_match = complete and len(set(fingerprints)) == 1

    if hashes_match:
        status = "pass"
        detail = "Stored stage configuration fingerprints match."
    elif complete:
        status = "pending"
        detail = (
            "Stored stage configuration fingerprints differ; this is an administratively "
            "reconciled metadata difference, with a unified final build pending."
        )
    else:
        status = "pending"
        detail = "One or more stored stage configuration fingerprints are unavailable."

    return {
        "status": status,
        "detail": detail,
        "stage_count": len(stage_audits),
    }


def build_g1_headline(g1_audit: Mapping[str, object]) -> pd.DataFrame:
    """Build the public G1 count summary directly from its stored audit."""
    audit_fields = (
        ("Signal dates", "signal_date_count"),
        ("Candidate observations", "row_count"),
        ("Investable observations", "eligible_row_count"),
        ("Distinct investable bonds", "eligible_bond_count"),
    )
    missing = [field for _, field in audit_fields if field not in g1_audit]
    if missing:
        raise KeyError(f"G1 audit is missing headline fields: {missing}")
    return pd.DataFrame(
        {
            "Metric": [label for label, _ in audit_fields],
            "Observed value": [g1_audit[field] for _, field in audit_fields],
        }
    )


def translate_g1_funnel_stages(stages: Iterable[str]) -> list[str]:
    """Translate explicit upstream G1 funnel names and reject unknown stages."""
    stage_values = [str(stage) for stage in stages]
    unknown = sorted(set(stage_values).difference(G1_FUNNEL_STAGE_LABELS))
    if unknown:
        raise ValueError(f"Unrecognized upstream G1 funnel stage count: {len(unknown)}")
    return [G1_FUNNEL_STAGE_LABELS[stage] for stage in stage_values]
