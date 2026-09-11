# Contributing

Contributions should preserve the repository's point-in-time research design, deterministic run
identity, and separation between licensed local data and public reproducibility assets.

## Development Setup

```bash
conda create -n cb-multifactor python=3.11 -y
conda activate cb-multifactor
python -m pip install -e ".[dev]"
```

## Change Requirements

- Create a focused branch and keep unrelated refactors out of the change.
- Never commit credentials, local absolute paths, licensed market data, generated research
  panels, or full run artifacts.
- Add synthetic fixtures and tests for behavior changes. Factor plugins must follow the
  `trade_date`, `security_id`, and `raw_value` output contract.
- Preserve close-signal/next-open information timing and document any proposed change to the
  frozen research or execution specification before implementation.
- Do not report validation or holdout performance as replication evidence.

## Verification

Run the same gates used by continuous integration before opening a pull request:

```bash
python -m pytest
python -m compileall -q src scripts
python scripts/validate_research_config.py --check-only
python -m pip check
python -m build
```

Pull requests should explain the economic rationale, information timestamp, expected sign,
coverage implications, tests, and any change to execution assumptions.
