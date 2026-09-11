"""List explicitly registered production factor plugins."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from cb_quant.factors import build_default_factor_registry


def build_factor_inventory() -> list[dict[str, object]]:
    """Return stable metadata for the approved production catalog."""
    registry = build_default_factor_registry()
    inventory: list[dict[str, object]] = []
    for name in registry.names():
        spec = registry.get_spec(name)
        inventory.append(
            {
                "name": spec.name,
                "family": spec.family,
                "direction": spec.direction,
                "default_parameters": dict(spec.default_parameters),
                "required_fields": list(spec.required_fields),
                "required_history_fields": {
                    table: list(columns)
                    for table, columns in spec.required_history_fields.items()
                },
                "neutralizers": list(spec.neutralizers),
                "version": spec.version,
            }
        )
    return inventory


def _format_table(inventory: list[dict[str, object]]) -> str:
    headers = ("name", "family", "direction", "parameters", "version")
    rows = [
        (
            str(item["name"]),
            str(item["family"]),
            str(item["direction"]),
            json.dumps(
                item["default_parameters"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            str(item["version"]),
        )
        for item in inventory
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()

    separator = tuple("-" * width for width in widths)
    return "\n".join(
        [format_row(headers), format_row(separator), *(format_row(row) for row in rows)]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json", "table"), default="json")
    args = parser.parse_args(argv)
    inventory = build_factor_inventory()
    if args.format == "table":
        print(_format_table(inventory))
    else:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
