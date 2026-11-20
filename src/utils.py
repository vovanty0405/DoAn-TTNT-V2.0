"""Utility helpers for JSON I/O, hashing, and formatting."""

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write JSON preserving Vietnamese characters and NumPy scalar values."""
    Path(path).write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def print_terminal_table(table) -> None:
    """Print an ASCII-bordered DataFrame table safe for Windows terminals."""
    headers = list(table.columns)
    rows = list(table.itertuples(index=False, name=None))
    widths = [
        max([len(header), *(len(str(row[i])) for row in rows)])
        for i, header in enumerate(headers)
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row):
        return (
            "| "
            + " | ".join(str(value).ljust(width) for value, width in zip(row, widths))
            + " |"
        )

    print(border)
    print(format_row(headers))
    print(border)
    for row in rows:
        print(format_row(row))
    print(border)

