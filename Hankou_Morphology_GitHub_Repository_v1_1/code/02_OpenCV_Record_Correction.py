from __future__ import annotations

from pathlib import Path
import argparse
import csv


CORRECTIONS = {
    "F030": {
        "Total window-area ratio": 0.3758,
        "Total door-area ratio": 0.2675,
        "Total column-area ratio": 0.4122,
    },
    "F072": {
        "Total window-area ratio": 0.2142,
        "Total door-area ratio": 0.1517,
        "Total column-area ratio": 0.1953,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        for field, corrected_value in CORRECTIONS.get(
            row["facade_id"], {}
        ).items():
            row[field] = corrected_value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
