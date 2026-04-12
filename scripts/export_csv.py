#!/usr/bin/env python3
"""ASPI dataset CSV export CLI.

Thin wrapper around ``aspi.dataset.export.main``. Exports a JSONL dataset
to a flat CSV for human review.

Usage:
    python scripts/export_csv.py --input data/aspi_*_25pairs.jsonl \\
        --output review.csv
"""

from aspi.dataset.export import main

if __name__ == "__main__":
    main()
