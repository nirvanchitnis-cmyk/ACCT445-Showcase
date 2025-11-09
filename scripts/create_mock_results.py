"""Create mock result files so the dashboard has data to display locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)


def create_mock_results(results_dir: Path, force: bool = False) -> None:
    """Create lightweight CSVs mirroring production outputs."""
    results_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "decile_summary_latest.csv": pd.DataFrame(
            {
                "decile": list(range(1, 11)),
                "n_obs": [120] * 10,
                "mean_ret": np.linspace(-0.01, 0.02, 10),
                "std_ret": np.linspace(0.04, 0.06, 10),
                "t_stat": np.linspace(-1.0, 2.5, 10),
            }
        ),
        "decile_long_short_latest.csv": pd.DataFrame(
            {
                "period": pd.date_range("2023-01-31", periods=24, freq="ME"),
                "ret": np.random.normal(0.004, 0.015, 24),
            }
        ),
        "cnoi_with_tickers.csv": pd.DataFrame(
            {
                "cik": range(1000, 1040),
                "ticker": [f"BANK{i}" for i in range(40)],
                "company_name": [f"Bank {i}" for i in range(40)],
                "CNOI": np.random.uniform(10, 90, 40),
            }
        ),
        "event_study_results.csv": pd.DataFrame(
            {
                "cnoi_quartile": ["Q1", "Q2", "Q3", "Q4"],
                "CAR_mean": np.random.uniform(-0.12, -0.03, 4),
                "CAR_std": np.random.uniform(0.03, 0.09, 4),
                "CAR_count": [8] * 4,
            }
        ),
        "event_study_daily_cum_ar.csv": pd.DataFrame(
            {
                "date": pd.date_range("2023-03-01", periods=21, freq="D"),
                "cnoi_quartile": ["All"] * 21,
                "abnormal_return": np.random.normal(-0.002, 0.01, 21),
                "cum_ar": np.linspace(-0.08, 0.01, 21),
            }
        ),
    }

    for filename, df in fixtures.items():
        path = results_dir / filename
        if path.exists() and not force:
            print(f"[SKIP] {filename} already exists (use --force to overwrite).")
            continue
        df.to_csv(path, index=False)
        print(f"[OK] Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where mock result files should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_mock_results(Path(args.results_dir), force=args.force)
