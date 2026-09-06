"""Generate comparison tables and summary report from evaluation results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = BACKEND_ROOT / "metrics" / "evaluation_results.json"
REPORT_PATH = BACKEND_ROOT / "metrics" / "comparison_report.csv"


def main() -> None:
    if not METRICS_PATH.exists():
        print(f"No evaluation results at {METRICS_PATH}. Run ml.evaluate first.")
        sys.exit(1)

    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for model in data["comparison"]["models"]:
        m = model["metrics"]
        rows.append(
            {
                "Model": model["display_name"],
                "MAE (months)": round(model["metrics"]["mae"], 2),
                "MSE": round(model["metrics"]["mse"], 2),
                "RMSE (months)": round(model["metrics"]["rmse"], 2),
                "MAE (months)": round(m["mae"], 2),
                "RMSE (months)": round(m["rmse"], 2),
                "R²": round(m.get("r2", 0.0), 4),
                "Pearson r": round(m.get("pearson", 0.0), 4),
                "MAE (years)": round(m.get("mae_years", m["mae"] / 12.0), 2),
                "Samples": model["num_samples"],
            }
        )

    df = pd.DataFrame(rows).sort_values("MAE (months)")
    df.to_csv(REPORT_PATH, index=False)

    print("\n=== Bone Age Model Comparison ===\n")
    print(df.to_string(index=False))
    print(f"\nBest model (lowest MAE): {data['comparison']['best_model']}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
