"""Evaluate trained models on validation set.

Usage:
    cd backend
    python -m ml.evaluate --data-dir /path/to/rsna --checkpoints-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import r2_score

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from ml.dataset import create_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bone age models")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=BACKEND_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=BACKEND_ROOT / "metrics"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, males: np.ndarray | None = None) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mae_years = float(mae / 12.0)

    if len(y_true) > 1 and np.std(y_pred) > 1e-6:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0

    metrics = {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "pearson": round(pearson, 4),
        "mae_years": round(mae_years, 2),
    }

    if males is not None:
        male_mask = males > 0.5
        female_mask = ~male_mask
        if np.any(male_mask):
            metrics["mae_male"] = round(float(np.mean(np.abs(y_true[male_mask] - y_pred[male_mask]))), 2)
        if np.any(female_mask):
            metrics["mae_female"] = round(float(np.mean(np.abs(y_true[female_mask] - y_pred[female_mask]))), 2)

    return metrics


@torch.no_grad()
def evaluate_model(model_type: str, loader, device, checkpoints_dir: Path):
    meta = MODEL_REGISTRY[model_type]
    ckpt_pth = checkpoints_dir / f"{model_type}_best.pth"
    ckpt_pt = checkpoints_dir / f"{model_type}_best.pt"
    ckpt_path = ckpt_pth if ckpt_pth.exists() else ckpt_pt

    if not ckpt_path.exists():
        print(f"Skipping {model_type}: checkpoint not found at {ckpt_path}")
        return None, None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = checkpoint.get("backbone", "resnet18")
    target_mean = checkpoint.get("target_mean", None)
    target_std = checkpoint.get("target_std", None)

    if model_type == "cnn_rf":
        cnn = CNNFeatureExtractor(pretrained=False, backbone_name=backbone)
        cnn.load_state_dict(checkpoint["model_state_dict"], strict=False)
        cnn.to(device).eval()

        rf_candidates = [
            checkpoints_dir / "cnn_rf_sklearn.joblib",
            BACKEND_ROOT / "rf_models" / "cnn_rf.joblib",
        ]
        rf_path = None
        for p in rf_candidates:
            if p.exists():
                rf_path = p
                break

        if rf_path is None:
            print(f"Skipping {model_type}: RF joblib not found")
            return None, None, None, None

        rf = joblib.load(rf_path)
        wrapper = CNNWithRFWrapper(cnn, rf, target_mean=target_mean, target_std=target_std)
        preds, targets, ids, males = [], [], [], []

        for batch in loader:
            imgs = batch["image"].to(device)
            feats = cnn(imgs).cpu().numpy()
            for i in range(feats.shape[0]):
                pred_val = wrapper.predict_numpy(feats[i])
                preds.append(pred_val)
            targets.extend(batch["bone_age"].numpy().tolist())
            ids.extend(batch["id"])
            males.extend(batch["male"].numpy().tolist())

        return np.array(targets), np.array(preds), ids, np.array(males)

    model_cls = meta["class"]
    try:
        model = model_cls(pretrained=False, backbone_name=backbone)
    except TypeError:
        model = model_cls(pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()

    preds, targets, ids, males = [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        if model_type == "multimodal_cnn":
            gender = batch["male"].unsqueeze(-1).to(device)
            out = model(imgs, gender)
        else:
            out = model(imgs)

        if target_mean is not None and target_std is not None:
            out_months = out * target_std + target_mean
        else:
            out_months = out

        preds.extend(out_months.cpu().numpy().tolist())
        targets.extend(batch["bone_age"].numpy().tolist())
        ids.extend(batch["id"])
        males.extend(batch["male"].numpy().tolist())

    return np.array(targets), np.array(preds), ids, np.array(males)


def plot_comparison(results: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}
    sns.set_theme(style="whitegrid")

    models = list(results.keys())
    metrics = ["mae", "rmse", "r2"]
    labels = ["MAE (months)", "RMSE (months)", "R² Score"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, label in zip(axes, metrics, labels):
        values = [results[m]["metrics"][metric] for m in models]
        names = [results[m]["display_name"] for m in models]
        palette = sns.color_palette("viridis", len(models))
        bars = ax.bar(names, values, color=palette)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, values):
            fmt = f"{val:.2f}" if metric != "r2" else f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.92 if val < 0 else bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    comparison_path = output_dir / "metrics_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["metrics_comparison"] = str(comparison_path)

    # Scatter Comparison
    fig, ax = plt.subplots(figsize=(7, 7))
    palette = sns.color_palette("deep", len(models))
    for idx, (model_type, data) in enumerate(results.items()):
        df = pd.read_csv(data["predictions_csv"])
        ax.scatter(
            df["actual"],
            df["predicted"],
            alpha=0.35,
            s=14,
            label=f"{data['display_name']} (MAE: {data['metrics']['mae']:.1f}m)",
            color=palette[idx],
        )
    lims = [0, 240]
    ax.plot(lims, lims, "k--", alpha=0.7, linewidth=1.5, label="Perfect agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Bone Age (months)", fontsize=11)
    ax.set_ylabel("Predicted Bone Age (months)", fontsize=11)
    ax.set_title("Predicted vs Actual Bone Age", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    scatter_path = output_dir / "scatter_comparison.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["scatter_comparison"] = str(scatter_path)

    return plot_paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Evaluating Bone Age Models on Validation Set")
    print("==================================================")

    _, val_loader, _, val_df, _, _ = create_dataloaders(
        args.data_dir,
        batch_size=32,
        normalize_target=False,
    )

    all_results = {}
    for model_type in SUPPORTED_MODELS:
        print(f"\nEvaluating {model_type}...")
        y_true, y_pred, ids, males = evaluate_model(
            model_type, val_loader, device, args.checkpoints_dir
        )
        if y_true is None:
            continue

        metrics = compute_metrics(y_true, y_pred, males)
        meta = MODEL_REGISTRY[model_type]
        pred_csv = args.metrics_dir / f"predictions_{model_type}.csv"
        pd.DataFrame(
            {
                "id": ids,
                "actual": y_true,
                "predicted": y_pred,
                "male": males,
                "error": np.abs(y_true - y_pred),
            }
        ).to_csv(pred_csv, index=False)

        all_results[model_type] = {
            "model_type": model_type,
            "display_name": meta["display_name"],
            "metrics": metrics,
            "num_samples": len(y_true),
            "predictions_csv": str(pred_csv),
        }
        print(
            f"  MAE: {metrics['mae']:.2f} mo | RMSE: {metrics['rmse']:.2f} mo | "
            f"R²: {metrics['r2']:.4f} | Pearson: {metrics['pearson']:.4f}"
        )
        if "mae_male" in metrics:
            print(f"  Male MAE: {metrics['mae_male']:.2f} mo | Female MAE: {metrics['mae_female']:.2f} mo")

    if not all_results:
        print("No checkpoints found to evaluate.")
        return

    best = min(all_results.items(), key=lambda x: x[1]["metrics"]["mae"])
    plots_dir = args.metrics_dir / "plots"
    plot_paths = plot_comparison(all_results, plots_dir)

    output = {
        "comparison": {
            "models": [
                {
                    "model_type": k,
                    "display_name": v["display_name"],
                    "metrics": v["metrics"],
                    "num_samples": v["num_samples"],
                }
                for k, v in all_results.items()
            ],
            "best_model": best[0],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "plots": {k: str(v) for k, v in plot_paths.items()},
        "details": {
            "dataset": "RSNA Pediatric Bone Age",
            "validation_split": 0.15,
            "image_size": 512,
        },
    }

    out_path = args.metrics_dir / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved authentic evaluation results to {out_path}")


if __name__ == "__main__":
    main()
