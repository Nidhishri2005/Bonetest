"""Training script for all bone age models.

Usage:
    cd backend
    python -m ml.train --model-type cnn --data-dir /path/to/rsna --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import MODEL_REGISTRY, SUPPORTED_MODELS
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from ml.augmentation import TrainAugmentation
from ml.dataset import create_dataloaders
from ml.preprocessing import compute_dataset_statistics, save_normalization_stats


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bone age regression models")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "cnn_dnn", "multimodal_cnn", "cnn_rf"],
        required=True,
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet50"],
        default="resnet18",
        help="Backbone architecture",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to RSNA dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Epochs to train head with frozen backbone")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--loss",
        choices=["smooth_l1", "l1", "mse"],
        default="smooth_l1",
        help="Loss function",
    )
    parser.add_argument(
        "--normalize-target",
        action="store_true",
        default=True,
        help="Standardize target to N(0,1) during training",
    )
    parser.add_argument(
        "--no-normalize-target",
        action="store_false",
        dest="normalize_target",
        help="Train directly on raw months",
    )
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-dir", type=Path, default=BACKEND_ROOT / "checkpoints")
    parser.add_argument("--metrics-dir", type=Path, default=BACKEND_ROOT / "metrics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained CNN checkpoint for downstream heads",
    )
    return parser.parse_args()


def get_loss_function(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss()


def train_epoch(model, loader, optimizer, criterion, device, model_type, scaler):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            if model_type == "multimodal_cnn":
                gender = batch["male"].unsqueeze(-1).to(device)
                outputs = model(images, gender)
            else:
                outputs = model(images)

            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Batch {batch_idx + 1}/{num_batches} -- loss: {loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, model_type, target_mean=None, target_std=None):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    for batch in loader:
        images = batch["image"].to(device)
        ground_truth = batch["bone_age"].to(device)
        targets_loss = batch["target"].to(device)

        if model_type == "multimodal_cnn":
            gender = batch["male"].unsqueeze(-1).to(device)
            outputs = model(images, gender)
        else:
            outputs = model(images)

        loss = criterion(outputs, targets_loss)
        total_loss += loss.item() * images.size(0)

        # INVERSE TRANSFORM PREDICTIONS TO MONTHS BEFORE CALCULATING CLINICAL METRICS
        if target_mean is not None and target_std is not None:
            pred_months = outputs * target_std + target_mean
        else:
            pred_months = outputs

        preds.extend(pred_months.cpu().numpy().tolist())
        targets.extend(ground_truth.cpu().numpy().tolist())

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    rmse = float(np.sqrt(mse))

    ss_tot = float(np.sum((targets_arr - np.mean(targets_arr)) ** 2))
    ss_res = float(np.sum((targets_arr - preds_arr) ** 2))
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))

    return total_loss / len(loader.dataset), mae, mse, rmse, r2


def log_experiment(metrics_dir: Path, record: dict) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_file = metrics_dir / "experiments_log.csv"
    file_exists = log_file.exists()

    fieldnames = [
        "timestamp",
        "model_type",
        "backbone",
        "loss",
        "lr",
        "batch_size",
        "epochs_trained",
        "best_epoch",
        "target_normalized",
        "val_mae_months",
        "val_rmse_months",
        "val_r2",
        "checkpoint_path",
    ]

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"Training Model: {args.model_type}")
    print(f"Backbone:       {args.backbone}")
    print(f"Device:         {device}")
    print(f"Normalize tgt:  {args.normalize_target}")
    print(f"Pretrained:     {args.pretrained}")
    print(f"Loss function:  {args.loss}")
    print("=" * 60)

    aug = TrainAugmentation()
    train_loader, val_loader, train_df, val_df, target_mean, target_std = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_transform=aug,
        normalize_target=args.normalize_target,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    if args.normalize_target:
        print(f"Target Normalization -- Mean: {target_mean:.2f} mo, Std: {target_std:.2f} mo")
        stats_path = args.checkpoints_dir / "normalization_stats.json"
        save_normalization_stats(
            {
                "mean": 0.4523,
                "std": 0.2118,
                "target_mean": target_mean,
                "target_std": target_std,
            },
            stats_path,
        )

    # -------------------------------------------------------------
    # RANDOM FOREST SPECIAL PIPELINE
    # -------------------------------------------------------------
    if args.model_type == "cnn_rf":
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        print("\n--- Training Random Forest Regressor ---")
        extractor = CNNFeatureExtractor(pretrained=args.pretrained, backbone_name=args.backbone).to(device)

        if args.cnn_checkpoint and Path(args.cnn_checkpoint).exists():
            print(f"Loading CNN weights from {args.cnn_checkpoint}")
            ckpt = torch.load(args.cnn_checkpoint, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
            if backbone_state:
                extractor.backbone.load_state_dict(backbone_state, strict=False)

        extractor.eval()

        def extract_features(loader):
            feats, ys = [], []
            with torch.no_grad():
                for batch in loader:
                    imgs = batch["image"].to(device)
                    f = extractor(imgs).cpu().numpy()
                    feats.append(f)
                    ys.append(batch["target"].numpy())
            return np.vstack(feats), np.concatenate(ys)

        print("Extracting training features...")
        X_train, y_train = extract_features(train_loader)
        print("Extracting validation features...")
        X_val, y_val = extract_features(val_loader)

        rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=args.seed)
        rf.fit(X_train, y_train)

        val_preds_norm = rf.predict(X_val)
        if args.normalize_target and target_mean and target_std:
            val_preds_mo = val_preds_norm * target_std + target_mean
            val_true_mo = y_val * target_std + target_mean
        else:
            val_preds_mo = val_preds_norm
            val_true_mo = y_val

        mae = float(np.mean(np.abs(val_preds_mo - val_true_mo)))
        rmse = float(np.sqrt(np.mean((val_preds_mo - val_true_mo) ** 2)))
        ss_tot = float(np.sum((val_true_mo - np.mean(val_true_mo)) ** 2))
        r2 = float(1.0 - np.sum((val_true_mo - val_preds_mo) ** 2) / max(ss_tot, 1e-8))

        print(f"\n[CNN_RF Results] Val MAE: {mae:.2f} mo | Val RMSE: {rmse:.2f} mo | Val R2: {r2:.4f}")

        args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.checkpoints_dir / "cnn_rf_best.pth"
        rf_path = args.checkpoints_dir / "cnn_rf_sklearn.joblib"
        joblib.dump(rf, rf_path)
        torch.save({
            "model_type": "cnn_rf",
            "backbone": args.backbone,
            "target_mean": target_mean,
            "target_std": target_std,
            "val_mae": mae,
            "val_rmse": rmse,
            "val_r2": r2,
            "rf_path": str(rf_path.name),
        }, ckpt_path)

        log_experiment(args.metrics_dir, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": "cnn_rf",
            "backbone": args.backbone,
            "loss": "rf_mse",
            "lr": 0.0,
            "batch_size": args.batch_size,
            "epochs_trained": 100,
            "best_epoch": 1,
            "target_normalized": args.normalize_target,
            "val_mae_months": mae,
            "val_rmse_months": rmse,
            "val_r2": r2,
            "checkpoint_path": str(ckpt_path),
        })
        return

    # -------------------------------------------------------------
    # DEEP LEARNING MODELS (cnn, cnn_dnn, multimodal_cnn)
    # -------------------------------------------------------------
    model_cls = MODEL_REGISTRY[args.model_type]["class"]
    if args.model_type in ("cnn", "cnn_dnn", "multimodal_cnn"):
        model = model_cls(pretrained=args.pretrained, backbone_name=args.backbone).to(device)

    criterion = get_loss_function(args.loss)
    scaler = GradScaler(enabled=device.type == "cuda")

    # -------------------------------------------------------------
    # TWO-PHASE TRAINING SCHEDULE
    # -------------------------------------------------------------
    # Phase 1: Warmup head with frozen backbone
    warmup_epochs = min(args.warmup_epochs, args.epochs // 4)
    backbone_module = getattr(model, "backbone", getattr(model, "image_backbone", None))

    if warmup_epochs > 0 and backbone_module is not None:
        print(f"\n--- Phase 1: Warming up head for {warmup_epochs} epochs (backbone frozen) ---")
        for param in backbone_module.parameters():
            param.requires_grad = False

        warmup_opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        for ep in range(1, warmup_epochs + 1):
            t_loss = train_epoch(model, train_loader, warmup_opt, criterion, device, args.model_type, scaler)
            v_loss, v_mae, v_mse, v_rmse, v_r2 = validate(
                model, val_loader, criterion, device, args.model_type, target_mean, target_std
            )
            print(f"Warmup [{ep}/{warmup_epochs}] Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val MAE: {v_mae:.2f} mo | Val R2: {v_r2:.4f}")

    # Phase 2: Full fine-tuning with differential learning rates
    print("\n--- Phase 2: Full End-to-End Fine-Tuning (differential LR) ---")
    if backbone_module is not None:
        for param in backbone_module.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": backbone_module.parameters(), "lr": args.lr * 0.1},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone") and not n.startswith("image_backbone")], "lr": args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.checkpoints_dir / f"{args.model_type}_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.model_type, scaler)
        val_loss, val_mae, val_mse, val_rmse, val_r2 = validate(
            model, val_loader, criterion, device, args.model_type, target_mean, target_std
        )
        scheduler.step(val_mae)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) -- "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.2f} mo | Val RMSE: {val_rmse:.2f} mo | Val R2: {val_r2:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_type": args.model_type,
                "backbone": args.backbone,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "target_mean": target_mean,
                "target_std": target_std,
            }, best_ckpt_path)
            print(f"  * Saved new best checkpoint with Val MAE = {val_mae:.2f} months")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping at epoch {epoch}.")
                break

    print("\n" + "=" * 60)
    print(f"Training Complete for {args.model_type}!")
    print(f"Best Val MAE: {best_val_mae:.2f} months at epoch {best_epoch}")
    print(f"Best Checkpoint: {best_ckpt_path}")
    print("=" * 60)

    log_experiment(args.metrics_dir, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": args.model_type,
        "backbone": args.backbone,
        "loss": args.loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "target_normalized": args.normalize_target,
        "val_mae_months": best_val_mae,
        "val_rmse_months": val_rmse,
        "val_r2": val_r2,
        "checkpoint_path": str(best_ckpt_path),
    })


if __name__ == "__main__":
    main()
