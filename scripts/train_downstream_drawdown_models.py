#!/usr/bin/env python3
"""Train the same downstream risk model on real and flow-augmented datasets.

Financial task
--------------
Predict the maximum 30-day drawdown magnitude of a 60% BTC / 40% ETH
portfolio from the 14 condition features describing the preceding market state.

The architecture, initialization, optimizer, target scaling and real-only
validation/test sets are identical for every augmentation ratio. Only the
training dataset changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_FILES = (
    "train_real_only.npz",
    "train_real_plus_025pct_flow.npz",
    "train_real_plus_050pct_flow.npz",
    "train_real_plus_100pct_flow.npz",
)


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 100
    patience: int = 15
    seed: int = 2026


class DrawdownMLP(nn.Module):
    """Small fixed MLP used identically for all training datasets."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/augmented/normalizing_flow"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/downstream_drawdown_flow"),
    )
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim-1", type=int, default=64)
    parser.add_argument("--hidden-dim-2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--portfolio-value", type=float, default=100_000.0)
    parser.add_argument("--btc-weight", type=float, default=0.60)
    parser.add_argument("--eth-weight", type=float, default=0.40)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if name != "cpu":
        raise ValueError("device must be one of: cpu, mps, cuda")
    return torch.device("cpu")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def maximum_drawdown_magnitude(
    log_return_paths: np.ndarray,
    *,
    btc_weight: float,
    eth_weight: float,
) -> np.ndarray:
    """Return positive maximum-drawdown magnitudes for each path."""
    paths = np.asarray(log_return_paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError(f"Expected [samples, steps, 2], got {paths.shape}")

    cumulative = np.cumsum(paths, axis=1)
    gross = np.exp(cumulative)
    portfolio = btc_weight * gross[:, :, 0] + eth_weight * gross[:, :, 1]
    initial = np.ones((len(paths), 1), dtype=np.float64)
    portfolio = np.concatenate([initial, portfolio], axis=1)
    running_peak = np.maximum.accumulate(portfolio, axis=1)
    drawdown = portfolio / running_peak - 1.0
    return (-drawdown.min(axis=1)).astype(np.float32)


def dataset_xy(
    arrays: dict[str, np.ndarray],
    *,
    btc_weight: float,
    eth_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "condition_features",
        "target_returns_original_scale",
        "source_is_synthetic",
    }
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"Dataset is missing keys: {sorted(missing)}")

    x = arrays["condition_features"].astype(np.float32)
    y = maximum_drawdown_magnitude(
        arrays["target_returns_original_scale"],
        btc_weight=btc_weight,
        eth_weight=eth_weight,
    )
    source = arrays["source_is_synthetic"].astype(np.int8)
    if len(x) != len(y) or len(y) != len(source):
        raise ValueError("Condition, target and source arrays have different lengths")
    return x, y, source


def make_loader(
    x: np.ndarray,
    y_scaled: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y_scaled.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            total_loss += float(loss.item()) * len(x_batch)
            total_count += len(x_batch)
    return total_loss / max(total_count, 1)


def train_one_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    config: TrainConfig,
    device: torch.device,
) -> tuple[DrawdownMLP, list[dict[str, float]], int, float]:
    # Resetting the seed guarantees identical initial weights for every dataset.
    set_global_seed(config.seed)
    model = DrawdownMLP(
        input_dim=train_x.shape[1],
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
        dropout=config.dropout,
    ).to(device)

    train_y_scaled = (train_y - target_mean) / target_std
    validation_y_scaled = (validation_y - target_mean) / target_std

    train_loader = make_loader(
        train_x,
        train_y_scaled,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = make_loader(
        validation_x,
        validation_y_scaled,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(x_batch)
            train_count += len(x_batch)

        train_loss = train_loss_sum / max(train_count, 1)
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": float(epoch),
                "train_mse_scaled": train_loss,
                "validation_mse_scaled": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_mse={train_loss:.6f} "
                f"val_mse={validation_loss:.6f} lr={learning_rate:.2e}"
            )

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_validation


def predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            batch = torch.from_numpy(x[start:stop].astype(np.float32)).to(device)
            scaled = model(batch).detach().cpu().numpy()
            outputs.append(scaled * target_std + target_mean)
    return np.concatenate(outputs).astype(np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
    if y_true.std() == 0 or y_pred.std() == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])

    threshold_90 = float(np.quantile(y_true, 0.90))
    tail_mask_90 = y_true >= threshold_90
    tail_mae_90 = float(np.mean(np.abs(residual[tail_mask_90])))

    threshold_95 = float(np.quantile(y_true, 0.95))
    tail_mask_95 = y_true >= threshold_95
    tail_mae_95 = float(np.mean(np.abs(residual[tail_mask_95])))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_correlation": correlation,
        "tail_mae_worst_10pct": tail_mae_90,
        "tail_mae_worst_5pct": tail_mae_95,
        "actual_mean": float(y_true.mean()),
        "predicted_mean": float(y_pred.mean()),
    }


def save_history(path: Path, history: Iterable[dict[str, float]]) -> None:
    rows = list(history)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "actual_drawdown", "predicted_drawdown"])
        for index, (actual, predicted_value) in enumerate(zip(y_true, y_pred)):
            writer.writerow([index, float(actual), float(predicted_value)])


def plot_history(
    history: list[dict[str, float]],
    *,
    title: str,
    best_epoch: int,
    destination: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train = [row["train_mse_scaled"] for row in history]
    validation = [row["validation_mse_scaled"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, validation, label="Validation")
    ax.axvline(best_epoch, linestyle="--", label=f"Best epoch: {best_epoch}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on standardized drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    destination: Path,
) -> None:
    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=14)
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual maximum drawdown")
    ax.set_ylabel("Predicted maximum drawdown")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def plot_metric_comparison(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not np.isclose(args.btc_weight + args.eth_weight, 1.0):
        raise ValueError("BTC and ETH weights must add up to 1")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience and batch-size must be positive")

    config = TrainConfig(
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_arrays = load_npz(args.data_dir / "validation_real_only.npz")
    test_arrays = load_npz(args.data_dir / "test_real_only.npz")
    validation_x, validation_y, _ = dataset_xy(
        validation_arrays,
        btc_weight=args.btc_weight,
        eth_weight=args.eth_weight,
    )
    test_x, test_y, _ = dataset_xy(
        test_arrays,
        btc_weight=args.btc_weight,
        eth_weight=args.eth_weight,
    )

    real_only_arrays = load_npz(args.data_dir / "train_real_only.npz")
    _, real_only_y, _ = dataset_xy(
        real_only_arrays,
        btc_weight=args.btc_weight,
        eth_weight=args.eth_weight,
    )
    target_mean = float(real_only_y.mean())
    target_std = float(real_only_y.std(ddof=0))
    if target_std <= 0:
        raise ValueError("Real-only training target has zero standard deviation")

    print("\nDOWNSTREAM TASK")
    print("=" * 72)
    print("Predict 30-day maximum drawdown magnitude for a 60/40 BTC-ETH portfolio")
    print(f"Device: {device}")
    print(f"Real validation samples: {len(validation_x)}")
    print(f"Real test samples: {len(test_x)}")
    print(f"Shared target scaling from real train: mean={target_mean:.6f}, std={target_std:.6f}")

    summary_rows: list[dict[str, object]] = []
    all_results: dict[str, object] = {
        "task": "maximum_30d_drawdown_magnitude_regression",
        "portfolio": {
            "initial_value": args.portfolio_value,
            "btc_weight": args.btc_weight,
            "eth_weight": args.eth_weight,
        },
        "shared_target_scaling": {
            "mean_from_real_train": target_mean,
            "std_from_real_train": target_std,
        },
        "configuration": asdict(config),
        "device": str(device),
        "datasets": {},
    }

    labels: list[str] = []
    test_maes: list[float] = []
    test_tail_maes: list[float] = []

    for train_filename in TRAIN_FILES:
        dataset_label = train_filename.removeprefix("train_").removesuffix(".npz")
        dataset_output = args.output_dir / dataset_label
        dataset_output.mkdir(parents=True, exist_ok=True)

        arrays = load_npz(args.data_dir / train_filename)
        train_x, train_y, source = dataset_xy(
            arrays,
            btc_weight=args.btc_weight,
            eth_weight=args.eth_weight,
        )
        real_count = int(np.sum(source == 0))
        synthetic_count = int(np.sum(source == 1))

        print("\n" + "-" * 72)
        print(
            f"Training {dataset_label}: total={len(train_x)}, "
            f"real={real_count}, synthetic={synthetic_count}"
        )

        model, history, best_epoch, best_validation = train_one_model(
            train_x,
            train_y,
            validation_x,
            validation_y,
            target_mean=target_mean,
            target_std=target_std,
            config=config,
            device=device,
        )

        validation_prediction = predict(
            model,
            validation_x,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=config.batch_size,
            device=device,
        )
        test_prediction = predict(
            model,
            test_x,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=config.batch_size,
            device=device,
        )
        validation_metrics = regression_metrics(validation_y, validation_prediction)
        test_metrics = regression_metrics(test_y, test_prediction)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "configuration": asdict(config),
            "target_mean": target_mean,
            "target_std": target_std,
            "best_epoch": best_epoch,
            "best_validation_mse_scaled": best_validation,
            "dataset": train_filename,
        }
        torch.save(checkpoint, dataset_output / "best_model.pt")
        save_history(dataset_output / "training_history.csv", history)
        save_predictions(
            dataset_output / "validation_predictions.csv",
            validation_y,
            validation_prediction,
        )
        save_predictions(
            dataset_output / "test_predictions.csv",
            test_y,
            test_prediction,
        )
        plot_history(
            history,
            title=f"Downstream training — {dataset_label}",
            best_epoch=best_epoch,
            destination=dataset_output / "training_curve.png",
        )
        plot_predictions(
            test_y,
            test_prediction,
            title=f"Test predictions — {dataset_label}",
            destination=dataset_output / "test_predicted_vs_actual.png",
        )

        dataset_result = {
            "training_file": train_filename,
            "total_training_samples": len(train_x),
            "real_training_samples": real_count,
            "synthetic_training_samples": synthetic_count,
            "best_epoch": best_epoch,
            "best_validation_mse_scaled": best_validation,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }
        all_results["datasets"][dataset_label] = dataset_result

        summary_rows.append(
            {
                "dataset": dataset_label,
                "total_training_samples": len(train_x),
                "real_training_samples": real_count,
                "synthetic_training_samples": synthetic_count,
                "best_epoch": best_epoch,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_correlation": test_metrics["pearson_correlation"],
                "test_tail_mae_worst_10pct": test_metrics[
                    "tail_mae_worst_10pct"
                ],
                "test_tail_mae_worst_5pct": test_metrics[
                    "tail_mae_worst_5pct"
                ],
            }
        )

        labels.append(dataset_label)
        test_maes.append(test_metrics["mae"])
        test_tail_maes.append(test_metrics["tail_mae_worst_10pct"])

        print(
            f"Best epoch={best_epoch} | "
            f"validation MAE={validation_metrics['mae']:.4%} | "
            f"test MAE={test_metrics['mae']:.4%} | "
            f"test tail MAE={test_metrics['tail_mae_worst_10pct']:.4%}"
        )

    summary_path = args.output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    (args.output_dir / "comparison_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    plot_metric_comparison(
        labels,
        test_maes,
        title="Test MAE by Normalizing Flow augmentation ratio",
        ylabel="MAE of maximum drawdown prediction",
        destination=args.output_dir / "comparison_test_mae.png",
    )
    plot_metric_comparison(
        labels,
        test_tail_maes,
        title="Worst-decile test MAE by augmentation ratio",
        ylabel="MAE on worst 10% actual drawdowns",
        destination=args.output_dir / "comparison_test_tail_mae.png",
    )

    print("\n" + "=" * 72)
    print("FINAL REAL-ONLY TEST COMPARISON")
    print("=" * 72)
    for row in summary_rows:
        print(
            f"{str(row['dataset']):<28} "
            f"MAE={float(row['test_mae']):.4%} "
            f"RMSE={float(row['test_rmse']):.4%} "
            f"R2={float(row['test_r2']): .4f} "
            f"Tail-MAE={float(row['test_tail_mae_worst_10pct']):.4%}"
        )
    print(f"\nSaved models, curves and comparison in: {args.output_dir}")


if __name__ == "__main__":
    main()
