from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qnn_experiments import (  # noqa: E402
    ARTIFACT_ROOT as QNN_ARTIFACT_ROOT,
    ExperimentConfig,
    build_splits,
    cache_path_for_backbone,
    compute_metrics,
    extract_transfer_features,
    set_seed,
    train_head_experiment,
)
from experiments.segmentation_and_hypothesis_tests import load_checkpoint_model  # noqa: E402


ARTIFACT_ROOT = ROOT / "artifacts" / "methodology_coherence"
PAPER_DIR = (
    ROOT
    / "Utilizing_Hybrid_Quantum_Neural_Networks__H_QNNs__and_Complex_valued_Neural_Networks__CVNNs__for_Thermogram_Breast_Cancer_Classification"
)
PLOT_DATA_DIR = PAPER_DIR / "PlotData"
REPEATED_SPLIT_SEEDS = [42, 123, 314, 2024, 2025]
QNN_RESTART_OFFSETS = [0, 333, 666]
BACKBONE_DISPLAY_NAMES = {
    "resnet18": "ResNet18",
    "densenet121": "DenseNet121",
    "mobilenet_v3_small": "MobileNetV3-Small",
}


def ensure_dirs() -> None:
    for path in [
        ARTIFACT_ROOT,
        ARTIFACT_ROOT / "cache",
        ARTIFACT_ROOT / "models",
        ARTIFACT_ROOT / "tables",
        PLOT_DATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 8) -> float:
    quantile_edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 2:
        return 0.0

    ece = 0.0
    for bin_index, (left, right) in enumerate(zip(quantile_edges[:-1], quantile_edges[1:]), start=1):
        if bin_index == len(quantile_edges) - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        mean_confidence = float(probs[mask].mean())
        empirical_positive_rate = float(labels[mask].mean())
        ece += (mask.sum() / len(labels)) * abs(mean_confidence - empirical_positive_rate)
    return float(ece)


def classification_metrics_from_logits(
    labels: np.ndarray,
    logits: torch.Tensor,
) -> dict[str, float]:
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    metrics = compute_metrics(labels, preds, probs)
    metrics["ece"] = compute_ece(labels, probs, n_bins=8)
    metrics["brier"] = float(brier_score_loss(labels, probs))
    metrics["nll"] = float(log_loss(labels, probs, labels=[0, 1]))
    metrics["mean_positive_probability"] = float(probs.mean())
    return metrics


def backbone_display_name(backbone: str) -> str:
    return BACKBONE_DISPLAY_NAMES.get(backbone, backbone)


def best_qnn_result_row() -> pd.Series:
    tuning_results_path = QNN_ARTIFACT_ROOT / "tables" / "tuning_results.csv"
    if tuning_results_path.exists():
        tuning_results = pd.read_csv(tuning_results_path)
        if not tuning_results.empty:
            return tuning_results.sort_values(["val_f1", "test_f1"], ascending=False).iloc[0]

    architecture_results = pd.read_csv(QNN_ARTIFACT_ROOT / "tables" / "architecture_results.csv")
    qnn_rows = architecture_results[architecture_results["model_type"] == "qnn"].copy()
    if qnn_rows.empty:
        raise RuntimeError("No QNN model found in architecture_results.csv")
    return qnn_rows.sort_values(["val_f1", "test_f1"], ascending=False).iloc[0]


def strongest_classical_baseline_row() -> pd.Series:
    architecture_results = pd.read_csv(QNN_ARTIFACT_ROOT / "tables" / "architecture_results.csv")
    classical_rows = architecture_results[architecture_results["model_type"] == "mlp"].copy()
    if classical_rows.empty:
        raise RuntimeError("No classical MLP baseline found in architecture_results.csv")
    return classical_rows.sort_values(["test_f1", "val_f1"], ascending=False).iloc[0]


def collect_split_logits(
    checkpoint_path: Path,
    split_name: str,
    cache_tag: str | None = None,
    artifact_root: Path = QNN_ARTIFACT_ROOT,
) -> tuple[np.ndarray, torch.Tensor, str]:
    model, backbone = load_checkpoint_model(checkpoint_path)
    bundle = torch.load(
        cache_path_for_backbone(backbone, cache_tag=cache_tag, artifact_root=artifact_root),
        map_location="cpu",
    )
    features = bundle[split_name]["features"]
    labels = bundle[split_name]["labels"].numpy()
    with torch.no_grad():
        logits = model(features)
    return labels, logits, backbone


def fit_temperature(val_logits: torch.Tensor, val_labels: np.ndarray) -> float:
    labels = torch.from_numpy(val_labels).long()
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = criterion(val_logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).item())


def export_calibration_curve(
    output_path: Path,
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    n_bins: int = 8,
) -> None:
    quantile_edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0
    quantile_edges = np.unique(quantile_edges)
    rows: list[dict[str, float | str | int]] = []
    for bin_index, (left, right) in enumerate(zip(quantile_edges[:-1], quantile_edges[1:]), start=1):
        if bin_index == len(quantile_edges) - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "model": model_name,
                "bin": bin_index,
                "mean_predicted_probability": float(probs[mask].mean()),
                "empirical_positive_rate": float(labels[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_temperature_scaling_study() -> pd.DataFrame:
    best_qnn_row = best_qnn_result_row()
    best_qnn_path = Path(str(best_qnn_row["model_path"]))
    best_qnn_name = f"{backbone_display_name(str(best_qnn_row['backbone']))} + QNN"
    baseline_row = strongest_classical_baseline_row()
    baseline_path = Path(str(baseline_row["model_path"]))
    baseline_name = str(baseline_row["name"])

    qnn_val_labels, qnn_val_logits, _ = collect_split_logits(best_qnn_path, "val")
    qnn_test_labels, qnn_test_logits, _ = collect_split_logits(best_qnn_path, "test")
    temperature = fit_temperature(qnn_val_logits, qnn_val_labels)

    qnn_raw_probs = torch.softmax(qnn_test_logits, dim=1)[:, 1].detach().cpu().numpy()
    qnn_scaled_logits = qnn_test_logits / temperature
    qnn_scaled_probs = torch.softmax(qnn_scaled_logits, dim=1)[:, 1].detach().cpu().numpy()

    baseline_labels, baseline_logits, _ = collect_split_logits(baseline_path, "test")
    baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1].detach().cpu().numpy()

    rows = [
        {
            "model": f"{best_qnn_name} (raw)",
            "temperature": 1.0,
            **classification_metrics_from_logits(qnn_test_labels, qnn_test_logits),
        },
        {
            "model": f"{best_qnn_name} (temperature scaled)",
            "temperature": temperature,
            **classification_metrics_from_logits(qnn_test_labels, qnn_scaled_logits),
        },
        {
            "model": f"{baseline_name} (raw)",
            "temperature": 1.0,
            **classification_metrics_from_logits(baseline_labels, baseline_logits),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACT_ROOT / "tables" / "temperature_scaling_results.csv", index=False)

    export_calibration_curve(
        PLOT_DATA_DIR / "calibration_best_qnn_raw.csv",
        qnn_test_labels,
        qnn_raw_probs,
        f"{best_qnn_name} raw",
    )
    export_calibration_curve(
        PLOT_DATA_DIR / "calibration_best_qnn_temp_scaled.csv",
        qnn_test_labels,
        qnn_scaled_probs,
        f"{best_qnn_name} temperature scaled",
    )
    export_calibration_curve(
        PLOT_DATA_DIR / "calibration_strongest_classical_baseline.csv",
        baseline_labels,
        baseline_probs,
        baseline_name,
    )

    return df


def repeated_split_experiment_configs() -> list[tuple[str, ExperimentConfig]]:
    best_qnn_row = best_qnn_result_row()
    best_backbone = str(best_qnn_row["backbone"])
    best_qnn_label = f"{backbone_display_name(best_backbone)} + QNN"
    best_mlp_label = f"{backbone_display_name(best_backbone)} + MLP"
    return [
        (
            best_qnn_label,
            ExperimentConfig(
                name=f"Repeated {best_qnn_label}",
                study="robustness",
                model_type="qnn",
                backbone=best_backbone,
                q_depth=int(best_qnn_row["q_depth"]),
                optimizer=str(best_qnn_row["optimizer"]),
                lr=float(best_qnn_row["lr"]),
            ),
        ),
        (
            best_mlp_label,
            ExperimentConfig(
                name=f"Repeated {best_mlp_label}",
                study="robustness",
                model_type="mlp",
                backbone=best_backbone,
                optimizer="adamw",
                lr=1e-3,
            ),
        ),
    ]


def run_repeated_split_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    for split_seed in REPEATED_SPLIT_SEEDS:
        splits = build_splits(seed=split_seed, artifact_root=ARTIFACT_ROOT)
        feature_cache: dict[str, dict[str, dict[str, Any]]] = {}

        for model_label, base_config in repeated_split_experiment_configs():
            if base_config.backbone not in feature_cache:
                feature_cache[base_config.backbone] = extract_transfer_features(
                    base_config.backbone,
                    splits,
                    cache_tag=f"split_seed_{split_seed}",
                    artifact_root=ARTIFACT_ROOT,
                )
            input_dim = feature_cache[base_config.backbone]["train"]["features"].shape[1]
            if base_config.model_type == "qnn":
                split_results = []
                for restart_offset in QNN_RESTART_OFFSETS:
                    restart_seed = split_seed + restart_offset
                    set_seed(restart_seed)
                    config = replace(
                        base_config,
                        name=f"{base_config.name} split_seed={split_seed} restart_seed={restart_seed}",
                    )
                    result = train_head_experiment(
                        config,
                        feature_cache[base_config.backbone],
                        input_dim=input_dim,
                        artifact_root=ARTIFACT_ROOT,
                    )
                    result["split_seed"] = split_seed
                    result["restart_seed"] = restart_seed
                    result["model_label"] = model_label
                    restart_rows.append(result.copy())
                    split_results.append(result)
                best_result = max(split_results, key=lambda row: row["val_f1"])
                rows.append(best_result)
            else:
                set_seed(split_seed)
                config = replace(
                    base_config,
                    name=f"{base_config.name} split_seed={split_seed}",
                )
                result = train_head_experiment(
                    config,
                    feature_cache[base_config.backbone],
                    input_dim=input_dim,
                    artifact_root=ARTIFACT_ROOT,
                )
                result["split_seed"] = split_seed
                result["restart_seed"] = split_seed
                result["model_label"] = model_label
                rows.append(result)

    per_split_df = pd.DataFrame(rows)
    per_split_df.to_csv(ARTIFACT_ROOT / "tables" / "repeated_split_results.csv", index=False)
    if restart_rows:
        pd.DataFrame(restart_rows).to_csv(
            ARTIFACT_ROOT / "tables" / "qnn_restart_results.csv",
            index=False,
        )

    summary_rows = []
    for model_label, group in per_split_df.groupby("model_label"):
        summary_rows.append(
            {
                "model": model_label,
                "n_splits": int(group["split_seed"].nunique()),
                "mean_test_accuracy": float(group["test_accuracy"].mean()),
                "std_test_accuracy": float(group["test_accuracy"].std(ddof=1)),
                "mean_test_f1": float(group["test_f1"].mean()),
                "std_test_f1": float(group["test_f1"].std(ddof=1)),
                "mean_test_auc": float(group["test_auc"].mean()),
                "std_test_auc": float(group["test_auc"].std(ddof=1)),
                "mean_test_recall": float(group["test_recall"].mean()),
                "std_test_recall": float(group["test_recall"].std(ddof=1)),
                "mean_test_specificity": float(group["test_specificity"].mean()),
                "std_test_specificity": float(group["test_specificity"].std(ddof=1)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_test_f1", ascending=False)
    summary_df.to_csv(ARTIFACT_ROOT / "tables" / "repeated_split_summary.csv", index=False)
    return per_split_df, summary_df


def write_summary(
    temperature_df: pd.DataFrame,
    repeated_split_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Methodology Coherence Experiments",
        "",
        "## Temperature Scaling",
    ]
    for row in temperature_df.to_dict(orient="records"):
        lines.append(
            f"- {row['model']}: ECE={row['ece']:.4f}, Brier={row['brier']:.4f}, NLL={row['nll']:.4f}, AUC={row['auc']:.4f}, temperature={row['temperature']:.4f}"
        )
    lines.extend(["", "## Repeated Split Robustness"])
    for row in repeated_split_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['model']}: acc={row['mean_test_accuracy']:.4f}±{row['std_test_accuracy']:.4f}, F1={row['mean_test_f1']:.4f}±{row['std_test_f1']:.4f}, AUC={row['mean_test_auc']:.4f}±{row['std_test_auc']:.4f}"
        )

    (ARTIFACT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (ARTIFACT_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "temperature_scaling": temperature_df.to_dict(orient="records"),
                "repeated_split_summary": repeated_split_summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )


def main() -> None:
    ensure_dirs()
    set_seed(42)
    temperature_df = run_temperature_scaling_study()
    _, repeated_split_summary = run_repeated_split_robustness()
    write_summary(temperature_df, repeated_split_summary)
    print(ARTIFACT_ROOT / "tables" / "temperature_scaling_results.csv")
    print(ARTIFACT_ROOT / "tables" / "repeated_split_summary.csv")


if __name__ == "__main__":
    main()
