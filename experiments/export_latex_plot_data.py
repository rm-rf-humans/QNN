from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.methodology_coherence_experiments import (  # noqa: E402
    backbone_display_name,
    best_qnn_result_row,
    collect_split_logits,
    fit_temperature,
    strongest_classical_baseline_row,
)
from experiments.qnn_experiments import cache_path_for_backbone  # noqa: E402
from experiments.segmentation_and_hypothesis_tests import (  # noqa: E402
    DMRIRSegmentationDataset,
    SmallUNet,
    concat_tables,
    download_shards,
    evaluate_otsu_baseline,
    load_checkpoint_model,
    metric_dict,
)


PAPER_DIR = (
    ROOT
    / "Utilizing_Hybrid_Quantum_Neural_Networks__H_QNNs__and_Complex_valued_Neural_Networks__CVNNs__for_Thermogram_Breast_Cancer_Classification"
)
PLOT_DATA_DIR = PAPER_DIR / "PlotData"
QNN_TABLES = ROOT / "artifacts" / "qnn_experiments" / "tables"
QNN_MODELS = ROOT / "artifacts" / "qnn_experiments" / "models"
SEG_ROOT = ROOT / "artifacts" / "segmentation_hypothesis"


def ensure_dir() -> None:
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def export_architecture_plot_data() -> None:
    df = pd.read_csv(QNN_TABLES / "architecture_results.csv")
    short_names = {
        "ResNet18 + MLP": "ResNet18+MLP",
        "DenseNet121 + MLP": "DenseNet121+MLP",
        "DenseNet121 + QNN": "DenseNet121+QNN",
        "MobileNetV3-Small + QNN": "MobileNetV3+QNN",
        "ResNet18 + QNN": "ResNet18+QNN",
    }
    out = df[["name", "test_accuracy", "test_f1", "test_auc"]].copy()
    out["model"] = out["name"].map(short_names)
    out = out[["model", "test_accuracy", "test_f1", "test_auc"]]
    out.to_csv(PLOT_DATA_DIR / "architecture_bars.csv", index=False)


def export_tuning_plot_data() -> None:
    df = pd.read_csv(QNN_TABLES / "tuning_results.csv")
    depths = sorted(df["q_depth"].unique())
    out = pd.DataFrame({"depth": depths})
    series = {
        "adam_lr_1e3": (("adam", 0.0010)),
        "adamw_lr_1e3": (("adamw", 0.0010)),
        "adam_lr_5e4": (("adam", 0.0005)),
        "adamw_lr_5e4": (("adamw", 0.0005)),
    }
    for column, (optimizer, lr) in series.items():
        values = []
        for depth in depths:
            row = df[
                (df["optimizer"] == optimizer)
                & (np.isclose(df["lr"], lr))
                & (df["q_depth"] == depth)
            ]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(float(row.iloc[0]["test_f1"]))
        out[column] = values
    out.to_csv(PLOT_DATA_DIR / "tuning_depth_lines.csv", index=False)


def model_probabilities(checkpoint_ref: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    checkpoint_path = Path(checkpoint_ref)
    if not checkpoint_path.is_absolute():
        checkpoint_path = QNN_MODELS / str(checkpoint_ref)
    model, backbone = load_checkpoint_model(checkpoint_path)
    bundle = torch.load(cache_path_for_backbone(backbone), map_location="cpu")
    features = bundle["test"]["features"]
    labels = bundle["test"]["labels"].numpy()
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return labels, probs, backbone


def best_qnn_plot_label() -> str:
    best_row = best_qnn_result_row()
    return f"{backbone_display_name(str(best_row['backbone']))}+QNN"


def pr_roc_model_entries() -> list[tuple[Path, str, str]]:
    architecture_df = pd.read_csv(QNN_TABLES / "architecture_results.csv")
    best_row = best_qnn_result_row()
    best_backbone = str(best_row["backbone"])
    strongest_baseline = strongest_classical_baseline_row()

    entries: list[tuple[Path, str, str]] = [
        (Path(str(best_row["model_path"])), best_qnn_plot_label(), "best_qnn"),
    ]

    alt_qnns = architecture_df[
        (architecture_df["model_type"] == "qnn")
        & (architecture_df["backbone"] != best_backbone)
    ].sort_values("test_f1", ascending=False)
    for alt_index, (_, row) in enumerate(alt_qnns.iterrows(), start=1):
        label = str(row["name"]).replace(" + ", "+")
        entries.append((Path(str(row["model_path"])), label, f"alt_qnn_{alt_index}"))

    baseline_label = str(strongest_baseline["name"]).replace(" + ", "+")
    entries.append(
        (
            Path(str(strongest_baseline["model_path"])),
            baseline_label,
            "strongest_classical_baseline",
        )
    )
    return entries


def export_pr_curve_data() -> None:
    summary_rows = []
    for checkpoint_path, model_name, output_stub in pr_roc_model_entries():
        labels, probs, _ = model_probabilities(checkpoint_path)
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        curve_df = pd.DataFrame({"recall": recall, "precision": precision})
        curve_df.to_csv(
            PLOT_DATA_DIR / f"pr_{output_stub}.csv",
            index=False,
        )
        summary_rows.append({"model": model_name, "average_precision": ap})
    pd.DataFrame(summary_rows).to_csv(PLOT_DATA_DIR / "pr_summary.csv", index=False)


def export_roc_curve_data() -> None:
    summary_rows = []
    for checkpoint_path, model_name, output_stub in pr_roc_model_entries():
        labels, probs, _ = model_probabilities(checkpoint_path)
        fpr, tpr, _ = roc_curve(labels, probs)
        curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        curve_df.to_csv(
            PLOT_DATA_DIR / f"roc_{output_stub}.csv",
            index=False,
        )
        summary_rows.append(
            {
                "model": model_name,
                "auc": float(np.trapezoid(tpr, fpr)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(PLOT_DATA_DIR / "roc_summary.csv", index=False)


def export_calibration_plot_data() -> None:
    def summarize_curve(
        labels: np.ndarray,
        probs: np.ndarray,
        model_name: str,
        output_name: str,
        temperature: float = 1.0,
    ) -> tuple[list[dict[str, float | str | int]], dict[str, float | str | int]]:
        quantile_edges = np.quantile(probs, np.linspace(0.0, 1.0, 9))
        quantile_edges[0] = 0.0
        quantile_edges[-1] = 1.0
        quantile_edges = np.unique(quantile_edges)

        ece = 0.0
        model_rows: list[dict[str, float | str | int]] = []
        for bin_index, (left, right) in enumerate(zip(quantile_edges[:-1], quantile_edges[1:]), start=1):
            if bin_index == len(quantile_edges) - 1:
                mask = (probs >= left) & (probs <= right)
            else:
                mask = (probs >= left) & (probs < right)
            if not np.any(mask):
                continue
            mean_confidence = float(probs[mask].mean())
            empirical_accuracy = float(labels[mask].mean())
            count = int(mask.sum())
            ece += (count / len(labels)) * abs(mean_confidence - empirical_accuracy)
            model_rows.append(
                {
                    "model": model_name,
                    "bin": bin_index,
                    "mean_predicted_probability": mean_confidence,
                    "empirical_positive_rate": empirical_accuracy,
                    "count": count,
                }
            )

        pd.DataFrame(model_rows).to_csv(PLOT_DATA_DIR / output_name, index=False)

        return model_rows, {
            "model": model_name,
            "temperature": temperature,
            "ece": float(ece),
            "brier": float(brier_score_loss(labels, probs)),
            "points": len(model_rows),
        }

    best_qnn = best_qnn_result_row()
    qnn_checkpoint = Path(str(best_qnn["model_path"]))
    qnn_name = best_qnn_plot_label()
    strongest_baseline = strongest_classical_baseline_row()
    mlp_checkpoint = Path(str(strongest_baseline["model_path"]))
    mlp_name = str(strongest_baseline["name"]).replace(" + ", "+")

    qnn_val_labels, qnn_val_logits, _ = collect_split_logits(qnn_checkpoint, "val")
    qnn_test_labels, qnn_test_logits, _ = collect_split_logits(qnn_checkpoint, "test")
    qnn_temperature = fit_temperature(qnn_val_logits, qnn_val_labels)
    qnn_raw_probs = torch.softmax(qnn_test_logits, dim=1)[:, 1].detach().cpu().numpy()
    qnn_scaled_probs = torch.softmax(qnn_test_logits / qnn_temperature, dim=1)[:, 1].detach().cpu().numpy()

    mlp_labels, mlp_logits, _ = collect_split_logits(mlp_checkpoint, "test")
    mlp_probs = torch.softmax(mlp_logits, dim=1)[:, 1].detach().cpu().numpy()

    calibration_rows: list[dict[str, float | str | int]] = []
    summary_rows: list[dict[str, float | str | int]] = []

    for labels, probs, model_name, output_name, temperature in [
        (
            qnn_test_labels,
            qnn_raw_probs,
            f"{qnn_name} raw",
            "calibration_best_qnn.csv",
            1.0,
        ),
        (
            qnn_test_labels,
            qnn_raw_probs,
            f"{qnn_name} raw",
            "calibration_best_qnn_raw.csv",
            1.0,
        ),
        (
            qnn_test_labels,
            qnn_scaled_probs,
            f"{qnn_name} temperature scaled",
            "calibration_best_qnn_temp_scaled.csv",
            float(qnn_temperature),
        ),
        (
            mlp_labels,
            mlp_probs,
            mlp_name,
            "calibration_strongest_classical_baseline.csv",
            1.0,
        ),
    ]:
        model_rows, summary_row = summarize_curve(
            labels=labels,
            probs=probs,
            model_name=model_name,
            output_name=output_name,
            temperature=temperature,
        )
        if output_name != "calibration_densenet121_qnn.csv":
            calibration_rows.extend(model_rows)
            summary_rows.append(summary_row)

    pd.DataFrame(calibration_rows).to_csv(PLOT_DATA_DIR / "calibration_curves.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(PLOT_DATA_DIR / "calibration_summary.csv", index=False)


def export_score_separation_data() -> None:
    labels, probs, _ = model_probabilities(Path(str(best_qnn_result_row()["model_path"])))
    bins = np.linspace(0.0, 1.0, 21)
    normal_probs = probs[labels == 0]
    abnormal_probs = probs[labels == 1]
    normal_hist, edges = np.histogram(normal_probs, bins=bins, density=True)
    abnormal_hist, _ = np.histogram(abnormal_probs, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = pd.DataFrame(
        {
            "bin_center": centers,
            "normal_density": normal_hist,
            "abnormal_density": abnormal_hist,
        }
    )
    out.to_csv(PLOT_DATA_DIR / "qnn_score_density.csv", index=False)


def export_segmentation_plot_data() -> None:
    metrics = pd.read_csv(SEG_ROOT / "tables" / "segmentation_results.csv")
    metric_cols = [
        ("Dice", "test_dice"),
        ("IoU", "test_iou"),
        ("Precision", "test_precision"),
        ("Recall", "test_recall"),
        ("PixelAcc", "test_pixel_accuracy"),
    ]
    unet_row = metrics[metrics["model"] == "U-Net"].iloc[0]
    otsu_row = metrics[metrics["model"] == "Otsu threshold baseline"].iloc[0]
    rows = []
    for metric_name, col in metric_cols:
        rows.append(
            {
                "metric": metric_name,
                "unet": float(unet_row[col]),
                "otsu": float(otsu_row[col]),
            }
        )
    pd.DataFrame(rows).to_csv(PLOT_DATA_DIR / "segmentation_metric_bars.csv", index=False)

    history = json.loads((SEG_ROOT / "models" / "history.json").read_text())
    pd.DataFrame(history).to_csv(PLOT_DATA_DIR / "segmentation_history.csv", index=False)


def export_segmentation_distribution_data() -> None:
    downloaded_shards = download_shards()
    test_dataset = DMRIRSegmentationDataset(concat_tables(downloaded_shards["test"]))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    checkpoint = torch.load(SEG_ROOT / "models" / "dmr_ir_subset_unet.pt", map_location="cpu")
    model = SmallUNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_batches = []
    target_batches = []
    with torch.no_grad():
        for images, masks in test_loader:
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            pred_batches.append(preds.squeeze(1).cpu().numpy())
            target_batches.append(masks.squeeze(1).cpu().numpy())

    preds_all = np.concatenate(pred_batches, axis=0)
    targets_all = np.concatenate(target_batches, axis=0)
    unet_metrics = metric_dict(preds_all, targets_all)
    _, otsu_metrics = evaluate_otsu_baseline(test_loader)

    boxplot_rows = []
    for metric_name in ("dice", "iou"):
        for method_name, metric_values in (
            ("U-Net", unet_metrics[metric_name]),
            ("Otsu", otsu_metrics[metric_name]),
        ):
            q1, median, q3 = np.quantile(metric_values, [0.25, 0.5, 0.75])
            boxplot_rows.append(
                {
                    "metric": metric_name,
                    "method": method_name,
                    "lower_whisker": float(metric_values.min()),
                    "lower_quartile": float(q1),
                    "median": float(median),
                    "upper_quartile": float(q3),
                    "upper_whisker": float(metric_values.max()),
                    "mean": float(metric_values.mean()),
                }
            )
    pd.DataFrame(boxplot_rows).to_csv(
        PLOT_DATA_DIR / "segmentation_boxplot_summary.csv",
        index=False,
    )

    dice_gain = unet_metrics["dice"] - otsu_metrics["dice"]
    hist, edges = np.histogram(dice_gain, bins=np.linspace(-1.0, 1.0, 21))
    interval_rows = [
        {
            "bin_left": float(left_edge),
            "count": int(count),
        }
        for left_edge, count in zip(edges[:-1], hist)
    ]
    interval_rows.append(
        {
            "bin_left": float(edges[-1]),
            "count": 0,
        }
    )
    pd.DataFrame(interval_rows).to_csv(
        PLOT_DATA_DIR / "segmentation_dice_gain_hist.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "mean_gain": float(dice_gain.mean()),
                "median_gain": float(np.median(dice_gain)),
                "fraction_positive": float((dice_gain > 0).mean()),
                "fraction_nonnegative": float((dice_gain >= 0).mean()),
            }
        ]
    ).to_csv(PLOT_DATA_DIR / "segmentation_gain_summary.csv", index=False)

    for metric_name in ("dice", "iou"):
        for method_name, metric_values in (
            ("U-Net", unet_metrics[metric_name]),
            ("Otsu", otsu_metrics[metric_name]),
        ):
            sorted_values = np.sort(metric_values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            pd.DataFrame(
                {
                    "score": sorted_values.astype(float),
                    "cdf": cumulative.astype(float),
                }
            ).to_csv(
                PLOT_DATA_DIR
                / f"segmentation_{metric_name}_cdf_{method_name.lower().replace('-', '').replace(' ', '_')}.csv",
                index=False,
            )


def main() -> None:
    ensure_dir()
    export_architecture_plot_data()
    export_tuning_plot_data()
    export_pr_curve_data()
    export_roc_curve_data()
    export_calibration_plot_data()
    export_score_separation_data()
    export_segmentation_plot_data()
    export_segmentation_distribution_data()
    print(PLOT_DATA_DIR)


if __name__ == "__main__":
    main()
