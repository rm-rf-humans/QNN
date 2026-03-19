from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.stats import binomtest, wilcoxon
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qnn_experiments import (  # noqa: E402
    ARTIFACT_ROOT,
    MLPHead,
    QuantumHead,
    cache_path_for_backbone,
)

SEGMENTATION_ARTIFACT_ROOT = ROOT / "artifacts" / "segmentation_hypothesis"
DMR_CACHE_ROOT = ROOT / "artifacts" / "dmr_ir_cache"
PAPER_FIGURES_DIR = (
    ROOT
    / "Utilizing_Hybrid_Quantum_Neural_Networks__H_QNNs__and_Complex_valued_Neural_Networks__CVNNs__for_Thermogram_Breast_Cancer_Classification"
    / "Figures"
)

SEED = 42
TARGET_SIZE = (224, 224)
TRAIN_BATCH_SIZE = 8
EPOCHS = 6
PATIENCE = 2

SHARD_CONFIG = {
    "train": [
        "with_mastectomy/train-00000-of-00013.parquet",
        "with_mastectomy/train-00001-of-00013.parquet",
    ],
    "validation": [
        "with_mastectomy/validation-00000-of-00003.parquet",
    ],
    "test": [
        "with_mastectomy/test-00000-of-00003.parquet",
    ],
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    for path in [
        SEGMENTATION_ARTIFACT_ROOT,
        SEGMENTATION_ARTIFACT_ROOT / "figures",
        SEGMENTATION_ARTIFACT_ROOT / "models",
        SEGMENTATION_ARTIFACT_ROOT / "tables",
        DMR_CACHE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_shards() -> dict[str, list[Path]]:
    downloaded: dict[str, list[Path]] = {}
    for split, files in SHARD_CONFIG.items():
        split_paths: list[Path] = []
        for filename in files:
            local_path = hf_hub_download(
                repo_id="SemilleroCV/DMR-IR",
                repo_type="dataset",
                filename=filename,
                local_dir=str(DMR_CACHE_ROOT),
            )
            split_paths.append(Path(local_path))
        downloaded[split] = split_paths
    return downloaded


def concat_tables(paths: list[Path]) -> pa.Table:
    tables = []
    for path in paths:
        table = pq.read_table(
            path,
            columns=["image", "segmentation_mask", "label", "view", "record"],
        )
        tables.append(table)
    return pa.concat_tables(tables)


class DMRIRSegmentationDataset(Dataset):
    def __init__(self, table: pa.Table) -> None:
        self.table = table

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.table.slice(index, 1)
        image_dict = row["image"][0].as_py()
        mask_nested = row["segmentation_mask"][0].as_py()

        with Image.open(BytesIO(image_dict["bytes"])) as image:
            image_array = np.array(image, dtype=np.float32)

        image_resized = np.array(
            Image.fromarray(image_array, mode="F").resize(TARGET_SIZE, resample=Image.BILINEAR),
            dtype=np.float32,
        )
        image_resized -= image_resized.min()
        image_resized /= image_resized.max() + 1e-8

        mask_array = np.array(mask_nested[0], dtype=np.float32)
        mask_array = (mask_array > 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image_resized).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 16)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)
        self.bridge = DoubleConv(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.bridge(self.pool(x3))

        y = self.up3(x4)
        y = self.conv3(torch.cat([y, x3], dim=1))
        y = self.up2(y)
        y = self.conv2(torch.cat([y, x2], dim=1))
        y = self.up1(y)
        y = self.conv1(torch.cat([y, x1], dim=1))
        return self.out(y)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def metric_dict(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)

    intersection = (preds * targets).sum(axis=(1, 2))
    pred_sum = preds.sum(axis=(1, 2))
    target_sum = targets.sum(axis=(1, 2))
    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)
    pixel_acc = (preds == targets).mean(axis=(1, 2))
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": pixel_acc,
    }


def otsu_threshold(image: np.ndarray) -> float:
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0.0, 1.0))
    total = image.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0

    for level in range(256):
        weight_background += hist[level]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += level * hist[level]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance > max_variance:
            max_variance = variance
            threshold = level

    return threshold / 255.0


def evaluate_segmentation_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_example_pool: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict[str, np.ndarray | float | int]]]:
    model.eval()
    losses = []
    pred_batches = []
    target_batches = []
    example_pool: list[dict[str, np.ndarray | float | int]] = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            losses.append(loss.item() * images.size(0))
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            pred_np = preds.squeeze(1).cpu().numpy()
            target_np = masks.squeeze(1).cpu().numpy()
            image_np = images.squeeze(1).cpu().numpy()
            pred_batches.append(pred_np)
            target_batches.append(target_np)

            if collect_example_pool:
                prob_np = probs.squeeze(1).cpu().numpy()
                for batch_index in range(len(image_np)):
                    example_pool.append(
                        {
                            "image": image_np[batch_index],
                            "target": target_np[batch_index],
                            "prediction": pred_np[batch_index],
                            "probability": prob_np[batch_index],
                        }
                    )

    preds_all = np.concatenate(pred_batches, axis=0)
    targets_all = np.concatenate(target_batches, axis=0)
    metrics = metric_dict(preds_all, targets_all)
    summary = {key: float(values.mean()) for key, values in metrics.items()}
    summary["loss"] = float(sum(losses) / len(loader.dataset))
    return summary, metrics, example_pool


def case_metric_summary(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    metrics = metric_dict(pred[None, ...], target[None, ...])
    return {key: float(values[0]) for key, values in metrics.items()}


def evaluate_otsu_baseline(
    loader: DataLoader,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pred_batches = []
    target_batches = []
    for images, masks in loader:
        image_np = images.squeeze(1).numpy()
        target_np = masks.squeeze(1).numpy()
        preds = []
        for image in image_np:
            threshold = otsu_threshold(image)
            preds.append((image > threshold).astype(np.float32))
        pred_batches.append(np.stack(preds))
        target_batches.append(target_np)

    preds_all = np.concatenate(pred_batches, axis=0)
    targets_all = np.concatenate(target_batches, axis=0)
    metrics = metric_dict(preds_all, targets_all)
    summary = {key: float(values.mean()) for key, values in metrics.items()}
    return summary, metrics


def select_representative_segmentation_examples(
    example_pool: list[dict[str, np.ndarray | float | int]],
    max_examples: int = 4,
) -> list[dict[str, np.ndarray | float | str]]:
    enriched_cases: list[dict[str, np.ndarray | float | str]] = []
    for record in example_pool:
        image = np.asarray(record["image"], dtype=np.float32)
        target = np.asarray(record["target"], dtype=np.float32)
        prediction = np.asarray(record["prediction"], dtype=np.float32)
        otsu_pred = (image > otsu_threshold(image)).astype(np.float32)
        unet_metrics = case_metric_summary(prediction, target)
        otsu_metrics = case_metric_summary(otsu_pred, target)
        enriched_cases.append(
            {
                "image": image,
                "target": target,
                "prediction": prediction,
                "otsu_prediction": otsu_pred,
                "probability": np.asarray(record["probability"], dtype=np.float32),
                "unet_dice": unet_metrics["dice"],
                "otsu_dice": otsu_metrics["dice"],
                "dice_gain": unet_metrics["dice"] - otsu_metrics["dice"],
            }
        )

    if not enriched_cases:
        return []

    median_dice = float(np.median([float(case["unet_dice"]) for case in enriched_cases]))
    selection_specs = [
        ("Largest Gain", lambda case: float(case["dice_gain"])),
        ("Near-Median", lambda case: -abs(float(case["unet_dice"]) - median_dice)),
        ("Best U-Net", lambda case: float(case["unet_dice"])),
        ("Hard Case", lambda case: -float(case["unet_dice"])),
    ]

    selected_indices: list[int] = []
    selected_cases: list[dict[str, np.ndarray | float | str]] = []
    for label, score_fn in selection_specs[:max_examples]:
        ranked = sorted(
            enumerate(enriched_cases),
            key=lambda item: score_fn(item[1]),
            reverse=True,
        )
        for index, case in ranked:
            if index in selected_indices:
                continue
            selected_indices.append(index)
            selected_cases.append({**case, "case_label": label})
            break

    return selected_cases


@dataclass
class SegmentationTrainingResult:
    train_examples: int
    validation_examples: int
    test_examples: int
    history: list[dict[str, float]]
    best_epoch: int
    metrics_validation: dict[str, float]
    metrics_test: dict[str, float]
    baseline_test: dict[str, float]
    dice_wilcoxon_pvalue: float
    iou_wilcoxon_pvalue: float


def train_segmentation_model(downloaded_shards: dict[str, list[Path]]) -> SegmentationTrainingResult:
    device = choose_device()
    train_dataset = DMRIRSegmentationDataset(concat_tables(downloaded_shards["train"]))
    val_dataset = DMRIRSegmentationDataset(concat_tables(downloaded_shards["validation"]))
    test_dataset = DMRIRSegmentationDataset(concat_tables(downloaded_shards["test"]))

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_val_dice = -math.inf
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_summary, _, _ = evaluate_segmentation_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_summary["loss"],
                "val_dice": val_summary["dice"],
                "val_iou": val_summary["iou"],
            }
        )

        if val_summary["dice"] > best_val_dice:
            best_val_dice = val_summary["dice"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
            }
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state["model"])

    validation_summary, _, _ = evaluate_segmentation_model(model, val_loader, device)
    test_summary, test_metrics_per_image, example_pool = evaluate_segmentation_model(
        model,
        test_loader,
        device,
        collect_example_pool=True,
    )
    baseline_summary, baseline_metrics_per_image = evaluate_otsu_baseline(test_loader)

    dice_test = wilcoxon(
        test_metrics_per_image["dice"],
        baseline_metrics_per_image["dice"],
        alternative="greater",
        zero_method="wilcox",
    )
    iou_test = wilcoxon(
        test_metrics_per_image["iou"],
        baseline_metrics_per_image["iou"],
        alternative="greater",
        zero_method="wilcox",
    )

    model_path = SEGMENTATION_ARTIFACT_ROOT / "models" / "dmr_ir_subset_unet.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "history": history,
        },
        model_path,
    )
    (SEGMENTATION_ARTIFACT_ROOT / "models" / "history.json").write_text(
        json.dumps(history, indent=2)
    )

    representative_examples = select_representative_segmentation_examples(example_pool, max_examples=4)
    render_segmentation_examples(representative_examples)

    return SegmentationTrainingResult(
        train_examples=len(train_dataset),
        validation_examples=len(val_dataset),
        test_examples=len(test_dataset),
        history=history,
        best_epoch=best_epoch,
        metrics_validation=validation_summary,
        metrics_test=test_summary,
        baseline_test=baseline_summary,
        dice_wilcoxon_pvalue=float(dice_test.pvalue),
        iou_wilcoxon_pvalue=float(iou_test.pvalue),
    )


def render_segmentation_examples(
    examples: list[dict[str, np.ndarray | float | str]],
) -> Path:
    output_path = SEGMENTATION_ARTIFACT_ROOT / "figures" / "segmentation_examples.png"
    rows = len(examples)
    columns = 6
    fig, axes = plt.subplots(rows, columns, figsize=(18, 2.9 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Thermogram",
        "Grayscale",
        "Ground Truth",
        "Otsu Overlay",
        "U-Net Overlay",
        "Agreement Map",
    ]
    for col_index, title in enumerate(column_titles):
        axes[0, col_index].set_title(title, fontsize=10)

    for row_index, case in enumerate(examples):
        image = np.asarray(case["image"], dtype=np.float32)
        target = np.asarray(case["target"], dtype=np.float32)
        prediction = np.asarray(case["prediction"], dtype=np.float32)
        otsu_pred = np.asarray(case["otsu_prediction"], dtype=np.float32)

        agreement_map = np.zeros((*prediction.shape, 3), dtype=np.float32)
        true_positive = (prediction > 0.5) & (target > 0.5)
        false_positive = (prediction > 0.5) & (target <= 0.5)
        false_negative = (prediction <= 0.5) & (target > 0.5)
        agreement_map[true_positive] = np.array([0.12, 0.78, 0.22], dtype=np.float32)
        agreement_map[false_positive] = np.array([0.18, 0.45, 0.95], dtype=np.float32)
        agreement_map[false_negative] = np.array([0.92, 0.18, 0.18], dtype=np.float32)

        axes[row_index, 0].imshow(image, cmap="inferno")
        axes[row_index, 1].imshow(image, cmap="gray")
        axes[row_index, 2].imshow(target, cmap="gray")
        axes[row_index, 3].imshow(image, cmap="gray")
        axes[row_index, 3].imshow(otsu_pred, cmap="Blues", alpha=0.35)
        axes[row_index, 4].imshow(image, cmap="gray")
        axes[row_index, 4].imshow(prediction, cmap="viridis", alpha=0.35)
        axes[row_index, 5].imshow(image, cmap="gray")
        axes[row_index, 5].imshow(agreement_map, alpha=0.75)

        row_label = (
            f"{case['case_label']}\n"
            f"U-Net Dice {float(case['unet_dice']):.3f}\n"
            f"Otsu Dice {float(case['otsu_dice']):.3f}"
        )
        axes[row_index, 0].set_ylabel(
            row_label,
            rotation=0,
            fontsize=9,
            labelpad=42,
            va="center",
        )

        for col_index in range(columns):
            axes[row_index, col_index].axis("off")

    plt.tight_layout()
    fig.subplots_adjust(left=0.12, wspace=0.04, hspace=0.18)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    paper_path = PAPER_FIGURES_DIR / "Segmentation_Examples.png"
    paper_path.write_bytes(output_path.read_bytes())
    return output_path


def load_checkpoint_model(checkpoint_path: Path) -> tuple[nn.Module, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    input_dim = int(checkpoint["input_dim"])
    if config["model_type"] == "qnn":
        model = QuantumHead(
            input_dim=input_dim,
            q_depth=int(config["q_depth"]),
            dropout=float(config["dropout"]),
        )
    else:
        model = MLPHead(
            input_dim=input_dim,
            dropout=float(config["dropout"]),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config["backbone"]


def model_predictions(checkpoint_path: Path, backbone: str) -> tuple[np.ndarray, np.ndarray]:
    bundle = torch.load(cache_path_for_backbone(backbone), map_location="cpu")
    features = bundle["test"]["features"]
    labels = bundle["test"]["labels"].numpy()
    model, _ = load_checkpoint_model(checkpoint_path)
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1).numpy()
    return preds, labels


def exact_mcnemar(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c
    if discordant == 0:
        pvalue = 1.0
    else:
        pvalue = float(binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
    return {
        "b_only_model_a_correct": float(b),
        "c_only_model_b_correct": float(c),
        "discordant_pairs": float(discordant),
        "p_value": pvalue,
    }


def run_hypothesis_tests() -> pd.DataFrame:
    from experiments.methodology_coherence_experiments import backbone_display_name, best_qnn_result_row

    best_qnn_row = best_qnn_result_row()
    best_qnn_path = Path(str(best_qnn_row["model_path"]))
    best_backbone = str(best_qnn_row["backbone"])
    best_reference_name = (
        f"{backbone_display_name(best_backbone)} + QNN "
        f"(best tuned: depth={int(best_qnn_row['q_depth'])}, "
        f"{str(best_qnn_row['optimizer']).upper()}, lr={float(best_qnn_row['lr']):g})"
    )

    architecture_results = pd.read_csv(ARTIFACT_ROOT / "tables" / "architecture_results.csv")
    comparators: list[tuple[str, Path, str]] = []

    same_backbone_qnn = architecture_results[
        (architecture_results["model_type"] == "qnn")
        & (architecture_results["backbone"] == best_backbone)
    ]
    if not same_backbone_qnn.empty:
        arch_row = same_backbone_qnn.iloc[0]
        arch_path = Path(str(arch_row["model_path"]))
        if arch_path != best_qnn_path:
            comparators.append(
                (
                    f"{backbone_display_name(best_backbone)} + QNN (architecture-sweep checkpoint)",
                    arch_path,
                    best_backbone,
                )
            )

    same_backbone_mlp = architecture_results[
        (architecture_results["model_type"] == "mlp")
        & (architecture_results["backbone"] == best_backbone)
    ]
    if not same_backbone_mlp.empty:
        mlp_row = same_backbone_mlp.iloc[0]
        comparators.append(
            (
                str(mlp_row["name"]),
                Path(str(mlp_row["model_path"])),
                str(mlp_row["backbone"]),
            )
        )

    other_qnns = architecture_results[
        (architecture_results["model_type"] == "qnn")
        & (architecture_results["backbone"] != best_backbone)
    ].sort_values("test_f1", ascending=False)
    for _, row in other_qnns.iterrows():
        comparators.append(
            (
                str(row["name"]),
                Path(str(row["model_path"])),
                str(row["backbone"]),
            )
        )

    best_preds, labels = model_predictions(best_qnn_path, best_backbone)
    rows = []
    for comparator_name, comparator_path, backbone in comparators:
        preds, comp_labels = model_predictions(comparator_path, backbone)
        assert np.array_equal(labels, comp_labels)
        result = exact_mcnemar(best_preds, preds, labels)
        rows.append(
            {
                "reference_model": best_reference_name,
                "comparator_model": comparator_name,
                **result,
            }
        )
    return pd.DataFrame(rows)


def export_segmentation_tables(result: SegmentationTrainingResult) -> None:
    segmentation_table = pd.DataFrame(
        [
            {
                "model": "U-Net",
                "test_dice": result.metrics_test["dice"],
                "test_iou": result.metrics_test["iou"],
                "test_precision": result.metrics_test["precision"],
                "test_recall": result.metrics_test["recall"],
                "test_pixel_accuracy": result.metrics_test["pixel_accuracy"],
            },
            {
                "model": "Otsu threshold baseline",
                "test_dice": result.baseline_test["dice"],
                "test_iou": result.baseline_test["iou"],
                "test_precision": result.baseline_test["precision"],
                "test_recall": result.baseline_test["recall"],
                "test_pixel_accuracy": result.baseline_test["pixel_accuracy"],
            },
        ]
    )
    segmentation_table.to_csv(
        SEGMENTATION_ARTIFACT_ROOT / "tables" / "segmentation_results.csv",
        index=False,
    )
    (SEGMENTATION_ARTIFACT_ROOT / "tables" / "segmentation_results.tex").write_text(
        segmentation_table.to_latex(index=False, float_format=lambda value: f"{value:.4f}")
    )

    summary_table = pd.DataFrame(
        [
            {
                "train_examples": result.train_examples,
                "validation_examples": result.validation_examples,
                "test_examples": result.test_examples,
                "best_epoch": result.best_epoch,
                "dice_wilcoxon_pvalue": result.dice_wilcoxon_pvalue,
                "iou_wilcoxon_pvalue": result.iou_wilcoxon_pvalue,
            }
        ]
    )
    summary_table.to_csv(
        SEGMENTATION_ARTIFACT_ROOT / "tables" / "segmentation_hypothesis_summary.csv",
        index=False,
    )


def export_classification_hypothesis_table(df: pd.DataFrame) -> None:
    csv_path = SEGMENTATION_ARTIFACT_ROOT / "tables" / "classification_hypothesis_tests.csv"
    tex_path = SEGMENTATION_ARTIFACT_ROOT / "tables" / "classification_hypothesis_tests.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(df.to_latex(index=False, float_format=lambda value: f"{value:.4f}"))


def write_summary(
    segmentation_result: SegmentationTrainingResult,
    hypothesis_df: pd.DataFrame,
) -> None:
    best_row = hypothesis_df.sort_values("p_value").iloc[0]
    summary = {
        "segmentation_result": {
            "train_examples": segmentation_result.train_examples,
            "validation_examples": segmentation_result.validation_examples,
            "test_examples": segmentation_result.test_examples,
            "best_epoch": segmentation_result.best_epoch,
            "validation_metrics": segmentation_result.metrics_validation,
            "test_metrics": segmentation_result.metrics_test,
            "baseline_test_metrics": segmentation_result.baseline_test,
            "dice_wilcoxon_pvalue": segmentation_result.dice_wilcoxon_pvalue,
            "iou_wilcoxon_pvalue": segmentation_result.iou_wilcoxon_pvalue,
        },
        "most_significant_classification_comparison": best_row.to_dict(),
    }
    (SEGMENTATION_ARTIFACT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Segmentation and Hypothesis Test Summary",
        "",
        f"Segmentation subset sizes: train={segmentation_result.train_examples}, validation={segmentation_result.validation_examples}, test={segmentation_result.test_examples}",
        f"Best U-Net epoch: {segmentation_result.best_epoch}",
        f"U-Net test Dice: {segmentation_result.metrics_test['dice']:.4f}",
        f"U-Net test IoU: {segmentation_result.metrics_test['iou']:.4f}",
        f"Otsu baseline Dice: {segmentation_result.baseline_test['dice']:.4f}",
        f"Dice Wilcoxon p-value: {segmentation_result.dice_wilcoxon_pvalue:.6f}",
        "",
        "Most significant classification comparison:",
        f"- Comparator: {best_row['comparator_model']}",
        f"- McNemar p-value: {best_row['p_value']:.6f}",
    ]
    (SEGMENTATION_ARTIFACT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    set_seed()
    ensure_dirs()
    downloaded = download_shards()
    segmentation_result = train_segmentation_model(downloaded)
    export_segmentation_tables(segmentation_result)

    hypothesis_df = run_hypothesis_tests()
    export_classification_hypothesis_table(hypothesis_df)
    write_summary(segmentation_result, hypothesis_df)

    print("Finished segmentation experiment and hypothesis tests.")
    print(SEGMENTATION_ARTIFACT_ROOT / "tables" / "segmentation_results.csv")
    print(SEGMENTATION_ARTIFACT_ROOT / "tables" / "classification_hypothesis_tests.csv")


if __name__ == "__main__":
    main()
