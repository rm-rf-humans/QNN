from __future__ import annotations

import copy
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "qnn_experiments"
CLASSIFICATION_DATA_ROOT = ROOT / "artifacts" / "raw_dmr_ir_unet_classification"
NORMAL_DIR = CLASSIFICATION_DATA_ROOT / "normal"
ABNORMAL_DIR = CLASSIFICATION_DATA_ROOT / "abnormal"

SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 2
N_QUBITS = 4
DEFAULT_EPOCHS = 8
DEFAULT_PATIENCE = 3
TRANSFER_BATCH_SIZE = 32
HEAD_BATCH_SIZE = 32
CLASSIFICATION_IMAGES_PER_CLASS = 500
DATASET_TAG = "raw_dmr_ir_unet_v1"
HF_DATASET_ID = "SemilleroCV/DMR-IR"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_feature_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_head_device() -> torch.device:
    return torch.device("cpu")


def ensure_dirs() -> None:
    for path in [
        ARTIFACT_ROOT,
        ARTIFACT_ROOT / "cache",
        ARTIFACT_ROOT / "figures",
        ARTIFACT_ROOT / "models",
        ARTIFACT_ROOT / "tables",
        CLASSIFICATION_DATA_ROOT,
        NORMAL_DIR,
        ABNORMAL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def choose_segmentation_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_thermal_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        image_array = np.array(image, dtype=np.float32)

    image_resized = np.array(
        Image.fromarray(image_array, mode="F").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR),
        dtype=np.float32,
    )
    image_resized -= image_resized.min()
    image_resized /= image_resized.max() + 1e-8
    return image_resized


def prepare_unet_classification_dataset(force_rebuild: bool = False) -> Path:
    metadata_path = CLASSIFICATION_DATA_ROOT / "metadata.csv"
    summary_path = CLASSIFICATION_DATA_ROOT / "summary.json"
    if not force_rebuild and metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        class_counts = metadata["class_name"].value_counts().to_dict()
        if class_counts.get("normal", 0) == CLASSIFICATION_IMAGES_PER_CLASS and class_counts.get(
            "abnormal", 0
        ) == CLASSIFICATION_IMAGES_PER_CLASS:
            if not summary_path.exists():
                summary_path.write_text(
                    json.dumps(
                        {
                            "dataset_tag": DATASET_TAG,
                            "images_per_class": CLASSIFICATION_IMAGES_PER_CLASS,
                            "selected_counts": {
                                "normal": int(class_counts.get("normal", 0)),
                                "abnormal": int(class_counts.get("abnormal", 0)),
                            },
                            "total_images": int(len(metadata)),
                            "class_distribution": class_counts,
                            "source_split_distribution": [
                                {
                                    "source_split": str(source_split),
                                    "class_name": str(class_name),
                                    "count": int(count),
                                }
                                for (source_split, class_name), count in metadata.groupby(
                                    ["source_split", "class_name"]
                                ).size().items()
                            ],
                        },
                        indent=2,
                    )
                )
            return metadata_path

    from datasets import Image as HFImage
    from datasets import load_dataset

    from experiments.segmentation_and_hypothesis_tests import (  # local import avoids circular import at module load
        SEGMENTATION_ARTIFACT_ROOT,
        SmallUNet,
    )

    checkpoint_path = SEGMENTATION_ARTIFACT_ROOT / "models" / "dmr_ir_subset_unet.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing segmentation checkpoint required for the integrated pipeline: {checkpoint_path}"
        )

    if CLASSIFICATION_DATA_ROOT.exists():
        shutil.rmtree(CLASSIFICATION_DATA_ROOT)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    segmentation_model = SmallUNet()
    segmentation_model.load_state_dict(checkpoint["model_state_dict"])
    segmentation_device = choose_segmentation_device()
    segmentation_model = segmentation_model.to(segmentation_device).eval()

    selected_counts = {0: 0, 1: 0}
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for split_name in ("train", "validation", "test"):
            dataset_stream = load_dataset(HF_DATASET_ID, split=split_name, streaming=True)
            dataset_stream = dataset_stream.cast_column("image", HFImage(decode=False))

            for row in dataset_stream:
                label = int(row["label"])
                if selected_counts[label] >= CLASSIFICATION_IMAGES_PER_CLASS:
                    continue

                image_info = row["image"]
                image_bytes = image_info["bytes"]
                if image_bytes is None:
                    continue

                normalized_image = preprocess_thermal_image(image_bytes)
                input_tensor = (
                    torch.from_numpy(normalized_image)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(segmentation_device)
                )
                mask_logits = segmentation_model(input_tensor)
                mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze().cpu().numpy()
                masked_image = np.clip(normalized_image * mask, 0.0, 1.0)
                output_image = Image.fromarray(np.round(masked_image * 255.0).astype(np.uint8), mode="L")

                class_name = "normal" if label == 0 else "abnormal"
                output_dir = NORMAL_DIR if label == 0 else ABNORMAL_DIR
                stem = Path(image_info["path"] or f"sample_{selected_counts[label]:04d}.tiff").stem
                output_name = (
                    f"{split_name}_{row['patient_id']}_{row['record']}_{int(row['view'])}_{stem}.png"
                )
                output_path = output_dir / output_name
                output_image.save(output_path)

                records.append(
                    {
                        "path": str(output_path),
                        "class_name": class_name,
                        "label": label,
                        "source_split": split_name,
                        "source_image_path": image_info["path"],
                        "patient_id": row["patient_id"],
                        "record": row["record"],
                        "view": int(row["view"]),
                        "mask_fraction": float(mask.mean()),
                    }
                )
                selected_counts[label] += 1

                if all(count >= CLASSIFICATION_IMAGES_PER_CLASS for count in selected_counts.values()):
                    break

            if all(count >= CLASSIFICATION_IMAGES_PER_CLASS for count in selected_counts.values()):
                break

    if not all(count >= CLASSIFICATION_IMAGES_PER_CLASS for count in selected_counts.values()):
        raise RuntimeError(
            "Unable to construct the balanced raw DMR-IR U-Net dataset with "
            f"{CLASSIFICATION_IMAGES_PER_CLASS} images per class. Counts: {selected_counts}"
        )

    metadata = pd.DataFrame(records).sort_values(["label", "path"]).reset_index(drop=True)
    metadata.to_csv(metadata_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "dataset_tag": DATASET_TAG,
                "images_per_class": CLASSIFICATION_IMAGES_PER_CLASS,
                "selected_counts": selected_counts,
                "total_images": int(len(metadata)),
                "class_distribution": metadata["class_name"].value_counts().to_dict(),
                "source_split_distribution": [
                    {
                        "source_split": str(source_split),
                        "class_name": str(class_name),
                        "count": int(count),
                    }
                    for (source_split, class_name), count in metadata.groupby(
                        ["source_split", "class_name"]
                    ).size().items()
                ],
            },
            indent=2,
        )
    )
    return metadata_path


def gather_image_records() -> pd.DataFrame:
    prepare_unet_classification_dataset()
    records: list[dict[str, Any]] = []
    for label, class_name, class_dir in [
        (0, "normal", NORMAL_DIR),
        (1, "abnormal", ABNORMAL_DIR),
    ]:
        image_paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for image_path in image_paths:
            records.append(
                {
                    "path": str(image_path),
                    "label": label,
                    "class_name": class_name,
                    "image_name": image_path.name,
                }
            )
    return pd.DataFrame(records)


def build_splits(
    seed: int = SEED,
    artifact_root: Path = ARTIFACT_ROOT,
    write_summary: bool = True,
) -> dict[str, pd.DataFrame]:
    data = gather_image_records()
    train_df, temp_df = train_test_split(
        data,
        test_size=0.30,
        random_state=seed,
        stratify=data["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )
    split_map = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    summary = []
    for split_name, split_df in split_map.items():
        counts = split_df["class_name"].value_counts().to_dict()
        summary.append(
            {
                "split": split_name,
                "total": len(split_df),
                "normal": counts.get("normal", 0),
                "abnormal": counts.get("abnormal", 0),
            }
        )
    if write_summary:
        (artifact_root / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(artifact_root / "tables" / "dataset_split.csv", index=False)
    return split_map


class ImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: transforms.Compose,
        color_mode: str,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.color_mode = color_mode

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"])
        if self.color_mode == "rgb":
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        tensor = self.transform(image)
        return tensor, int(row["label"]), row["path"]


def transfer_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class FrozenBackbone(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

        if name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone.fc = nn.Identity()
            self.feature_dim = 512
            self.target_layer = backbone.layer4[-1].conv2
        elif name == "densenet121":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            backbone.classifier = nn.Identity()
            self.feature_dim = 1024
            self.target_layer = None
        elif name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            backbone.classifier = nn.Identity()
            self.feature_dim = 576
            self.target_layer = None
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.backbone = backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def cache_path_for_backbone(
    backbone_name: str,
    cache_tag: str | None = None,
    artifact_root: Path = ARTIFACT_ROOT,
) -> Path:
    full_cache_tag = DATASET_TAG if cache_tag is None else f"{DATASET_TAG}_{cache_tag}"
    suffix = f"_{full_cache_tag}" if full_cache_tag else ""
    return artifact_root / "cache" / f"{backbone_name}{suffix}_features.pt"


def extract_transfer_features(
    backbone_name: str,
    splits: dict[str, pd.DataFrame],
    batch_size: int = TRANSFER_BATCH_SIZE,
    cache_tag: str | None = None,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, dict[str, Any]]:
    cache_path = cache_path_for_backbone(
        backbone_name,
        cache_tag=cache_tag,
        artifact_root=artifact_root,
    )
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    device = choose_feature_device()
    model = FrozenBackbone(backbone_name).to(device)
    bundle: dict[str, dict[str, Any]] = {}

    for split_name, split_df in splits.items():
        loader = DataLoader(
            ImageDataset(split_df, transfer_transform(), color_mode="rgb"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        paths: list[str] = []

        with torch.no_grad():
            for images, batch_labels, batch_paths in loader:
                images = images.to(device)
                batch_features = model(images).detach().cpu()
                features.append(batch_features)
                labels.append(batch_labels.cpu())
                paths.extend(batch_paths)

        bundle[split_name] = {
            "features": torch.cat(features, dim=0),
            "labels": torch.cat(labels, dim=0).long(),
            "paths": paths,
        }

    torch.save(bundle, cache_path)
    return bundle


class VariationalQuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = N_QUBITS, depth: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.q_weights = nn.Parameter(0.02 * torch.randn(depth, n_qubits))
        self.device = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(self.device, interface="torch", diff_method="adjoint")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [torch.stack(self.circuit(sample, self.q_weights)) for sample in x]
        return torch.stack(outputs, dim=0)


class QuantumHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        q_depth: int,
        dropout: float,
        n_qubits: int = N_QUBITS,
    ) -> None:
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, n_qubits),
        )
        self.quantum = VariationalQuantumLayer(n_qubits=n_qubits, depth=q_depth)
        self.classifier = nn.Linear(n_qubits, NUM_CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        q_in = torch.tanh(self.pre_net(features)) * (math.pi / 2.0)
        q_out = self.quantum(q_in)
        return self.classifier(q_out)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass
class ExperimentConfig:
    name: str
    study: str
    model_type: str
    backbone: str
    q_depth: int = 0
    optimizer: str = "adamw"
    lr: float = 1e-3
    dropout: float = 0.2
    weight_decay: float = 1e-4
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = HEAD_BATCH_SIZE


def get_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def create_feature_loaders(
    feature_bundle: dict[str, dict[str, Any]],
    batch_size: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split_name, split_data in feature_bundle.items():
        dataset = TensorDataset(split_data["features"], split_data["labels"])
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=0,
        )
    return loaders


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
        if total == 0:
            return 0.0, 0.0
        phat = successes / total
        denominator = 1.0 + (z**2) / total
        center = (phat + (z**2) / (2.0 * total)) / denominator
        margin = (
            z
            * math.sqrt((phat * (1.0 - phat) / total) + (z**2) / (4.0 * (total**2)))
            / denominator
        )
        return max(0.0, center - margin), min(1.0, center + margin)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(labels, probs)
    n_examples = int(len(labels))
    n_correct = int((labels == predictions).sum())
    accuracy_ci_low, accuracy_ci_high = wilson_interval(n_correct, n_examples)
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "auc": float(auc),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_examples": float(n_examples),
        "n_correct": float(n_correct),
        "accuracy_ci_low": float(accuracy_ci_low),
        "accuracy_ci_high": float(accuracy_ci_high),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_predictions)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np, preds_np, probs_np)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def train_head_experiment(
    config: ExperimentConfig,
    feature_bundle: dict[str, dict[str, Any]],
    input_dim: int,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    device = choose_head_device()
    loaders = create_feature_loaders(feature_bundle, batch_size=config.batch_size)

    if config.model_type == "qnn":
        model = QuantumHead(
            input_dim=input_dim,
            q_depth=config.q_depth,
            dropout=config.dropout,
        )
    elif config.model_type == "mlp":
        model = MLPHead(input_dim=input_dim, dropout=config.dropout)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    model = model.to(device)
    optimizer = get_optimizer(config.optimizer, model, config.lr, config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    patience_counter = 0
    history: list[dict[str, float]] = []
    started_at = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for features, labels in loaders["train"]:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(loaders["train"].dataset)
        train_metrics = evaluate_model(model, loaders["train"], device)
        val_metrics = evaluate_model(model, loaders["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_metrics["f1"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    model.load_state_dict(best_state)
    train_metrics = evaluate_model(model, loaders["train"], device)
    val_metrics = evaluate_model(model, loaders["val"], device)
    test_metrics = evaluate_model(model, loaders["test"], device)
    finished_at = time.time()

    experiment_id = config.name.replace(" ", "_").lower()
    model_path = artifact_root / "models" / f"{experiment_id}.pt"
    history_path = artifact_root / "models" / f"{experiment_id}_history.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
        },
        model_path,
    )
    history_path.write_text(json.dumps(history, indent=2))

    result = {
        **asdict(config),
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_specificity": test_metrics["specificity"],
        "test_auc": test_metrics["auc"],
        "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "test_tp": test_metrics["tp"],
        "test_n_examples": test_metrics["n_examples"],
        "test_n_correct": test_metrics["n_correct"],
        "test_accuracy_ci_low": test_metrics["accuracy_ci_low"],
        "test_accuracy_ci_high": test_metrics["accuracy_ci_high"],
        "test_loss": test_metrics["loss"],
        "train_seconds": finished_at - started_at,
        "model_path": str(model_path),
        "history_path": str(history_path),
    }
    return result


def plot_architecture_results(results: pd.DataFrame) -> Path:
    figure_path = ARTIFACT_ROOT / "figures" / "architecture_comparison.png"
    plot_df = results.sort_values("test_f1", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(plot_df["name"], plot_df["test_accuracy"], color="#3b82f6")
    axes[0].set_title("Test Accuracy by Model")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_xlim(0.0, 1.0)

    axes[1].barh(plot_df["name"], plot_df["test_f1"], color="#10b981")
    axes[1].set_title("Test F1 by Model")
    axes[1].set_xlabel("F1 Score")
    axes[1].set_xlim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_tuning_heatmap(results: pd.DataFrame) -> Path:
    figure_path = ARTIFACT_ROOT / "figures" / "circuit_tuning_heatmap.png"
    heatmap_df = results.copy()
    heatmap_df["column"] = heatmap_df.apply(
        lambda row: f"{row['optimizer']} | lr={row['lr']}",
        axis=1,
    )
    pivot = heatmap_df.pivot(index="q_depth", columns="column", values="val_f1")
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(index) for index in pivot.index])
    ax.set_xlabel("Optimizer and Learning Rate")
    ax.set_ylabel("Quantum Circuit Depth")
    ax.set_title("Validation F1 for Circuit Optimization Sweep")

    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.values[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


class ResNet18QNNWrapper(nn.Module):
    def __init__(self, head_state: dict[str, torch.Tensor], dropout: float, q_depth: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        self.backbone = backbone
        self.head = QuantumHead(input_dim=feature_dim, q_depth=q_depth, dropout=dropout)
        self.head.load_state_dict(head_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def gradcam_from_model(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
) -> tuple[np.ndarray, int, float]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    image_tensor = image_tensor.clone().requires_grad_(True)
    handle_forward = target_layer.register_forward_hook(forward_hook)

    model.eval()
    logits = model(image_tensor)
    target_class = int(logits.argmax(dim=1).item())
    target_score = logits[0, target_class]
    model.zero_grad()
    target_score.backward()

    handle_forward.remove()

    if not gradients:
        raise RuntimeError("Grad-CAM gradient capture failed for the selected layer.")

    grad = gradients[-1]
    activation = activations[-1]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    probability = float(torch.softmax(logits, dim=1)[0, target_class].item())
    return cam, target_class, probability


def render_gradcam_figure(
    image_path: str,
    cam: np.ndarray,
    predicted_class: int,
    probability: float,
) -> Path:
    output_path = ARTIFACT_ROOT / "figures" / "resnet18_qnn_gradcam.png"
    original = Image.open(image_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(original)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Segmented Thermogram")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="inferno")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(image_np, cmap="gray")
    axes[2].imshow(cam, cmap="inferno", alpha=0.45)
    label_name = "abnormal" if predicted_class == 1 else "normal"
    axes[2].set_title(f"Overlay: {label_name} ({probability:.2%})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_resnet18_gradcam(
    feature_bundle: dict[str, dict[str, Any]],
    architecture_results: pd.DataFrame,
) -> Path | None:
    resnet_rows = architecture_results[
        (architecture_results["backbone"] == "resnet18")
        & (architecture_results["model_type"] == "qnn")
    ]
    if resnet_rows.empty:
        return None

    best_row = resnet_rows.sort_values("val_f1", ascending=False).iloc[0]
    checkpoint = torch.load(best_row["model_path"], map_location="cpu")
    model = ResNet18QNNWrapper(
        head_state=checkpoint["state_dict"],
        dropout=float(best_row["dropout"]),
        q_depth=int(best_row["q_depth"]),
    )

    head = QuantumHead(input_dim=512, q_depth=int(best_row["q_depth"]), dropout=float(best_row["dropout"]))
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    features = feature_bundle["test"]["features"]
    labels = feature_bundle["test"]["labels"].numpy()
    with torch.no_grad():
        logits = head(features)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()

    candidate_indices = np.where((labels == 1) & (preds == 1))[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(labels))

    selected_idx = candidate_indices[np.argmax(probs[candidate_indices])]
    image_path = feature_bundle["test"]["paths"][int(selected_idx)]
    image = Image.open(image_path).convert("RGB")
    tensor = transfer_transform()(image).unsqueeze(0)
    cam, predicted_class, probability = gradcam_from_model(
        model=model,
        image_tensor=tensor,
        target_layer=model.backbone.layer4[-1].conv2,
    )
    return render_gradcam_figure(
        image_path=image_path,
        cam=cam,
        predicted_class=predicted_class,
        probability=probability,
    )


def export_table(df: pd.DataFrame, csv_name: str, tex_name: str, columns: list[str]) -> None:
    csv_path = ARTIFACT_ROOT / "tables" / csv_name
    tex_path = ARTIFACT_ROOT / "tables" / tex_name
    df.to_csv(csv_path, index=False)
    latex_df = df[columns].copy()
    tex_path.write_text(
        latex_df.to_latex(index=False, float_format=lambda value: f"{value:.4f}")
    )


def write_summary(
    architecture_results: pd.DataFrame,
    tuning_results: pd.DataFrame,
    figure_paths: dict[str, str],
) -> None:
    best_arch = architecture_results.sort_values("test_f1", ascending=False).iloc[0]
    best_tuning = tuning_results.sort_values("test_f1", ascending=False).iloc[0]

    summary = {
        "best_architecture": best_arch.to_dict(),
        "best_tuning": best_tuning.to_dict(),
        "figures": figure_paths,
    }
    (ARTIFACT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Quantum Experiment Summary",
        "",
        f"Best architecture: {best_arch['name']}",
        f"- Test accuracy: {best_arch['test_accuracy']:.4f}",
        f"- Test F1: {best_arch['test_f1']:.4f}",
        f"- Test AUC: {best_arch['test_auc']:.4f}",
        "",
        f"Best circuit tuning run: {best_tuning['name']}",
        f"- Depth: {int(best_tuning['q_depth'])}",
        f"- Optimizer: {best_tuning['optimizer']}",
        f"- Learning rate: {best_tuning['lr']}",
        f"- Test F1: {best_tuning['test_f1']:.4f}",
        "",
        "Figures:",
    ]
    for label, path in figure_paths.items():
        lines.append(f"- {label}: {path}")
    (ARTIFACT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def architecture_experiments() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="ResNet18 + QNN",
            study="architecture",
            model_type="qnn",
            backbone="resnet18",
            q_depth=2,
            optimizer="adamw",
            lr=1e-3,
        ),
        ExperimentConfig(
            name="DenseNet121 + QNN",
            study="architecture",
            model_type="qnn",
            backbone="densenet121",
            q_depth=2,
            optimizer="adamw",
            lr=1e-3,
        ),
        ExperimentConfig(
            name="DenseNet121 + MLP",
            study="architecture",
            model_type="mlp",
            backbone="densenet121",
            optimizer="adamw",
            lr=1e-3,
        ),
        ExperimentConfig(
            name="MobileNetV3-Small + QNN",
            study="architecture",
            model_type="qnn",
            backbone="mobilenet_v3_small",
            q_depth=2,
            optimizer="adamw",
            lr=1e-3,
        ),
        ExperimentConfig(
            name="ResNet18 + MLP",
            study="architecture",
            model_type="mlp",
            backbone="resnet18",
            optimizer="adamw",
            lr=1e-3,
        ),
    ]


def tuning_experiments(best_backbone: str) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    for depth, optimizer_name, lr in [
        (1, "adam", 1e-3),
        (2, "adam", 1e-3),
        (4, "adam", 1e-3),
        (2, "adamw", 1e-3),
        (2, "adam", 5e-4),
        (4, "adamw", 5e-4),
    ]:
        configs.append(
            ExperimentConfig(
                name=f"{best_backbone} depth={depth} {optimizer_name} lr={lr}",
                study="tuning",
                model_type="qnn",
                backbone=best_backbone,
                q_depth=depth,
                optimizer=optimizer_name,
                lr=lr,
            )
        )
    return configs


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    prepare_unet_classification_dataset()
    splits = build_splits(seed=SEED)

    results: list[dict[str, Any]] = []
    feature_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for config in architecture_experiments():
        if config.backbone not in feature_cache:
            feature_cache[config.backbone] = extract_transfer_features(config.backbone, splits)
        input_dim = feature_cache[config.backbone]["train"]["features"].shape[1]
        result = train_head_experiment(config, feature_cache[config.backbone], input_dim=input_dim)
        results.append(result)

    results_df = pd.DataFrame(results)
    architecture_results = results_df[results_df["study"] == "architecture"].copy()
    best_qnn_row = architecture_results[architecture_results["model_type"] == "qnn"].sort_values(
        "val_f1", ascending=False
    ).iloc[0]
    best_backbone = str(best_qnn_row["backbone"])

    tuning_results_list: list[dict[str, Any]] = []
    for config in tuning_experiments(best_backbone):
        if config.backbone not in feature_cache:
            feature_cache[config.backbone] = extract_transfer_features(config.backbone, splits)
        input_dim = feature_cache[config.backbone]["train"]["features"].shape[1]
        result = train_head_experiment(config, feature_cache[config.backbone], input_dim=input_dim)
        tuning_results_list.append(result)
        results.append(result)

    tuning_results = pd.DataFrame(tuning_results_list)
    all_results = pd.DataFrame(results).sort_values(["study", "test_f1"], ascending=[True, False])

    architecture_figure = plot_architecture_results(architecture_results)
    tuning_figure = plot_tuning_heatmap(tuning_results)
    gradcam_figure = generate_resnet18_gradcam(feature_cache["resnet18"], architecture_results)

    export_table(
        architecture_results.sort_values("test_f1", ascending=False),
        csv_name="architecture_results.csv",
        tex_name="architecture_results.tex",
        columns=[
            "name",
            "backbone",
            "model_type",
            "test_accuracy",
            "test_f1",
            "test_recall",
            "test_specificity",
            "test_auc",
        ],
    )
    export_table(
        tuning_results.sort_values("test_f1", ascending=False),
        csv_name="tuning_results.csv",
        tex_name="tuning_results.tex",
        columns=[
            "name",
            "q_depth",
            "optimizer",
            "lr",
            "val_f1",
            "test_accuracy",
            "test_f1",
            "test_auc",
        ],
    )
    all_results.to_csv(ARTIFACT_ROOT / "tables" / "all_results.csv", index=False)

    figure_paths = {
        "architecture_comparison": str(architecture_figure),
        "circuit_tuning_heatmap": str(tuning_figure),
    }
    if gradcam_figure is not None:
        figure_paths["resnet18_qnn_gradcam"] = str(gradcam_figure)
    write_summary(architecture_results, tuning_results, figure_paths)

    print("Finished quantum experiment suite.")
    print(f"Best QNN backbone: {best_backbone}")
    print(f"Architecture results: {ARTIFACT_ROOT / 'tables' / 'architecture_results.csv'}")
    print(f"Tuning results: {ARTIFACT_ROOT / 'tables' / 'tuning_results.csv'}")
    print(f"Figures: {ARTIFACT_ROOT / 'figures'}")


if __name__ == "__main__":
    main()
