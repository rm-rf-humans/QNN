from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeLagosV2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qnn_experiments import (  # noqa: E402
    ARTIFACT_ROOT as QNN_ARTIFACT_ROOT,
    N_QUBITS,
    QuantumHead,
    cache_path_for_backbone,
    compute_metrics,
    set_seed,
)


ARTIFACT_ROOT = ROOT / "artifacts" / "ibm_quantum_study"
PAPER_DIR = (
    ROOT
    / "Utilizing_Hybrid_Quantum_Neural_Networks__H_QNNs__and_Complex_valued_Neural_Networks__CVNNs__for_Thermogram_Breast_Cancer_Classification"
)
PAPER_FIGURE_PATH = PAPER_DIR / "Figures" / "IBM_Lagos_QNN_Noise.png"
DEFAULT_CHECKPOINT = QNN_ARTIFACT_ROOT / "models" / "densenet121_depth=2_adamw_lr=0.001.pt"
DEFAULT_BACKBONE = "densenet121"
DEFAULT_SHOTS = 1024
DEFAULT_BATCH_SIZE = 24
SEED = 42


def ensure_dirs() -> None:
    for path in [ARTIFACT_ROOT, ARTIFACT_ROOT / "figures", ARTIFACT_ROOT / "tables"]:
        path.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_best_qnn_and_inputs(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    split_name: str = "test",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    input_dim = int(checkpoint["input_dim"])
    head = QuantumHead(
        input_dim=input_dim,
        q_depth=int(config["q_depth"]),
        dropout=float(config["dropout"]),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    feature_bundle = torch.load(
        cache_path_for_backbone(DEFAULT_BACKBONE),
        map_location="cpu",
    )
    features = feature_bundle[split_name]["features"]
    labels = feature_bundle[split_name]["labels"].numpy()

    with torch.no_grad():
        q_inputs = torch.tanh(head.pre_net(features)) * (math.pi / 2.0)
        ideal_q_out = head.quantum(q_inputs)
        ideal_logits = head.classifier(ideal_q_out)

    classifier_weight = head.classifier.weight.detach().cpu().numpy()
    classifier_bias = head.classifier.bias.detach().cpu().numpy()
    q_weights = head.quantum.q_weights.detach().cpu().numpy()

    return {
        "head": head,
        "config": config,
        "labels": labels,
        "q_inputs": q_inputs.detach().cpu().numpy(),
        "ideal_q_out": ideal_q_out.detach().cpu().numpy(),
        "ideal_logits": ideal_logits.detach().cpu().numpy(),
        "classifier_weight": classifier_weight,
        "classifier_bias": classifier_bias,
    }


def build_qiskit_template(q_weights: np.ndarray) -> tuple[QuantumCircuit, ParameterVector]:
    angles = ParameterVector("x", N_QUBITS)
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    for qubit, angle in enumerate(angles):
        qc.ry(angle, qubit)

    for layer in q_weights:
        for qubit, weight in enumerate(layer):
            qc.rx(float(weight), qubit)
        for control in range(N_QUBITS):
            qc.cx(control, (control + 1) % N_QUBITS)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc, angles


def transpile_template(
    backend: Any,
    q_weights: np.ndarray,
) -> tuple[QuantumCircuit, ParameterVector]:
    template, angles = build_qiskit_template(q_weights)
    transpiled_template = transpile(
        template,
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED,
    )
    return transpiled_template, angles


def bound_circuits(
    template: QuantumCircuit,
    parameters: ParameterVector,
    q_inputs: np.ndarray,
) -> list[QuantumCircuit]:
    circuits: list[QuantumCircuit] = []
    for sample in q_inputs:
        value_map = {param: float(sample[index]) for index, param in enumerate(parameters)}
        circuits.append(template.assign_parameters(value_map, inplace=False))
    return circuits


def z_expectations_from_counts(counts: dict[str, int], num_qubits: int) -> np.ndarray:
    total = sum(counts.values())
    expectations = np.zeros(num_qubits, dtype=np.float64)
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        for qubit in range(num_qubits):
            measured_bit = int(clean[num_qubits - 1 - qubit])
            expectations[qubit] += (1.0 if measured_bit == 0 else -1.0) * count
    if total == 0:
        return expectations
    return expectations / total


def run_backend_expectations(
    backend_for_run: Any,
    circuits: list[QuantumCircuit],
    shots: int,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(circuits), batch_size):
        batch = circuits[start : start + batch_size]
        job = backend_for_run.run(batch, shots=shots)
        result = job.result()
        for index in range(len(batch)):
            counts = result.get_counts(index)
            outputs.append(z_expectations_from_counts(counts, N_QUBITS))
    return np.stack(outputs, axis=0)


def layout_active_qubits(transpiled_template: QuantumCircuit) -> list[int]:
    physical_bits = transpiled_template.layout.initial_layout.get_physical_bits()
    active = [
        physical
        for physical, qubit in physical_bits.items()
        if getattr(qubit, "_register", None) is not None
        and qubit._register.name == "q"
        and qubit._register.size == N_QUBITS
    ]
    return sorted(active)


def active_cx_edges(transpiled_template: QuantumCircuit) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for instruction in transpiled_template.data:
        if instruction.operation.name != "cx":
            continue
        qubits = tuple(qubit._index for qubit in instruction.qubits)
        edges.add(tuple(sorted(qubits)))
    return sorted(edges)


def undirected_backend_edges(backend: Any) -> list[tuple[int, int]]:
    edges = {tuple(sorted(edge)) for edge in backend.coupling_map}
    return sorted(edges)


def average_cx_error(properties: Any, edge: tuple[int, int]) -> float:
    forward = []
    for directed in [edge, (edge[1], edge[0])]:
        try:
            forward.append(float(properties.gate_error("cx", directed)))
        except Exception:
            continue
    if not forward:
        return float("nan")
    return float(np.mean(forward))


def generate_backend_figure(
    backend: Any,
    transpiled_template: QuantumCircuit,
    results_df: pd.DataFrame,
) -> Path:
    properties = backend.properties()
    coords_raw = backend._conf_dict["coords"]
    coords = {
        qubit: (float(point[0]), -float(point[1]))
        for qubit, point in enumerate(coords_raw)
    }

    readout_errors = {qubit: float(properties.readout_error(qubit)) for qubit in range(backend.num_qubits)}
    all_edges = undirected_backend_edges(backend)
    edge_errors = {edge: average_cx_error(properties, edge) for edge in all_edges}
    active_qubits = layout_active_qubits(transpiled_template)
    active_edges = set(active_cx_edges(transpiled_template))

    fig = plt.figure(figsize=(11.4, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.35, hspace=0.35)
    ax_graph = fig.add_subplot(grid[:, 0])
    ax_readout = fig.add_subplot(grid[0, 1])
    ax_metrics = fig.add_subplot(grid[1, 1])

    edge_values = np.array([edge_errors[edge] for edge in all_edges], dtype=np.float64)
    edge_norm = colors.Normalize(vmin=float(np.nanmin(edge_values)), vmax=float(np.nanmax(edge_values)))
    edge_cmap = plt.get_cmap("Blues")
    node_values = np.array([readout_errors[q] for q in range(backend.num_qubits)], dtype=np.float64)
    node_norm = colors.Normalize(vmin=float(np.min(node_values)), vmax=float(np.max(node_values)))
    node_cmap = plt.get_cmap("magma_r")

    for edge in all_edges:
        x0, y0 = coords[edge[0]]
        x1, y1 = coords[edge[1]]
        is_active = edge in active_edges
        ax_graph.plot(
            [x0, x1],
            [y0, y1],
            color="black" if is_active else edge_cmap(edge_norm(edge_errors[edge])),
            linewidth=4.0 if is_active else 2.0,
            alpha=0.95 if is_active else 0.8,
            zorder=1,
        )
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax_graph.text(
            xm,
            ym + 0.14,
            f"{100.0 * edge_errors[edge]:.2f}%",
            fontsize=8,
            ha="center",
            va="center",
            color="black" if is_active else "#244B7A",
        )

    xs = [coords[q][0] for q in range(backend.num_qubits)]
    ys = [coords[q][1] for q in range(backend.num_qubits)]
    facecolors = [node_cmap(node_norm(readout_errors[q])) for q in range(backend.num_qubits)]
    edgecolors = ["gold" if q in active_qubits else "white" for q in range(backend.num_qubits)]
    linewidths = [3.0 if q in active_qubits else 1.2 for q in range(backend.num_qubits)]

    ax_graph.scatter(
        xs,
        ys,
        c=facecolors,
        s=620,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=2,
    )
    for qubit in range(backend.num_qubits):
        x, y = coords[qubit]
        ax_graph.text(x, y + 0.03, f"q{qubit}", fontsize=10, ha="center", va="center", color="white", weight="bold")
        ax_graph.text(
            x,
            y - 0.19,
            f"RO {100.0 * readout_errors[qubit]:.1f}%",
            fontsize=7.5,
            ha="center",
            va="center",
            color="black",
        )

    ax_graph.set_title("Generated IBM Lagos Calibration Map", fontsize=12)
    ax_graph.set_xticks([])
    ax_graph.set_yticks([])
    ax_graph.set_frame_on(False)
    ax_graph.set_aspect("equal")
    node_sm = plt.cm.ScalarMappable(norm=node_norm, cmap=node_cmap)
    cbar = fig.colorbar(node_sm, ax=ax_graph, fraction=0.04, pad=0.02)
    cbar.set_label("Readout error", fontsize=10)

    active_readout = [100.0 * readout_errors[q] for q in active_qubits]
    ax_readout.bar(
        [f"q{q}" for q in active_qubits],
        active_readout,
        color="#CC5A71",
    )
    ax_readout.set_title("Active-Qubit Readout Error", fontsize=11)
    ax_readout.set_ylabel("Percent")
    ax_readout.grid(axis="y", alpha=0.25)

    metric_names = ["Accuracy", "F1", "AUC"]
    ideal_row = results_df[results_df["execution"] == "ideal_statevector"].iloc[0]
    noisy_row = results_df[results_df["execution"] == "fake_lagos_noisy_sim"].iloc[0]
    metric_matrix = np.array(
        [
            [ideal_row["accuracy"], ideal_row["f1"], ideal_row["auc"]],
            [noisy_row["accuracy"], noisy_row["f1"], noisy_row["auc"]],
        ]
    )
    x = np.arange(len(metric_names))
    width = 0.36
    ax_metrics.bar(x - width / 2, metric_matrix[0], width=width, label="Ideal", color="#4C78A8")
    ax_metrics.bar(x + width / 2, metric_matrix[1], width=width, label="Noisy", color="#F58518")
    ax_metrics.set_xticks(x, metric_names)
    ax_metrics.set_ylim(0.0, 1.05)
    ax_metrics.set_title("Held-Out Performance Under Noise", fontsize=11)
    ax_metrics.grid(axis="y", alpha=0.25)
    ax_metrics.legend(fontsize=8, frameon=False)

    summary_text = (
        f"Mapped qubits: {active_qubits}\n"
        f"Active couplers: {sorted(active_edges)}\n"
        f"Transpiled depth: {transpiled_template.depth()}\n"
        f"Physical CX count: {int(transpiled_template.count_ops().get('cx', 0))}\n"
        f"Mean |Δ<Z>|: {noisy_row['mean_abs_quantum_shift']:.4f}\n"
        f"Mean |Δp|: {noisy_row['mean_abs_probability_shift']:.4f}\n"
        f"Shots: {int(noisy_row['shots'])}"
    )
    ax_metrics.text(
        1.02,
        0.5,
        summary_text,
        transform=ax_metrics.transAxes,
        fontsize=8.5,
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F6F6F6", "edgecolor": "#BBBBBB"},
    )

    output_path = ARTIFACT_ROOT / "figures" / "ibm_lagos_qnn_noise.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(output_path, PAPER_FIGURE_PATH)
    return output_path


def evaluate_backend(
    q_inputs: np.ndarray,
    ideal_q_out: np.ndarray,
    labels: np.ndarray,
    ideal_probs: np.ndarray,
    ideal_preds: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    q_weights: np.ndarray,
    backend_for_transpile: Any,
    backend_for_run: Any,
    shots: int,
    batch_size: int,
    execution_name: str,
) -> tuple[pd.DataFrame, QuantumCircuit]:
    transpiled_template, parameters = transpile_template(
        backend_for_transpile,
        q_weights=q_weights,
    )
    circuits = bound_circuits(transpiled_template, parameters, q_inputs)
    noisy_q_out = run_backend_expectations(
        backend_for_run=backend_for_run,
        circuits=circuits,
        shots=shots,
        batch_size=batch_size,
    )
    noisy_logits = noisy_q_out @ classifier_weight.T + classifier_bias
    noisy_probs = softmax(noisy_logits)[:, 1]
    noisy_preds = noisy_logits.argmax(axis=1)
    metrics = compute_metrics(labels, noisy_preds, noisy_probs)
    metrics["execution"] = execution_name
    metrics["shots"] = float(shots)
    metrics["mean_abs_quantum_shift"] = float(np.mean(np.abs(noisy_q_out - ideal_q_out)))
    metrics["mean_abs_probability_shift"] = float(np.mean(np.abs(noisy_probs - ideal_probs)))
    metrics["prediction_disagreement_rate"] = float(np.mean(noisy_preds != ideal_preds))
    metrics["transpiled_depth"] = float(transpiled_template.depth())
    metrics["transpiled_cx_count"] = float(transpiled_template.count_ops().get("cx", 0))
    metrics["active_qubits"] = json.dumps(layout_active_qubits(transpiled_template))
    metrics["active_edges"] = json.dumps(active_cx_edges(transpiled_template))
    return pd.DataFrame([metrics]), transpiled_template


def make_service(token: str | None, instance: str | None = None) -> QiskitRuntimeService:
    if token:
        return QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
            instance=instance,
        )
    return QiskitRuntimeService()


def run_real_ibm(
    q_inputs: np.ndarray,
    ideal_q_out: np.ndarray,
    labels: np.ndarray,
    ideal_probs: np.ndarray,
    ideal_preds: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    q_weights: np.ndarray,
    shots: int,
    batch_size: int,
    backend_name: str | None,
    instance: str | None,
) -> tuple[pd.DataFrame, QuantumCircuit, str]:
    token = (
        os.environ.get("QISKIT_IBM_TOKEN")
        or os.environ.get("IBM_QUANTUM_TOKEN")
        or os.environ.get("QISKIT_IBM_RUNTIME_TOKEN")
    )
    if not token:
        raise RuntimeError("IBM Runtime token not found in environment.")

    service = make_service(token=token, instance=instance)
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(min_num_qubits=7, operational=True, simulator=False)

    results_df, transpiled_template = evaluate_backend(
        q_inputs=q_inputs,
        ideal_q_out=ideal_q_out,
        labels=labels,
        ideal_probs=ideal_probs,
        ideal_preds=ideal_preds,
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        q_weights=q_weights,
        backend_for_transpile=backend,
        backend_for_run=backend,
        shots=shots,
        batch_size=batch_size,
        execution_name=f"ibm_backend::{backend.name}",
    )
    return results_df, transpiled_template, backend.name


def main() -> None:
    parser = argparse.ArgumentParser(description="IBM Quantum noise study for the best trained H-QNN.")
    parser.add_argument(
        "--mode",
        choices=["fake_lagos_noisy_sim", "ibm_backend"],
        default="fake_lagos_noisy_sim",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--backend-name", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(SEED)

    global loaded
    loaded = load_best_qnn_and_inputs()
    labels = loaded["labels"]
    ideal_logits = loaded["ideal_logits"]
    ideal_probs = softmax(ideal_logits)[:, 1]
    ideal_preds = ideal_logits.argmax(axis=1)
    ideal_metrics = compute_metrics(labels, ideal_preds, ideal_probs)
    ideal_metrics["execution"] = "ideal_statevector"
    ideal_metrics["shots"] = 0.0
    ideal_metrics["mean_abs_quantum_shift"] = 0.0
    ideal_metrics["mean_abs_probability_shift"] = 0.0
    ideal_metrics["prediction_disagreement_rate"] = 0.0
    ideal_metrics["transpiled_depth"] = 0.0
    ideal_metrics["transpiled_cx_count"] = 0.0
    ideal_metrics["active_qubits"] = "[]"
    ideal_metrics["active_edges"] = "[]"
    results_frames = [pd.DataFrame([ideal_metrics])]

    if args.mode == "fake_lagos_noisy_sim":
        backend = FakeLagosV2()
        simulator = AerSimulator.from_backend(backend)
        noisy_results, transpiled_template = evaluate_backend(
            q_inputs=loaded["q_inputs"],
            ideal_q_out=loaded["ideal_q_out"],
            labels=labels,
            ideal_probs=ideal_probs,
            ideal_preds=ideal_preds,
            classifier_weight=loaded["classifier_weight"],
            classifier_bias=loaded["classifier_bias"],
            q_weights=loaded["head"].quantum.q_weights.detach().cpu().numpy(),
            backend_for_transpile=backend,
            backend_for_run=simulator,
            shots=args.shots,
            batch_size=args.batch_size,
            execution_name="fake_lagos_noisy_sim",
        )
        results_frames.append(noisy_results)
        backend_name = backend.name
        figure_path = generate_backend_figure(backend, transpiled_template, pd.concat(results_frames, ignore_index=True))
    else:
        ibm_results, transpiled_template, backend_name = run_real_ibm(
            q_inputs=loaded["q_inputs"],
            ideal_q_out=loaded["ideal_q_out"],
            labels=labels,
            ideal_probs=ideal_probs,
            ideal_preds=ideal_preds,
            classifier_weight=loaded["classifier_weight"],
            classifier_bias=loaded["classifier_bias"],
            q_weights=loaded["head"].quantum.q_weights.detach().cpu().numpy(),
            shots=args.shots,
            batch_size=args.batch_size,
            backend_name=args.backend_name,
            instance=args.instance,
        )
        results_frames.append(ibm_results)
        if backend_name.lower().endswith("lagos"):
            figure_backend = FakeLagosV2()
            figure_path = generate_backend_figure(figure_backend, transpiled_template, pd.concat(results_frames, ignore_index=True))
        else:
            figure_path = ARTIFACT_ROOT / "figures" / "ibm_lagos_qnn_noise.png"

    results_df = pd.concat(results_frames, ignore_index=True)
    results_csv = ARTIFACT_ROOT / "tables" / "hardware_noise_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "mode": args.mode,
        "shots": args.shots,
        "backend_name": backend_name,
        "figure_path": str(figure_path),
        "paper_figure_path": str(PAPER_FIGURE_PATH),
        "results": results_df.to_dict(orient="records"),
    }
    (ARTIFACT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# IBM Quantum Noise Study",
        "",
        f"- Mode: `{args.mode}`",
        f"- Backend: `{backend_name}`",
        f"- Shots: `{args.shots}`",
        f"- Figure: `{figure_path}`",
        "",
        "## Results",
    ]
    for row in results_df.to_dict(orient="records"):
        lines.append(
            f"- {row['execution']}: acc={row['accuracy']:.4f}, F1={row['f1']:.4f}, AUC={row['auc']:.4f}, mean |Δ<Z>|={row['mean_abs_quantum_shift']:.4f}, mean |Δp|={row['mean_abs_probability_shift']:.4f}"
        )
    (ARTIFACT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")

    print(results_csv)
    print(figure_path)


if __name__ == "__main__":
    main()
