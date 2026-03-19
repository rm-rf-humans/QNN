# QNN

Code for hybrid classical-quantum breast thermography experiments and related segmentation studies.

## Repository layout

- `experiments/`
  Reproducible experiment scripts for the rerun, robustness analysis, calibration study, IBM noise study, and LaTeX plot-data export.
- `train.py`, `test.py`
  Simple TensorFlow U-Net segmentation training and inference scripts from the original workflow.
- `Architectures_Heatmap.ipynb`, `H_QNNs.ipynb`, `QuantumDressedNet.ipynb`, `Quantum Tranfer Learning (DressedQuantumNet)`
  Research notebooks and notebook exports used during development.
- `normal_segmented_parts/`, `abnormal_segmented_parts/`
  Legacy segmented-image assets used by the original workflow.

## Notes

- IBM Runtime access is read from environment variables such as `QISKIT_IBM_TOKEN`, `IBM_QUANTUM_TOKEN`, or `QISKIT_IBM_RUNTIME_TOKEN`.
- Large generated outputs, local caches, virtual environments, and manuscript/LaTeX files are intentionally excluded from Git tracking.
