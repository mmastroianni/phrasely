# Phrasely – Phrase Clustering and Embedding Pipeline (GPU‑Accelerated)
![CI](https://github.com/mmastroianni/phrasely/actions/workflows/ci.yml/badge.svg)

## 🧠 Overview
Phrasely is a modular, GPU‑accelerated pipeline for clustering and analyzing large phrase datasets. It uses **Sentence Transformers** for embeddings, **SVD/PCA** for dimensionality reduction, and **cuML HDBSCAN** for clustering on NVIDIA GPUs via the **RAPIDS** suite. The project is structured for clean testability, reproducibility, and eventual open‑source release.

---

## ⚙️ Features
- Modular pipeline: load → embed → reduce → cluster → medoid selection
- GPU acceleration with RAPIDS (cuML / cuDF / CuPy)
- CPU fallback for non‑GPU systems
- Test‑driven development layout (`src/` + `tests/`)
- Ready for PyCharm and Jupyter integration
- Fully reproducible via micromamba environments

---

## 🧱 Directory Structure
```
phrasely/
├── Makefile
├── pyproject.toml
├── pytest.ini
├── environment.yaml              # GPU-default (RAPIDS)
├── environment-cpu.yaml          # CPU fallback
├── src/
│   └── phrasely/
│       ├── pipeline.py
│       ├── utils/
│       ├── data_loading/
│       ├── embeddings/
│       ├── reduction/
│       ├── clustering/
│       └── medoids/
├── tests/
├── notebooks/
└── README.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Create Environment (GPU Default)
```bash
micromamba create -n phrasely -f environment.yaml
micromamba activate phrasely
```
If using a CPU‑only system:
```bash
micromamba create -n phrasely-cpu -f environment-cpu.yaml
micromamba activate phrasely-cpu
```

### 2️⃣ Verify GPU Access
```bash
python tests/test_gpu_setup.py
```
You should see your GPU name and a successful cuML HDBSCAN run.

---

## 🧩 PyCharm Integration
1. Open **Settings → Project → Python Interpreter**.
2. Add an interpreter: **Existing Environment**.
3. Point to your micromamba Python binary, e.g.:
   ```
   /home/michael/micromamba/envs/phrasely/bin/python
   ```
4. Rename it to “phrasely‑gpu (micromamba)” for clarity.

PyCharm will index all GPU‑enabled packages (cuML, cuDF, CuPy, etc.).

---

## 📓 Jupyter Kernel Setup
To use the environment in notebooks:
```bash
micromamba activate phrasely
micromamba install ipykernel
python -m ipykernel install --user --name phrasely --display-name "Phrasely (GPU)"
```
Then select **Phrasely (GPU)** from the kernel list in PyCharm or JupyterLab.

---

## 🧠 Pipeline Overview
Each step is modular and unit‑tested:
- **`data_loading`** → Load phrases from CSV or other sources.
- **`embeddings`** → Generate sentence embeddings (Sentence Transformers).
- **`reduction`** → Dimensionality reduction (SVD / PCA).
- **`clustering`** → HDBSCAN (GPU via cuML or CPU fallback).
- **`medoids`** → Select representative phrases per cluster.

Run the pipeline:
```bash
python -m phrasely.pipeline --input data/sample_phrases.csv --output results.csv
```

---

## 🧰 Developer Notes
- Use `pytest -v` to run all tests.
- Use `make format` to auto-format the codebase with **Black** and **isort**.
- Use `make lint` to run **flake8** (style) and **mypy** (type checking).
- Use `make install` to generate editable dev install.
- Use `make gpu-test` (optional) to run GPU sanity check.

---

## 📄 License
MIT © 2025 Michael Mastroianni
