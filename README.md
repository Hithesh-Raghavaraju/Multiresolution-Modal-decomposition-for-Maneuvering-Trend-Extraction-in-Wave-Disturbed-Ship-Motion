# Multiresolution Koopman Framework for Ship Motion Denoising & Prediction

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Field](https://img.shields.io/badge/Field-Control%20Systems%20%2F%20Ocean%20Engineering-green)](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition)

[cite_start]An operator-theoretic, data-driven framework designed to isolate low-frequency ship maneuvering trends from high-frequency wave-induced noise  [cite: 14-16, 55, 376-377]. [cite_start]This repository contains a two-stage implementation: **Multiresolution Dynamic Mode Decomposition (mrDMD)** for non-stationary trend extraction (denoising)  [cite: 16, 139-141] and an optimized multivariate **Hankel-DMD** model for highly accurate, short-term forecasting.

---

## ✨ Key Features

* [cite_start]**No Fixed Frequency Cutoffs:** Replaces traditional static filters (e.g., Butterworth or Low-Pass) with an adaptive, eigenvalue-driven separation scheme based on system physics  [cite: 58-59, 92].
* **State Space Augmentation:** Leverages Hankel embedding to calculate full system momentum and trajectory parameters from single-sensor profiles.
* **Recursive Multiscale Analysis:** Features custom multi-level windowing algorithms to capture transient behavior without spectral smearing.
* **Robust Multivariate Prediction:** Models cross-variable coupling (e.g., Surge-Sway-Roll coupling) to predict complex multi-axis motion states simultaneously.
* **Normalised Evaluation Metrics:** Avoids metric deflation errors via real-time window tracking for unbiased performance benchmarking.

---

## 📊 Experimental Results

Performance evaluation metrics averaged over an extensive 25-iteration Monte Carlo validation loop demonstrate exceptional denoising capabilities and forecasting consistency:

### 1. Denoising & Reconstruction Performance
The framework successfully purges stochastic noise while retaining underlying maneuver dynamics:
* **Roll Rate ($p$) Noise Reduction:** **99.33%** variance suppression.
* **Surge ($u$) Signal Reconstruction:** $R^2$ accuracy greater than **0.995**.

### 2. Multivariate Forecasting Metrics
Evaluating short-term predictive tracking over a specific time horizon:

| State Vector Component | RMSE | Normalized RMSE (NRMSE) | Relative $L_2$ Error | Prediction Horizon ($T_{ph}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Surge ($u$)** | 0.23736 | 0.20384 | **0.00730** | 6.56 Seconds |
| **Sway ($v$)** | 0.08272 | 0.28581 | 0.17904 | 2.52 Seconds |
| **Roll Rate ($p$)** | 0.00063 | 0.20079 | 0.33554 | **10.08 Seconds** |

---

## 📂 Repository Structure
├── final.m                 # Main execution script running Monte Carlo loops & analytics
├── run_mrdmd_recursive.m   # Engine implementing the tree level partitioning and mrDMD logic
├── get_level_data.m        # Helper function managing hierarchical matrix indexing
├── ship_data.csv           # Multi-axis ship simulation sensor logs (Raw Dataset)
└── README.md               # Documentation guide
