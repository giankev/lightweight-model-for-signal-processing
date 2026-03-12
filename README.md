# Lightweight Physics-Informed Hybrid Receivers for OFDM and Single-Carrier Communications

This repository contains the code developed for the study of **lightweight physics-informed neural receivers** for digital communications, with a focus on **16-QAM**, **5G LDPC coding**, and **receiver-side robustness under synchronization impairments**.

The project investigates how **hybrid neural architectures** can improve receiver performance in the presence of impairments such as:
- residual phase offset
- carrier frequency offset (CFO)
- additive noise
- reduced pilot availability in OFDM settings

The main idea is to combine:
- **model-based signal processing**, where the physical structure is well known
- **lightweight neural modules**, where classical estimation becomes brittle or suboptimal

In particular, the repository includes experiments on:
- **single-carrier transmission**
- **OFDM transmission**
- **standard pilot regimes**
- **pilot-starved OFDM regimes**
- **hybrid linear-attention receivers**
- **CNN-based baselines**
- **ablation studies and oracle variants**

The implementation follows the physics-informed hybrid philosophy described in the project paper, where neural components estimate or support impairment-aware processing while preserving compatibility with conventional decoding pipelines such as **5G LDPC** :contentReference[oaicite:1]{index=1}

---

## Repository structure

- `sionna_dg.py`
  - Core simulation module used across the project
  - Contains:
    - communication system configuration
    - dataset generation
    - OFDM and single-carrier simulation
    - impairment modeling
    - classical baseline receiver
    - utility functions for modulation, coding, demapping, and evaluation

- `environment.txt`
  - Environment notes and setup information

- `requirements.txt`
  - Python dependencies required to run the notebooks

- `plots/`
  - Contains scripts and utilities used to generate the figures and plots included in the paper

---

## Notebooks

- `Data_Generation.ipynb`
  - Generates the datasets used for training and evaluation
  - Builds synthetic communication data with Sionna-based simulations
  - Supports both standard and starved OFDM settings

- `dataset_creation.ipynb`
  - Utility notebook for exporting and organizing the datasets used in the experiments

- `Dataset_Examples.ipynb`
  - Visualizes example samples from the generated datasets
  - Includes received constellations, I/Q trajectories, and signal representations useful for analysis and for the paper figures

---

### Single-carrier experiments

- `Hybrid_NN_Single_Carrier.ipynb`
  - Implements the hybrid neural receiver for the **single-carrier** scenario
  - Studies coded 16-QAM transmission under phase and CFO impairments
  - Compares neural processing against the classical baseline

---

### OFDM standard regime

- `Hybrid_LA_NN_OFDM.ipynb`
  - Main hybrid OFDM receiver based on **Linear Attention**
  - Combines impairment-aware estimation with lightweight neural soft-output processing
  - This is one of the main notebooks of the repository

- `Hybrid_CNN_OFDM.ipynb`
  - Hybrid OFDM neural receiver using a CNN-based architecture
  - Used as a comparison against the linear-attention variant

- `CNN_OFDM.ipynb`
  - Pure CNN-based OFDM receiver baseline
  - Useful to compare fully learned convolutional processing against hybrid designs

- `Oracle_LA_NN_OFDM.ipynb`
  - Oracle version of the linear-attention hybrid model
  - Uses ideal or ground-truth impairment information to isolate the performance gap due to estimation

- `Classic_Param_LA_NN_OFDM.ipynb`
  - Hybrid OFDM pipeline where classical parameter estimation is combined with the linear-attention neural receiver
  - Useful to separate the contribution of neural soft processing from the contribution of impairment estimation

---

### OFDM starved-pilot regime

- `Hybrid_LA_NN_OFDM_Starved.ipynb`
  - Main hybrid linear-attention receiver for the **pilot-starved OFDM** regime
  - Focuses on reduced pilot density and the resulting difficulty of reliable synchronization and decoding

- `Hybrid_SE_NN_OFDM_Starved.ipynb`
  - Starved-regime hybrid receiver variant with **Squeeze-and-Excitation (SE)** style feature recalibration
  - Used for architecture comparison

- `Hybrid_GLU_NN_OFDM_Starved.ipynb`
  - Starved-regime hybrid receiver variant with **Gated Linear Units (GLU)**
  - Used for comparison with the linear-attention design

- `Hybrid_LA_NN_Training_Ablation_Studies_Starved.ipynb`
  - Ablation study notebook for the proposed starved-regime hybrid model
  - Investigates the contribution of training choices and architectural components

- `DeepRx_OFDM_Starved.ipynb`
  - DeepRx-style baseline adapted to the pilot-starved OFDM setting
  - Used as a neural comparison model

- `DeepWaveform_OFDM_Starved.ipynb`
  - DeepWaveform-style baseline for the pilot-starved setting
  - Used for benchmarking against other neural architectures

- `VanillaMHSA_OFDM_Starved.ipynb`
  - Baseline OFDM receiver based on standard Multi-Head Self-Attention
  - Used to compare quadratic-attention models against the proposed lightweight approach

- `DAT_OFDM_Starved.ipynb`
  - Dual-attention-transformer-inspired baseline for the starved OFDM setting
  - Used as an additional attention-based comparison architecture

---

## Study focus

- Lightweight and deployable neural receivers
- Physics-informed hybrid processing
- Soft-output receiver design compatible with LDPC decoding
- Standard and pilot-starved OFDM scenarios
- Comparison between:
  - classical baselines
  - hybrid neural receivers
  - CNN baselines
  - attention-based baselines
- Evaluation in terms of:
  - BER
  - BLER
  - robustness to impairments
  - model complexity
  - computational efficiency

---

## Notes

- The repository is organized around **reproducible link-level simulations**
- The datasets are generated synthetically through the shared simulation backend
- The notebooks are intentionally separated by scenario and architecture in order to make:
  - comparisons clearer
  - ablations easier to reproduce
  - figures and tables for the paper easier to regenerate

---

## Reference

This repository accompanies the project study on lightweight physics-informed hybrid receivers for OFDM systems, including standard and pilot-starved regimes, as described in the associated paper draft :contentReference[oaicite:2]{index=2}
