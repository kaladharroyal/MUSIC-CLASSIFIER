# Interpretable Music Auto-Tagging & Genre Classification via Augmented Perceptual Stacking

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+ (CUDA)](https://img.shields.io/badge/PyTorch-2.0+_(CUDA)-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost 2.0+](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![XAI - SHAP](https://img.shields.io/badge/XAI-SHAP-orange.svg)](https://shap.readthedocs.io/)
[![Evaluation - ROAR](https://img.shields.io/badge/Evaluation-ROAR%20Faithfulness-purple.svg)]()
[![Status](https://img.shields.io/badge/IEEE%20Access-Pre--Submission%20Ready-brightgreen.svg)](docs/manuscript_draft.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic research pipeline and reproducible benchmarking suite for **Interpretable Music Auto-Tagging** and **Multi-Dataset Genre Classification**. This framework bridges the classical trade-off between **black-box deep learning accuracy** and **human-understandable Explainable AI (XAI)** by combining deep spectrogram representations with acoustically grounded signal descriptors, mid-level perceptual features, and mathematical explanation faithfulness evaluation (**ROAR Deletion/Insertion curves**).

---

## 🌟 Key Highlights

- **Pareto-Optimal Stacking Architecture**: Combines deep spectrogram representations (2D CNN, CRNN with Attention) with 7 mid-level perceptual musical descriptors into an Augmented Stacking Meta-Learner, achieving **~86–89% test accuracy** while preserving full SHAP feature attribution transparency.
- **Deep Spectrogram Networks (PyTorch CUDA/CPU)**: 4-stage 2D CNN (~82–85%) and CRNN with Bidirectional LSTM + Softmax Self-Attention Pooling (~81–85%).
- **Mid-Level Perceptual Taxonomy**: Models 7 intuitive musical dimensions (*Melodiousness, Articulation, Rhythmic Stability, Rhythmic Complexity, Dissonance, Tonal Stability, Minorness*) alongside 23 signal descriptors.
- **Quantitative Explanation Faithfulness (ROAR)**: Implements genuine **Remove and Retrain (ROAR)** Deletion and Insertion retraining curves to mathematically verify SHAP explanation fidelity against post-hoc heuristics.
- **Authentic Multi-Dataset Validation**: Evaluated on real extracted audio features across **GTZAN** (10-genre multi-class), **MTG-Jamendo** (single-label genre subset, **0.719 ROC-AUC**), and **MagnaTagATune** (multi-label tagging, **0.706 macro ROC-AUC**).
- **Leakage-Free Validation & Dataset Integrity**: Enforces strict track-level segment-leakage-free train/val/test splits (70/15/15), addressing known GTZAN label noise and repetition issues (*Sturm 2012, 2014*).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Input ["🎵 Raw Audio & Feature Modalities"]
        A["Audio Track (30s Clip)"] --> B["Log Mel-Spectrogram (128 Bins)"]
        A --> C["23 Signal Descriptors (Essentia / Librosa)"]
        A --> D["7 Mid-Level Perceptual Descriptors"]
    end

    subgraph Deep_Models ["🧠 Deep Spectrogram Branch (PyTorch CUDA/CPU)"]
        B --> E["2D CNN (Conv + BatchNorm + Dropout)"]
        B --> F["CRNN (BiLSTM + Self-Attention Pooling)"]
    end

    subgraph Perceptual_Branch ["📊 Interpretable Perceptual Branch"]
        C --> G["XGBoost Perceptual Classifier"]
        D --> G
    end

    subgraph Stacking ["⚡ Augmented Perceptual Stacking Meta-Learner"]
        E -->|OOF Probability Vectors| H["Meta-Learner (Logistic Regression)"]
        F -->|OOF Probability Vectors| H
        G -->|OOF Probability Vectors| H
        D -->|Direct Mid-Level Augmentation| H
    end

    subgraph Outputs ["📈 Predictions & Explanation Faithfulness"]
        H --> I["Final Ensemble Prediction (~86-89%)"]
        G --> J["SHAP TreeExplainer Attribution"]
        J --> K["ROAR Faithfulness Engine (Deletion / Insertion Retraining)"]
    end
```

---

## 📊 Benchmark Performance Summary

### 1. GTZAN 10-Genre Benchmark (Strict Leakage-Free Track Split)

| Model Architecture | Feature Representation | Test Accuracy | Interpretability (XAI) |
| :--- | :--- | :---: | :--- |
| Majority Class Baseline | None | 10.00% | None |
| Logistic Regression | 57 Librosa Features | ~65–71% | Linear Weights |
| Random Forest Classifier | 57 Librosa Features | ~75–79% | Gini Importance |
| **Lyberatos et al. (IEEE Access 2025)** | 62 Perceptual / Harmonic | **79.00%** | SHAP / Feature Importance |
| Standalone Perceptual MLP | 64 Tabular (Acoustic + Mid-Level) | ~72–76% | Perceptual SHAP |
| Segment-Level XGBoost (Track Agg.) | 57 3-Sec Segment Features | ~78–83% | SHAP TreeExplainer |
| Standalone 2D CNN (PyTorch) | Log Mel-Spectrograms | ~82–85% | Black-Box Filters |
| Standalone CRNN + Self-Attention | Spectrogram Sequences | ~81–85% | Attention Weights |
| Base Stacking Ensemble | Deep OOF Probabilities | ~86–88% | Meta-Weights |
| **Augmented Perceptual Stacking Ensemble** | **Deep OOF + 7 Mid-Level Features** | **~86–89%** | **SHAP + ROAR Faithfulness** |

### 2. Multi-Dataset Cross-Validation (Authentic Extracted Features)

| Benchmark Dataset | Target Metadata | Evaluation Metric | Paper Baseline Reference | Authentic Extracted Result |
| :--- | :--- | :---: | :---: | :---: |
| **GTZAN** | 10 Genre Labels | Accuracy | 79.00% | **~86–89% (Stacking)** / **79.67% (XGB)** |
| **MTG-Jamendo** | Single-Label Genre Subset | ROC-AUC (OVR) | 0.729 | **0.719** (42.22% Multi-Class Acc) |
| **MagnaTagATune** | Multi-Label Audio Tags | Macro ROC-AUC | 0.840 | **0.706** (Classical: 0.921, Opera: 0.817) |

#### MagnaTagATune Tag-Wise Breakdown
- **Classical**: 0.921 ROC-AUC
- **Opera**: 0.817 ROC-AUC
- **Piano**: 0.782 ROC-AUC
- **Guitar**: 0.712 ROC-AUC
- **Drums**: 0.701 ROC-AUC

---

## 🔬 Quantitative Explanation Faithfulness (ROAR)

Most interpretability literature evaluates XAI methods through qualitative spot-checks or visual inspection. This framework implements the **Remove and Retrain (ROAR)** protocol (*Hooker et al., NeurIPS 2019*):

1. **Salience Ranking**: Compute mean absolute SHAP attributions $|\phi_i|$ across test predictions.
2. **Deletion Retraining Curve**: Incrementally drop top $k\%$ most salient features from both train and test sets and retrain a fresh classifier. A monotonic degradation confirms that explanations identify genuinely critical acoustic features.
3. **Insertion Retraining Curve**: Starting from an empty feature set, incrementally add top $k\%$ most salient features and retrain. Rapid performance recovery confirms predictive sufficiency.

![ROAR Faithfulness Curves](docs/figures/roar_faithfulness_curves.png)

```
ROAR Retraining Faithfulness Results on GTZAN:
- Modified   0% features -> Deletion Acc: 66.67% | Insertion Acc: 10.00%
- Modified  10% features -> Deletion Acc: 63.33% | Insertion Acc: 57.33%  (Rapid insertion jump confirms SHAP fidelity)
- Modified  50% features -> Deletion Acc: 58.00% | Insertion Acc: 67.33%
- Modified  80% features -> Deletion Acc: 35.33% | Insertion Acc: 67.33%
- Modified 100% features -> Deletion Acc: 10.00% | Insertion Acc: 66.67%
```

### Global & Dataset-Specific SHAP Attributions

![MTG-Jamendo SHAP Explanations](docs/figures/jamendo_real_shap.png)

---

## 📁 Repository Structure

```text
music classification/
├── docs/                                  # Academic documentation, reference papers & figures
│   ├── figures/                           # Publication-grade evaluation figures
│   │   ├── roar_faithfulness_curves.png   # ROAR Deletion vs Insertion curves
│   │   ├── jamendo_real_shap.png          # MTG-Jamendo authentic SHAP importance summary
│   │   ├── mtat_classical_shap.png        # MagnaTagATune Classical tag SHAP summary
│   │   └── mtat_guitar_shap.png           # MagnaTagATune Guitar tag SHAP summary
│   ├── manuscript_draft.md                # Full IEEE Access pre-submission manuscript draft
│   └── reference_paper.pdf                # Reference paper (Lyberatos et al., IEEE Access 2025)
│
├── src/                                   # Standalone Python research & verification package
│   ├── __init__.py                        # Package init
│   ├── essentia_features.py               # 23 signal processing descriptors (Essentia/Librosa)
│   ├── midlevel_vggish.py                 # 7-dim perceptual feature CNN (MidLevelVGGish PyTorch)
│   ├── faithfulness_eval.py               # Genuine ROAR quantitative XAI faithfulness engine
│   ├── extract_all_real_features.py       # Batch feature extractor for MTG-Jamendo & MTAT
│   ├── run_roar_evaluation.py             # Standalone runner for ROAR faithfulness verification
│   ├── run_real_feature_evaluation.py     # Standalone runner for multi-dataset benchmarks & SHAP
│   └── append_to_notebook.py              # Notebook synchronization utility
│
├── Data/                                  # Structured data directory (gitignored binaries)
│   ├── audio/                             # Raw audio tracks (GTZAN genres_original)
│   ├── csv/                               # Tabular CSVs (features_30_sec.csv, features_3_sec.csv)
│   ├── metadata/                          # Train/Val/Test split indices & cache
│   ├── models/                            # PyTorch (.pth) checkpoints
│   ├── processed/                         # Extracted multi-dataset feature CSVs
│   └── research_external/                 # MTG-Jamendo & MagnaTagATune annotations
│
├── music_classification_fixed.ipynb       # Master reproducible Jupyter Notebook (Parts 0–10)
├── requirements.txt                       # Consolidated Python dependencies
├── PROJECT_PROGRESS_AND_MILESTONES.md     # Detailed milestone tracking & technical audit log
└── README.md                              # Main documentation & execution guide
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.10+** installed. For GPU acceleration, ensure NVIDIA CUDA drivers are installed.

```bash
# Clone the repository
git clone https://github.com/kaladharroyal/music-classification.git
cd "music classification"

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Master Jupyter Notebook

Open `music_classification_fixed.ipynb` in VS Code, JupyterLab, or Cursor:

```bash
jupyter notebook music_classification_fixed.ipynb
```

The notebook contains 11 structured sections (82 cells):
- **Part 0**: Global Setup & PyTorch CUDA Initialization
- **Part 1**: Dataset Ingestion, Leakage-Free Splitting & Literature Caveats (*Sturm 2012, 2014*)
- **Part 2**: Exploratory Data Analysis & Feature Engineering
- **Part 3**: Classical Baselines (Logistic Regression, Random Forest)
- **Part 4**: Deep Spectrogram Models (2D CNN, CRNN with Self-Attention, Stacking Meta-Learners)
- **Part 5**: Explainable AI (SHAP TreeExplainer) & Perceptual Feature Taxonomy
- **Part 6**: Quantitative Explanation Faithfulness Engine (**ROAR Deletion & Insertion Retraining**)
- **Part 7**: Multi-Dataset Evaluation (**MTG-Jamendo** & **MagnaTagATune**)
- **Part 8**: Per-Genre Natural Language Explanation Synthesis
- **Part 9**: Cross-Dataset Master Benchmark Summary
- **Part 10**: Standalone Modules & Pre-Submission IEEE Access Draft

### 3. Run Standalone CLI Research Modules

Each research module in `src/` can be executed independently from the command line:

```bash
# 1. Compute true ROAR Quantitative Explanation Faithfulness Curves on GTZAN
python src/run_roar_evaluation.py

# 2. Extract 30-dimensional real features across MTG-Jamendo and MagnaTagATune
python src/extract_all_real_features.py

# 3. Evaluate Multi-Dataset XGBoost benchmarks and output authentic ROC-AUC & SHAP plots
python src/run_real_feature_evaluation.py
```

---

## 📖 Citation & References

```bibtex
@article{lyberatos2025challenges,
  title={Challenges and Perspectives in Interpretable Music Auto-Tagging Using Perceptual Features},
  author={Lyberatos, Vassilis and others},
  journal={IEEE Access},
  volume={13},
  year={2025},
  publisher={IEEE}
}

@article{sturm2014state,
  title={A Survey of Evaluation in Music Genre Recognition},
  author={Sturm, Bob L.},
  journal={IEEE Transactions on Multimedia},
  volume={16},
  number={1},
  pages={2--18},
  year={2014}
}

@inproceedings{hooker2019benchmark,
  title={A Benchmark for Interpretability Methods},
  author={Hooker, Sara and Erhan, Dumitru and Kindermans, Pieter-Jan and Been, Kim},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={32},
  year={2019}
}

@inproceedings{lundberg2017unified,
  title={A Unified Approach to Interpreting Model Predictions},
  author={Lundberg, Scott M. and Lee, Su-In},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
