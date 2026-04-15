# ECGTrace

**Mechanistic interpretability of ECG-language models for cardiac diagnosis.**

ECGTrace fine-tunes a multimodal transformer on paired ECG waveforms and clinical text to generate natural-language rhythm interpretations, then audits *which* morphological features - QRS width, ST elevation, RR variability, T-wave shape - causally drive each diagnostic conclusion.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-red.svg)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow.svg)]()

---

## Overview

Most ECG AI systems are black boxes: they output a label without explaining *why*. ECGTrace changes that by combining two components:

1. **ECG-to-text generation**: a PatchTST waveform encoder jointly trained with a BioGPT decoder to generate structured cardiac reports from 12-lead ECG signals.
2. **Mechanistic audit**: causal mediation analysis and cross-attention probing that identify which time-series features in the ECG causally mediate specific text outputs (e.g., "atrial fibrillation", "ST elevation", "left ventricular hypertrophy").

The result is an interpretable cardiac language model that generates a report *and* a saliency overlay showing exactly which waveform segments drove each diagnosis.

---

## Architecture

```
PTB-XL / MIMIC-IV-ECG
        |
  Preprocessing
  (bandpass, segment, normalize)
        |
   +---------+       +------------------+
   | PatchTST |       | ClinicalBERT      |
   | Encoder  |       | Text Encoder      |
   +---------+       +------------------+
        |                    |
        +--------------------+
               |
        Joint Training
     (contrastive + generative)
               |
        BioGPT Decoder
               |
       Generated Report
               |
    +----------+----------+
    |                     |
Causal Mediation    Cross-Attention
Analysis            Probing
    |                     |
    +----------+----------+
               |
     Attribution Heatmap
     (waveform saliency)
               |
    Streamlit Dashboard
```

---

## Datasets

| Dataset | Size | Access | Use |
|---|---|---|---|
| PTB-XL | 21,837 ECGs, 12-lead | PhysioNet (free) | Pre-training, fine-tuning |
| MIMIC-IV-ECG | ~800K ECGs + notes | PhysioNet (credentialed) | Paired text supervision |
| ECG-QA | QA pairs over PTB-XL | GitHub (free) | Optional QA extension |

> PhysioNet credentialing takes 3-7 days. Start it at https://physionet.org/register/

---

## Team Roles TBD

| Area | Component |
|---|---|
| **Person 1** | Mechanistic interpretability: causal mediation analysis, cross-attention probing, attribution heatmaps |
| **Person 2** | ECG data pipeline + PatchTST encoder + SimCLR pre-training |
| **Person 3** | BioGPT decoder, evaluation suite, Streamlit dashboard, HuggingFace deployment |

---

## Quickstart

```bash
git clone https://github.com/yourusername/ECGTrace.git
cd ECGTrace
pip install -e ".[dev]"

# Download PTB-XL (no credentialing needed)
python scripts/download_ptbxl.py

# Preprocess
python scripts/preprocess.py --dataset ptbxl --output data/processed/ptbxl

# Train ECG encoder (SimCLR pre-training)
python scripts/train_encoder.py --config configs/encoder_pretrain.yaml

# Fine-tune with language decoder
python scripts/train_joint.py --config configs/joint_finetune.yaml

# Run interpretability analysis
python scripts/run_interpretability.py --checkpoint checkpoints/best.pt

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Results

| Metric | Score |
|---|---|
| BLEU-4 | TBD |
| BERTScore F1 | TBD |
| Clinical accuracy (5-class) | TBD |
| Causal mediation R^2 | TBD |

---

## Project Structure

```
ECGTrace/
├── configs/                  # YAML experiment configs
├── data/
│   ├── raw/                  # Downloaded datasets (gitignored)
│   ├── processed/            # Preprocessed tensors
│   └── splits/               # Train/val/test split indices
├── src/
│   ├── data/                 # Dataset classes and loaders
│   ├── models/               # Encoder, decoder, joint model
│   ├── training/             # Training loops and losses
│   ├── interpretability/     # Causal mediation, attention probing
│   ├── evaluation/           # BLEU, BERTScore, clinical rubric
│   └── dashboard/            # Streamlit app
├── notebooks/                # Exploratory analysis
├── scripts/                  # CLI entry points
├── tests/                    # Unit tests
└── docs/                     # Extended documentation
```

---

## Citation

```bibtex
@misc{ecgtrace2026,
  title={ECGTrace: Mechanistic Interpretability of ECG-Language Models for Cardiac Diagnosis},
  author={Adam Hamadene, Mahdi Najjar, Jacob Ortiz},
  year={2026},
  url={https://github.com/yourusername/ECGTrace}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
