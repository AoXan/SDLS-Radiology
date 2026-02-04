# SDLS-Radiology: Semantically Decoupled Latent Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Official Reference Implementation for "Suppressing Prior-Comparison Hallucinations in Radiology Report Generation via Semantically Decoupled Latent Steering" (IEEE TPAMI).**

> **Note**: This repository serves as an illustrative implementation to demonstrate the algorithmic logic and mathematical fidelity of the SDLS framework. It is structured to facilitate code audit and reproducibility of the paper's methodology.

## 📖 Abstract

Automated radiology report generation often suffers from **Prior-Comparison Hallucination**, where models generate historical findings unsupported by current imaging data. We propose **Semantically Decoupled Latent Steering (SDLS)**, a training-free inference-time intervention framework. By constructing an orthogonal steering vector derived from contrastive report pairs, SDLS precisely neutralizes historical bias without compromising clinical accuracy.

## 📂 System Architecture

The codebase is organized into four logical stages, strictly aligning with the mathematical definitions in Section III of the paper.

```text
SDLS-Radiology/
├── data/                       # Stage I: Contrastive Context Mining
│   ├── mine_contrastive_pairs.py  # Mining (r_hist, r_curr) pairs
│   ├── link_images.py             # Multimodal data linking
│   └── semantics.py               # Semantic decomposition (LLM-based)
├── core/                       # Stage II & III: The SDLS Algorithm
│   ├── extract_states.py          # MCV Extraction (Eq. 1)
│   ├── decomposition.py           # Orthogonal Decomposition (Eq. 5)
│   └── steering.py                # Norm-Preserving Steering (Eq. 6)
├── experiments/                # Inference Implementation
│   └── inference.py               # Online Phase (Algorithm 1)
├── metrics/                    # Evaluation Framework
│   ├── hsr.py                     # History Span Rate (Eq. 7)
│   ├── classifier.py              # Hallucination Probability (FilBERT)
│   ├── clinical.py                # RadGraph & Clinical Efficacy
│   └── decoupling.py              # Metric Decoupling Analysis
└── utils/                      # Backbone Utilities
    └── model_loader.py            # Support for BiomedGPT, IAMJB, LLaVA
```

## 🧠 Methodological Implementation

### Stage I: Data Construction (Section III-A)

The foundation of SDLS is the construction of a specific difference vector $\Delta v$.

* **`data/mine_contrastive_pairs.py`**: Implements the regex logic to parse "Impression" sections and identify pairs where historical references exist in the original report ($r_{orig}$) but are removed in the current report ($r_{curr}$).
* **`data/semantics.py`**: Uses LLMs to decompose reports into atomic findings, categorizing them into *Comparison*, *Stability*, or *Progression* (Table I) to construct **Specific-ICVs**.

### Stage II: Orthogonal Decomposition (Section III-B)

This stage computes the steering vector $v_{SDIV}$.

* **`core/extract_states.py`**: Implements **Equation 1**, extracting the Multi-layer Contextual Vector (MCV) $h$ from all decoder layers.
* **`core/decomposition.py`**: Implements **Equation 5**. It stacks the difference vectors ($\Delta v$) and applies **QR Decomposition**. The mean of the orthogonal basis $Q$ is computed to derive the **SDIV (Semantically Decoupled Intervention Vector)**.
* *Support for Baseline*: Also implements PCA for Global ICV (Eq. 3).

### Stage III: Latent Steering (Section III-C)

This stage injects the vector during inference.

* **`core/steering.py`**: Encapsulates the `SDLSEngine`. It implements **Equation 6 (Norm-Preserving Addition)**:

$$
h' = \|h\| \cdot \text{norm}(h/\|h\| + \lambda \cdot v_{SDIV})
$$

It supports the **SteerFair** strategy by hooking into the Attention Output submodule, as validated in Table V.

### Evaluation Metrics (Section III-D)

* **`metrics/hsr.py`**: Implements **History Span Rate (HSR)**, a token-level metric defined in Eq. 7 to quantify the density of hallucinated prior comparisons.
* **`metrics/decoupling.py`**: Implements the linear regression analysis (Table IV) to demonstrate that hallucination suppression is independent of general text similarity metrics like BERTScore.
