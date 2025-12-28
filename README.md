# CHRONOS: Certified Temporal Indexing for Streaming Graph Fraud Detection

[![VLDB Journal](https://img.shields.io/badge/VLDB%20Journal-2025-blue.svg)](https://www.springer.com/journal/778)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **CHRONOS** (**C**ertified **H**igh-performance **R**eal-time **O**perations for **N**etwork-aware **O**nline **S**treaming detection), a production-ready system for real-time cryptocurrency fraud detection.

---

## Overview

CHRONOS is an integrated system combining certified O(log n) temporal indexing, topology-aware incremental learning, and adversarial-robust detection for cryptocurrency fraud detection on temporal multigraphs.

### Key Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| F1 Score | **0.9154** | ≥0.90 | ✓ |
| AUC-ROC | **0.9672** | — | — |
| Throughput | **9,847 TPS** | ≥5,000 TPS | ✓ |
| P99 Latency | **18.7 ms** | ≤50 ms | ✓ |
| Recovery Time | **40.9 s** | ≤60 s | ✓ |

### Contributions

1. **Certified Temporal Index (C1):** B⁺-tree index achieving O(log n) insertion and O(log n + k) range query complexity with empirical validation (R² > 0.98)

2. **Topology-Aware Incremental Learning (C2):** TAIL protocol achieving 9.9× speedup while updating only 4.04% of parameters

3. **Adversarial-Robust Architecture (C3):** Certified attention with spectral normalization maintaining <5% accuracy degradation under ε=0.1 perturbations

4. **Distributed Fault-Tolerant Training (C4):** 87.3% scaling efficiency across 8 GPUs with 100% recovery success

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CHRONOS System                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Certified      │  │  TAIL           │  │  Neural Architecture    │  │
│  │  Temporal Index │  │  Incremental    │  │  • Certified Attention  │  │
│  │  • B⁺-tree      │  │  Learning       │  │  • MGD Layers           │  │
│  │  • O(log n)     │  │  • 9.9× speedup │  │  • Cross-chain Corr.    │  │
│  │  • R² > 0.98    │  │  • 4.04% params │  │  • Adversarial Training │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│           │                    │                        │               │
│           └────────────────────┼────────────────────────┘               │
│                                │                                        │
│  ┌─────────────────────────────┴─────────────────────────────────────┐  │
│  │                    Distributed Training Layer                     │  │
│  │    • Locality-aware partitioning (8.3% cross-partition edges)     │  │
│  │    • Gradient compression (87.3% volume reduction)                │  │
│  │    • Two-phase commit checkpointing (40.9s recovery)              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### Hardware
- **GPU:** NVIDIA RTX 3090 (24 GB) × 4 or equivalent
- **CPU:** Intel Xeon 4314 (32 threads) or equivalent
- **RAM:** 384 GB (minimum 64 GB for single-GPU mode)
- **Storage:** 500 GB NVMe SSD

### Software
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+
- Additional dependencies in `requirements.txt`

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/[anonymized]/chronos.git
cd chronos
```

### 2. Create Environment

```bash
conda create -n chronos python=3.9
conda activate chronos
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Graph learning
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Additional requirements
pip install -r requirements.txt
```

### 4. Download Datasets

```bash
# Ethereum datasets (Etherscan)
python scripts/download_data.py --dataset ethereum-s
python scripts/download_data.py --dataset ethereum-p

# Bitcoin datasets (WalletExplorer)
python scripts/download_data.py --dataset bitcoin-m
python scripts/download_data.py --dataset bitcoin-l
```

---

## Quick Start

### Single Evaluation Run

```bash
python experiment_runner.py \
    --config config.yaml \
    --dataset ethereum-s \
    --mode evaluate
```

### Training from Scratch

```bash
python experiment_runner.py \
    --config config.yaml \
    --dataset ethereum-s \
    --mode train \
    --epochs 50
```

### Distributed Training (4 GPUs)

```bash
torchrun --nproc_per_node=4 experiment_runner.py \
    --config config.yaml \
    --dataset ethereum-s \
    --mode train \
    --distributed
```

---

## Configuration

All hyperparameters are specified in `config.yaml`:

```yaml
# Index Configuration
index:
  fanout: 64                    # B⁺-tree fanout (cache-aligned)
  L_min: 8                      # Minimum sequence length
  L_max: 512                    # Maximum sequence length
  theta_low: 0.3                # Reallocation threshold (low)
  theta_high: 0.8               # Reallocation threshold (high)
  alpha: 0.2                    # Geometric growth factor

# TAIL Configuration
tail:
  depth: 3                      # BFS traversal depth
  degree_threshold: 1000        # High-degree node threshold
  rho_attention: 0.1            # Attention update threshold
  rho_encoder: 0.3              # Encoder update threshold

# Neural Architecture
model:
  hidden_dim: 128               # Hidden dimension
  num_layers: 2                 # MGD layers
  num_heads: 4                  # Attention heads
  dropout: 0.1                  # Dropout rate
  temporal_decay: 0.1           # λ for temporal weighting

# Adversarial Training
adversarial:
  epsilon: 0.1                  # Perturbation budget
  alpha: 0.01                   # PGD step size
  iterations: 10                # PGD iterations
  lambda_adv: 0.5               # Adversarial loss weight

# Distributed Training
distributed:
  num_gpus: 4                   # Number of GPUs
  gradient_compression: 0.1     # Top-k sparsification ratio
  staleness_bound: 3            # Bounded staleness τ
  checkpoint_interval: 500      # Iterations between checkpoints

# Training
training:
  batch_size: 256               # Batch size per GPU
  learning_rate: 0.001          # Initial learning rate
  weight_decay: 0.0001          # L2 regularization
  epochs: 50                    # Training epochs
  early_stopping: 10            # Patience for early stopping
```

---

## Reproducing Experiments

### Full Reproduction (~83 hours)

```bash
# Run complete evaluation suite
python experiment_runner.py \
    --config config.yaml \
    --mode full_evaluation \
    --output results/
```

### Individual Experiments

```bash
# Table 1: Detection Performance
python evaluation_suite.py --experiment detection_performance

# Table 2: Complexity Validation
python evaluation_suite.py --experiment complexity_validation

# Table 3: Baseline Comparison
python evaluation_suite.py --experiment baseline_comparison

# Table 4: Ablation Study
python evaluation_suite.py --experiment ablation

# Table 5: TAIL Performance
python evaluation_suite.py --experiment tail_performance

# Table 6: Adversarial Robustness
python evaluation_suite.py --experiment adversarial_robustness

# Table 7: Cross-chain Correlation
python evaluation_suite.py --experiment cross_chain

# Table 8: Scaling Efficiency
python evaluation_suite.py --experiment scaling

# Table 9: Fault Tolerance
python evaluation_suite.py --experiment fault_tolerance
```

### Theorem Validation

```bash
# Theorem 1: Insertion Complexity O(log n)
python evaluation_suite.py --experiment theorem1_validation

# Theorem 2: Range Query Complexity O(log n + k)
python evaluation_suite.py --experiment theorem2_validation

# Theorem 3: TAIL Complexity
python evaluation_suite.py --experiment theorem3_validation
```

---

## Project Structure

```
chronos/
├── README.md                   # This file
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
│
├── core_system.py              # Certified temporal index implementation
│   ├── BPlusTree               # B⁺-tree with O(log n) operations
│   ├── TemporalIndex           # Three-layer hierarchical index
│   ├── AdaptiveSequence        # Dynamic sequence management
│   └── ComplexityTracker       # Empirical validation metrics
│
├── incremental_engine.py       # TAIL incremental learning
│   ├── AffectedSubgraph        # BFS-based identification
│   ├── SelectiveUpdate         # Layer-wise parameter selection
│   ├── CurriculumTrainer       # Temporal-weighted training
│   └── TAILProtocol            # Unified incremental pipeline
│
├── neural_architecture.py      # Adversarial-robust architecture
│   ├── CertifiedAttention      # Spectrally-normalized attention
│   ├── MGDLayer                # Multi-graph discrepancy layer
│   ├── CrossChainCorrelation   # Multi-blockchain fusion
│   ├── AdversarialTrainer      # PGD adversarial training
│   └── CHRONOSModel            # Complete detection model
│
├── evaluation_suite.py         # Comprehensive evaluation
│   ├── DetectionMetrics        # F1, AUC-ROC, MCC computation
│   ├── SystemMetrics           # Throughput, latency measurement
│   ├── StatisticalTests        # Paired t-tests, Bonferroni correction
│   └── TheoremValidation       # Complexity bound verification
│
├── experiment_runner.py        # Experiment orchestration
│   ├── DataLoader              # Dataset preprocessing
│   ├── DistributedTrainer      # Multi-GPU training
│   ├── FaultTolerance          # Checkpointing and recovery
│   └── ResultsExporter         # LaTeX table generation
│
├── Figures/                    # Generated figures
│   ├── fig_architecture.pdf
│   ├── fig_tail.pdf
│   ├── fig_neural_architecture.pdf
│   ├── fig_distributed.pdf
│   └── fig_experiments.pdf
│
└── data/                       # Datasets (download separately)
    ├── ethereum-s/
    ├── ethereum-p/
    ├── bitcoin-m/
    └── bitcoin-l/
```

---

## API Reference

### Core System

```python
from core_system import TemporalIndex, BPlusTree

# Initialize index
index = TemporalIndex(
    fanout=64,
    L_min=8,
    L_max=512,
    theta_low=0.3,
    theta_high=0.8
)

# Insert transaction
metrics = index.insert(
    account_id="0x1234...",
    transaction=tx,
    direction="in"
)
# Returns: InsertionMetrics(comparisons=17, time_us=2.84)

# Range query
results, metrics = index.range_query(
    account_id="0x1234...",
    t_start=1704067200,
    t_end=1704153600,
    direction="out"
)
# Returns: List[Transaction], QueryMetrics(comparisons=23, results=42)
```

### Incremental Learning

```python
from incremental_engine import TAILProtocol

# Initialize TAIL
tail = TAILProtocol(
    model=model,
    depth=3,
    degree_threshold=1000
)

# Incremental update
metrics = tail.update(
    new_transactions=batch,
    graph=G
)
# Returns: TAILMetrics(speedup=9.92, params_updated=0.0404)
```

### Neural Architecture

```python
from neural_architecture import CHRONOSModel

# Initialize model
model = CHRONOSModel(
    input_dim=16,
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    num_chains=2,
    adversarial_training=True
)

# Forward pass
logits = model(node_features, edge_index, timestamps)

# Adversarial training step
loss = model.adversarial_step(
    batch,
    epsilon=0.1,
    alpha=0.01,
    iterations=10
)
```

---

## Baselines

We compare against seven systems:

| System | Venue | Implementation |
|--------|-------|----------------|
| D3-GNN | PVLDB 2024 | [Official](https://github.com/d3gnn/d3gnn) |
| RapidStore | PVLDB 2025 | [Official](https://github.com/rapidstore/rapidstore) |
| PlatoD2GL | ICDE 2024 | [Official](https://github.com/platod2gl/platod2gl) |
| PipeTGL | PVLDB 2025 | [Official](https://github.com/pipetgl/pipetgl) |
| JODIE | KDD 2019 | [Official](https://github.com/srijankr/jodie) |
| Flink+GNN | — | Custom implementation |
| Neo4j+GNN | — | Custom implementation |



---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (No. U22B2029)
- Open Research Fund of The State Key Laboratory of Blockchain and Data Security, Zhejiang University
- 2025 Science and Technology Projects of the National Meteorological Information Center, China Meteorological Administration (MNICJBGS202516)

---



<p align="center">
  <i>CHRONOS: Enabling certified, real-time fraud detection on blockchain networks.</i>
</p>
