The project explores decentralized optimization algorithms with communication efficiency, particularly in the context of logistic regression and linear regression problems. It includes implementations of several gradient tracking based methods with various compression techniques.

## Project Structure

```
OCGT_BF/
├── nda/                      # Core algorithm and utilities package
│   ├── __init__.py
│   ├── log.py                # Logging utilities
│   ├── datasets/             # Dataset processing modules
│   ├── experiment_utils/     # Experiment helper utilities
│   ├── optimizers/           # Optimization algorithms
│   └── problems/             # Optimization problem definitions
├── experiments/              # Experiment scripts
│   └── convex/               # Convex optimization experiments
│       ├── log_cycle.py      # Cycle topology experiments
│       ├── log_er.py         # ER random graph experiments
│       ├── log_star.py       # Star topology experiments
│       ├── log_three.py      # Three topologies experiments
│       ├── data/             # Experiment data storage
│       └── figs/             # Experiment figure storage
├── requirements.txt          # Project dependencies
└── er_graph.eps              # Generated example graph
```

## Features

1. **Decentralized Optimization Algorithms**:
   - CDGT (Communication-Efficient Decentralized Gradient Tracking)
   - CDGT_bandit and variants (decay, topk, randk) **This refers to the algorithm OCGT_BF in our work.**
   - Q-SGT (Quantized Stochastic Gradient Tracking)
   - DGT (Decentralized Gradient Tracking)

2. **Optimization Problems**:
   - Logistic Regression
   - Linear Regression

3. **Dataset Handling**:
   - MNIST
   - Spam Email
   - Gisette
   - Generic LibSVM format data

4. **Experiment Tools**:
   - Run experiments and collect results (`run_exp`)
   - Result visualization (`plot_results`)
   - Performance evaluation (communication rounds, bits, accuracy)

5. **Network Topologies**:
   - Cycle network
   - ER random graph
   - Star network

## Installation

1. Create and activate a virtual environment (conda recommended):

```bash
conda create -n DGD python=3.8
conda activate DGD
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

### Logistic Regression Experiments

1. **Cycle Topology Experiment**:

```bash
python experiments/convex/log_cycle.py
```

2. **ER Random Graph Experiment**:

```bash
python experiments/convex/log_er.py
```

3. **Star Network Experiment**:

```bash
python experiments/convex/log_star.py
```

4. **Three-node Experiment**:

```bash
python experiments/convex/log_three.py
```

### Customizing Experiments

When running experiments, you can modify these parameters:

- `n_agent`: Number of agents (nodes)
- `m`: Number of samples per agent
- `dim`: Feature dimension
- `kappa`: Condition number
- `n_iters`: Number of iterations
- `dataset`: Dataset name (e.g., 'a5a', 'spam')
- `graph_type`: Network topology type ('cycle', 'er', 'star')
- `graph_params`: Network parameters (connection probability for ER graph)

## Results Analysis

Experiment results are automatically saved in:
- Data: data
- Figures: figs

The figures typically include:
1. Communication rounds vs. optimization error
2. Communication bits vs. optimization error
3. Accuracy vs. iterations

## Core Algorithms

The project implements several decentralized optimization algorithms, mainly variants of the CDGT series:

- `CDGT`: Basic decentralized gradient tracking algorithm
- `CDGT_bandit`: Bandit feedback variant
- `CDGT_bandit_decay`: Bandit variant with decay
- `CDGT_bandit_decay_topk`: Combined with Top-k sparsification
- `CDGT_bandit_decay_randk`: Combined with random-k sparsification
- `Q-SGT`: Quantized stochastic gradient tracking

These algorithms show different performance characteristics under various network topologies and problem settings.

## Datasets

The project supports various common datasets:

1. MNIST: Handwritten digit recognition
2. Spam Email: Email classification
3. Gisette: Feature selection challenge dataset
4. General LibSVM format datasets

Datasets are automatically downloaded and cached in the `~/data` directory.

## Logging

The project uses a Google-style logging system, which can be configured via the `nda.log` module:

```python
from nda import log
log.set_level(log.DEBUG)  # Or log.INFO, log.WARNING, etc.
```

## Notes

1. When running experiments for the first time, datasets may need to be downloaded, so ensure network connectivity.
2. Large-scale experiments can take significant time; consider using the `n_cpu_processes` parameter for multi-process acceleration.
3. Results are saved automatically, and more detailed data can be saved by setting `save=True`.
