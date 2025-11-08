# Constrained Bayesian Optimization (CBO)

Source code for the paper: [Optimistic Constrained Bayesian Optimization](https://openreview.net/pdf?id=D4NJFfrqoq)

Implementation of multiple acquisition functions for Bayesian Optimization with unknown constraints.

## Overview

This project implements and compares several constrained Bayesian optimization algorithms:

- **UCBC** (Upper Confidence Bound Constrained) - S⁻ set-based approach with adaptive/coupled query modes
- **ObsoleteAwareUCB** - Direct violation-based with obsolete constraint detection (recommended)
  - Paper: [Optimistic Constrained Bayesian Optimization](https://openreview.net/pdf?id=D4NJFfrqoq)
- **EIC** (Expected Improvement with Constraints) - Classic constrained BO
- **CMES-IBO** (Max-value Entropy Search) - Information-theoretic approach

## Installation

```bash
# Create virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Requirements
- Python 3.7+
- PyTorch >= 1.9.0
- GPyTorch >= 1.6.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

## Quick Start

```bash
# Run single experiment with ObsoleteAwareUCB (recommended)
python run-multi-constraint-bo.py \
  --num-rand 10 \
  --acqfunc obsolete_aware_ucb_mixed \
  experiment_config/branin_branin.json

# Run batch experiments
sh script_branin_branin.sh
sh script_gas_transmission.sh
sh script_welded_beam_design.sh
```

## Acquisition Functions

| Name | Description | Queries per Iteration | Notes |
|------|-------------|----------------------|-------|
| `ucbc_decoupled` | UCBC with adaptive queries | 1 (target OR constraint) | S⁻ set-based |
| `ucbc_coupled` | UCBC querying all together | n+1 (all) | Conservative |
| `obsolete_aware_ucb` | Direct violation-based | 1 (target OR constraint) | Simplified |
| `obsolete_aware_ucb_mixed` | With obsolete detection | 1+ | ⭐ Recommended |
| `eic` | Expected Improvement | 2 (target AND constraint) | Classic |
| `cmes_ibo` | Entropy-based | 2 (target AND constraint) | Info-theoretic |

### Algorithm Comparison

**UCBC vs ObsoleteAwareUCB:**
- **UCBC**: Uses S⁻ set (points where both lower and upper bounds satisfy constraints) to decide queries
- **ObsoleteAwareUCB**: Directly compares violation magnitude with target uncertainty (simpler, more robust)
- **Mixed mode**: Detects under-explored constraints (std > 2.0 × target_std) and queries them to prevent GP errors

## Experiment Configuration

Experiments are defined in JSON files under `experiment_config/`. Example:

```json
{
  "target": {
    "module_name": "branin",
    "identifier": "original_2_1",
    "n_init_observations": 3,
    "noise_std": 0.01,
    "ard": true,
    "is_hyperparameter_trainable": true,
    "gp_prior": {
      "lengthscale": {"gamma": {"concentration": 0.25, "rate": 0.5}},
      "outputscale": {"gamma": {"concentration": 2.0, "rate": 0.15}},
      "noise_std": {"gaussian": {"loc": 0.0, "scale": 0.1}}
    }
  },
  "constraints": [
    {
      "module_name": "branin",
      "share_observation_with_constraint": -1,
      "identifier": "original_2_1",
      "inequality": "greater_than_equal_to",
      "threshold": 0.6,
      "fix_mean_at": 0.6,
      "n_init_observations": 3,
      "noise_std": 0.01,
      "ard": true,
      "is_hyperparameter_trainable": true
    }
  ],
  "xsize": 100,
  "zsize": 100,
  "n_bo_iterations": 101,
  "beta": 40.0,
  "n_training_iter_gp_hyper": 30,
  "update_gp_hyper_every_iter": 1
}
```

### Key Configuration Parameters

**Target/Constraint Settings:**
- `module_name`: Function to optimize (e.g., `branin`, `welded_beam_design_2_3_5_obj`)
- `identifier`: Task variant (e.g., `original_2_1`)
- `n_init_observations`: Number of random initial observations
- `noise_std`: Observation noise standard deviation
- `ard`: Use Automatic Relevance Determination (different lengthscales per dimension)
- `is_hyperparameter_trainable`: Whether to optimize GP hyperparameters

**Constraint-Specific:**
- `share_observation_with_constraint`: Share observations with another constraint (-1 = no sharing)
- `inequality`: `"greater_than_equal_to"` or `"less_than_equal_to"`
- `threshold`: Constraint threshold value
- `fix_mean_at`: Fix GP mean at threshold to prevent empty feasible region estimation

**Optimization Settings:**
- `xsize`, `zsize`: Discrete domain grid size (total points = xsize × zsize)
- `n_bo_iterations`: Number of BO iterations
- `beta`: Exploration-exploitation trade-off (higher = more exploration)
- `n_training_iter_gp_hyper`: GP hyperparameter optimization iterations
- `update_gp_hyper_every_iter`: Update GP hyperparameters every N iterations

## Benchmark Problems

### Synthetic Functions (2D)
- **Branin**: Classic test function
- **Goldstein-Price**: Multi-modal function
- **Six-Hump Camel**: Multiple local optima
- **Hartmann 3D/6D**: Higher-dimensional test functions

### Real-World Engineering Problems
- **Welded Beam Design** (4D, 5 constraints): Structural optimization
- **Gas Transmission** (5D, 1 constraint): Pipeline design
- **CIFAR-10 CNN** (10 constraints): Neural network hyperparameter tuning

## Project Structure

```
cbo/
├── run-multi-constraint-bo.py      # Main runner for BO algorithms
├── multi_constraint_bo.py          # Core BO framework
├── ucbc.py                         # UCBC implementation (S⁻ set-based)
├── obsolete_aware_ucb.py           # ObsoleteAwareUCB (violation-based)
├── eic.py                          # Expected Improvement with Constraints
├── cmes_ibo.py                     # Max-value Entropy Search
├── gp.py                           # Gaussian Process wrapper (GPyTorch)
├── utils.py                        # Utility functions
├── function/                       # Benchmark function implementations
│   ├── branin.py
│   ├── welded_beam_design_2_3_5_*.py
│   ├── gas_transmission_2_3_15_*.py
│   └── ...
├── experiment_config/              # JSON experiment configurations
│   ├── branin_branin.json
│   ├── welded_beam_design_2_3_5.json
│   └── ...
├── script_*.sh                     # Batch execution scripts
├── out/                            # Results directory
└── log/                            # Execution logs
```

## Output Files

Results are saved in `out/` directory:

- `{experiment}_{acqfunc}_regrets.pkl`:
  - `instantaneous`: Regret at each iteration
  - `cumulative`: Cumulative regret over time

- `{experiment}_{acqfunc}_bo_info.pkl`:
  - `query_types`: History of what was queried (target/constraint)
  - `target_xz`, `target_y`: Target function observations
  - `constraint_xz_dict`, `constraint_y_dict`: Constraint observations
  - `cidx_to_moidx`: Constraint index to model index mapping

## Usage Examples

### Run Single Algorithm
```bash
python run-multi-constraint-bo.py \
  --num-rand 5 \
  --acqfunc obsolete_aware_ucb_mixed \
  experiment_config/branin_branin.json
```

### Compare Multiple Algorithms
Edit shell scripts to uncomment desired algorithms:
```bash
# In script_branin_branin.sh, uncomment:
python run-multi-constraint-bo.py ... --acqfunc obsolete_aware_ucb_mixed ...
python run-multi-constraint-bo.py ... --acqfunc eic ...
python run-multi-constraint-bo.py ... --acqfunc cmes_ibo ...
```

### Monitor Progress
```bash
# Watch log file
tail -f log/log_obsolete_aware_ucb_mixed_branin_branin.txt

# Check results
python -c "import pickle; print(pickle.load(open('out/branin_branin_obsolete_aware_ucb_mixed_regrets.pkl', 'rb')))"
```

## Citation

If you use this code, please cite the relevant papers for the algorithms you use.
