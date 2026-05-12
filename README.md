# CosmoPINNs

CosmoPINNs is a framework for solving canonical differential equations
(CDEs) for cosmological wavefunction coefficients with physics-informed neural
networks (PINNs). The project studies the two-site family of cosmological
wavefunction integrals in power-law FRW backgrounds and uses transfer learning
from a lower-loop source topology to higher-loop targets.

Result plots are also collected at:

<https://yf-hang.github.io/CosmoPINNs/>

## Overview

The core idea is to approximate the vector of master integrals (MIs) by a neural
network and train it directly against the canonical differential system. The
loss combines:

- loss of canonical differential equation (CDE) evaluated at collocation points;
- analytic boundary and anchor data that fix integration constants and
  normalization;
- post-training diagnostics against analytic solutions.

The code implements a three-phase hierarchy:

| Phase | Topology | Inputs | Real-sector MI dimension | Training role |
| --- | --- | --- | ---: | --- |
| Phase 1 | two-site chain, $\ell = 0$ | `(u1, u2)` | 4 | source model |
| Phase 2 | one-loop bubble, $\ell = 1$ | `(u1, u2, u3)` | 10 | transfer target |
| Phase 3 | two-loop sunset, $\ell = 2$ | `(u1, u2, u3, u4)` | 22 | transfer target |

For transfer learning, the Phase-1 hidden representation is copied into the
target model, frozen, and paired with new input and output layers matching the
loop-level topology.

## Scientific Setup

The default numerical setup follows the manuscript:

| $\ell$ | Collocation domain | Fixed scale | Collocation points | Boundary points | Epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | `[20,30] x [20,30]` | `c0 = 15` | `5e4` | `5e3` | `6000` |
| 1 | `[30,40] x [30,40] x [15,20]` | `c1 = 5` | `1e5` | `1e4` | `8000` |
| 2 | `[50,60] x [50,60] x [20,25] x [10,15]` | `c2 = 5` | `1.5e5` | `1.7e4` | `10000` |

The benchmark values of $\varepsilon$ are:

| $\ell$ | Background scan |
| --- | --- |
| 0 | $\varepsilon = \{0, -1, -2, -3, -4, +5\}$ |
| 1 | $\varepsilon = \{0, -1, -2, -3\}$ |
| 2 | $\varepsilon = \{0, -1, -4, +5\}$ |

Here $\varepsilon = 0$ is the dS benchmark and $\varepsilon = -1$ is the flat-space benchmark.
The remaining values correspond to the RD/MD transfer benchmarks used for the
one- and two-loop systems.

## Repository Layout

```text
.
|-- main.py                         # Main configuration-driven training entry point
|-- config.json                     # Default production/run configuration
|-- config_local_test.json          # Smaller or local-test configuration
|-- lib/
|   |-- models.py                   # PINN and transfer-PINN modules
|   |-- loss.py                     # CDE residual and boundary losses
|   `-- train.py                    # Optimizer, warmup, cosine schedule
|-- two_site_chain/                 # Phase-1 analytic targets and CDE matrices
|-- tl_two_site_bubble/             # Phase-2 one-loop transfer target
|-- tl_two_site_sunset/             # Phase-3 two-loop transfer target
|-- plot_tools/                     # Per-run plotting and post-training checks
|-- FinalResults/                   # Merged figures and result post-processing
|-- results/                        # Generated checkpoints, logs, plots, caches
```

## Installation

This is a research repository rather than an installable Python package. Run
commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib mpmath
```

For CUDA runs, install the PyTorch build that matches your CUDA driver before
running the training scripts.

## Running Training

Edit `config.json`, then launch:

```bash
python main.py
```

Important configuration fields:

| Key | Meaning |
| --- | --- |
| `device` | `auto`, `cpu`, or `cuda` |
| `eps_global` | fixed $\varepsilon$ value for the run |
| `enable_phase2`, `enable_phase3` | select the transfer target; only one can run in a single execution |
| `run_phase2_only`, `run_phase3_only` | skip Phase-1 training and load an existing Phase-1 checkpoint |
| `phase*_output_part` | `Re`, `Im`, or `Both`; manuscript runs use real-sector training |
| `lambda1`, `lambda2` | CDE and boundary loss weights |
| `reuse_saved_models`, `reuse_eval_bundle` | reuse saved artifacts when available |
| `phase*_model_load_path` | explicit checkpoint path for transfer or evaluation |

If `use_local_config` is set to `true` in `config.json`, `main.py` loads
`config_local_test.json` instead.

Generated artifacts are written under `results/` by default, or
`results_local_test/` when local-test mode is enabled. Each run stores
configuration snapshots, model checkpoints, loss histories, training logs,
diagnostic plots, and optional evaluation bundles.

## Transfer-Only Runs

To train a loop-level target from an existing Phase-1 checkpoint, set:

```json
{
  "run_phase2_only": true,
  "enable_phase2": true,
  "enable_phase3": false,
  "phase1_model_load_path": "path/to/P1_model_eps_...pt"
}
```

or use the analogous `run_phase3_only` / `enable_phase3` settings for Phase 3.
If `reuse_saved_models` is true and a matching Phase-1 checkpoint exists in the
standard output location, `main.py` can infer the checkpoint path automatically.

## Batch Utilities

The batch runners modify `config.json`, launch `main.py`, and restore the
original configuration unless `--keep-config` is used.

Preview a Phase-2 $\lambda_1$ scan:

```bash
python run_eps_global_batch.py --dry-run --python /path/to/python
```

Preview selected Phase-3 runs:

```bash
python run_lambda1_batch.py --dry-run --python /path/to/python
```

The default interpreter paths in these scripts are local machine paths, so pass
`--python` on a new system.

## Post-Processing

The `FinalResults/` scripts merge training histories and recompute diagnostics:

```bash
python FinalResults/merge_loss.py
python FinalResults/merge_L12.py --device auto
python FinalResults/merge_log.py
python FinalResults/merge_cos.py
```

See `FinalResults/README.md` for script-specific arguments and examples.

The main diagnostics are:

- total, CDE, and boundary loss histories,
- relative $\mathcal{L}_1$ and $\mathcal{L}_2$ norm-error distributions,
- log-ratio amplitude mismatch,
- cosine similarity $\mathcal{C}$ between predicted and analytic MI vectors.

Figures are stored in `FinalResults/`.

## Notes

The formulation implemented in this codebase originate from [arXiv:2410.17192](https://arxiv.org/abs/2410.17192), where the kinematic flow and CDEs for the relevant two-site loop-level cosmological wavefunction integrals were first analyzed and derived. We ask that papers using, discussing, or extending this formulation cite [arXiv:2410.17192](https://arxiv.org/abs/2410.17192) as the original work.
