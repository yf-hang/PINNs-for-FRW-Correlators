# CosmoPINNs

CosmoPINNs is a research codebase for solving canonical differential equations (CDEs) of cosmological wavefunction integrals with physics-informed neural networks (PINNs).

The project is organized as a three-phase hierarchy:

- `Phase 1`: two-site chain
- `Phase 2`: one-loop bubble, trained by transfer learning from matched Phase-1 checkpoints
- `Phase 3`: two-loop sunset, also initialized from matched Phase-1 checkpoints

## What This Repository Contains

- `main.py`: main training entry point for fixed-`epsilon` runs
- `lib/`: neural-network models, losses, and training utilities
- `two_site_chain/`: Phase-1 problem definition
- `tl_two_site_bubble/`: Phase-2 transfer-learning setup
- `tl_two_site_sunset/`: Phase-3 transfer-learning setup
- `plot_tools/`: plotting and post-training diagnostic utilities
- `FinalResults/`: merged figures, cached diagnostics, and post-processing scripts
- `ANote/`: manuscript drafts and LaTeX summaries
- `CHTC/`: batch and high-throughput computing helpers

## Main Features

- PINN training for fixed-`epsilon` cosmological CDE systems
- Transfer-learning workflow for higher-loop targets
- Built-in diagnostics for:
  - training losses
  - relative `L1/L2` errors
  - log-ratio amplitude mismatch
  - cosine-similarity shape alignment
- Manuscript-ready summaries and result plots

## Quick Start

1. Adjust the run settings in `config.json`.
2. Launch training from the project root:

```bash
python main.py
```

3. Use the scripts in `FinalResults/` to merge and visualize diagnostics.

## Notes

- The current production workflow is configuration-driven rather than packaged as a pip module.
- This repository is intended as research code, so some paths and runtime options are tuned for local or batch environments.
- A script-specific usage guide for the result-merging tools is available in `FinalResults/README.md`.
