## PINNs-for-FRW-Correlators

This repository contains a Physics-Informed Neural Network (PINN) framework designed to solve the **canonical differential equations (CDEs)** arising in the computation of **cosmological wavefunction coefficients** for power-law FRW universes, following the structure introduced in  
**arXiv:2312.05303 – *Differential Equations for Cosmological Correlators*.**

The goal is to develop a machine-learning-based solver capable of reproducing the analytic structures of the correlators—including branch cuts, real/imag parts, and behavior across different cosmological regions—using a stable two-phase PINN training strategy.


### Overview

Cosmological correlators satisfy differential equations analogous to multiloop Feynman integrals. This project implements a PINN approach to solve the canonical DE system:

$\displaystyle 
\partial_{x_i} I(x_1, x_2)
= \varepsilon\, A_i(x_1, x_2)\, I(x_1, x_2),
$

where the matrices $A_i$ correspond to the FRW twist-parameter–dependent canonical form.

The code supports:
- Two-phase training (fixed vs trainable A matrices)
- Complex-valued PINN outputs (real + imaginary components)
- Automatic integrability checks
- Boundary-condition sampling for the Euclidean region
- Analytic benchmark solutions
- Visualization of loss curves, real-imag scatter, histograms, etc.
