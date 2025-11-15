## PINNs-for-FRW-Correlators

This repository contains a PINN framework designed to solve the **canonical differential equations (CDEs)** arising in the computation of **cosmological wavefunction coefficients** for power-law FRW universes.

### Overview

Cosmological correlators satisfy differential equations analogous to multiloop Feynman integrals. This project implements a PINN approach to solve the CDE system:

$\displaystyle 
d \vec{I}(x_1, x_2, \varepsilon)
= \varepsilon A(x_1, x_2) \vec{I}(x_1, x_2, \varepsilon),
$

where the $A = \sum_{i=1}^5 a_i dlog(w_i)$ with $a_i$ being the constant matrices.
