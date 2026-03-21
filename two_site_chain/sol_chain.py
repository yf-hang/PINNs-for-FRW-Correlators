#import torch
#import numpy as np
import mpmath as mp
#from plot_tools.plot_bc_points import visualize_bc_points #visualize_collocation_points

# ---------------- letters ----------------
def _ws(x1: float, x2: float, c: float):
    x1m, x2m, cm = mp.mpf(x1), mp.mpf(x2), mp.mpf(c)
    w1 = x1m + cm
    w2 = x2m + cm
    w3 = x1m - cm
    w4 = x2m - cm
    w5 = x1m + x2m
    return w1, w2, w3, w4, w5

def _w_select(x1: float, x2: float, c: float, *idx):
    ws = _ws(x1, x2, c)
    return tuple(ws[i - 1] for i in idx)

def _eps_to_n_int(eps: float, tol=1e-12):
    epsm = mp.mpf(eps)
    # Support any negative integer eps = -n, n >= 1.
    n = int(mp.nint(-epsm))
    if n >= 1 and mp.fabs(epsm + n) < tol:
        return n
    return None

def _eps_to_n_pos_int(eps: float, tol=1e-12):
    epsm = mp.mpf(eps)
    n = int(mp.nint(epsm))
    if n >= 1 and mp.fabs(epsm - n) < tol:
        return n
    return None

def eps_to_n_int(eps):
    return _eps_to_n_int(eps)

def eps_to_n_pos_int(eps):
    return _eps_to_n_pos_int(eps)

# -----------------------------------------
# ---------------- eps = 0 ----------------
# -----------------------------------------
def I1_eps0(x1: float, x2: float, c: float):
    w1, w2, w3, w4, w5 = _w_select(x1, x2, c, 1, 2, 3, 4, 5)

    return (-(mp.pi**2)/6
            + mp.polylog(2, w3 / w1)
            + mp.polylog(2, w4 / w2)
            - mp.polylog(2, (w3 * w4) / (w1 * w2)))

def I2_eps0(x1: float, x2: float, c: float):
    w2, w5 = _w_select(x1, x2, c, 2, 5)

    return mp.log(w5 / w2)

def I3_eps0(x1: float, x2: float, c: float):
    w1, w5 = _w_select(x1, x2, c, 1, 5)

    return mp.log(w5 / w1)

def I4_eps0(_x1: float, _x2: float, _c: float):
    return mp.mpf("1")

# ------------------------------------------
# ---------------- eps = +n ----------------
# ------------------------------------------
def _pos_int_branch_sum(w_main, w_alt, n: int):
    coeff = mp.binomial(2 * n, n)
    series = mp.fsum(
        (
            mp.binomial(2 * n - 1, n + m - 1) * (w_alt ** (n + m)) * (w_main ** (n - m))
            - (mp.binomial(2 * n - 1, n - m - 1) if (n - m - 1) >= 0 else mp.mpf("0"))
            * (w_main ** (n + m)) * (w_alt ** (n - m))
        ) / m
        for m in range(1, n + 1)
    )
    prefactor = ((-1) ** (n + 1)) * mp.mpf("0.5") * (w_main ** n) * (w_alt ** n)
    log_term = mp.mpf("0.5") * (w_main ** n) * (w_alt ** n) * mp.log(w_alt / w_main)
    return prefactor + log_term + series / coeff

def I2_eps_pos_int(x1: float, x2: float, n: int, c: float):
    w2, w3 = _w_select(x1, x2, c, 2, 3)

    return _pos_int_branch_sum(w2, w3, n)

def I3_eps_pos_int(x1: float, x2: float, n: int, c: float):
    w1, w4 = _w_select(x1, x2, c, 1, 4)

    return _pos_int_branch_sum(w1, w4, n)

def I4_eps_pos_int(x1: float, x2: float, n: int, c: float):
    (w5,) = _w_select(x1, x2, c, 5)

    return (w5 ** (2 * n)) / (n * mp.binomial(2 * n, n))

def I1_eps_pos_int(x1: float, x2: float, n: int, c: float):
    w1, w2 = _w_select(x1, x2, c, 1, 2)
    I2t = I2_eps_pos_int(x1, x2, n, c)
    I3t = I3_eps_pos_int(x1, x2, n, c)
    I4t = I4_eps_pos_int(x1, x2, n, c)

    return (w1 ** n) * (w2 ** n) + I2t + I3t - I4t

# ------------------------------------------
# ---------------- eps = -n ----------------
# ------------------------------------------
def _jacobi_P(n: int, x):
    z = (1 - x) / 2
    return mp.binomial(2*n - 1, n - 1) * mp.hyper([1 - n, n], [n + 1], z)

def _int_branch_series_sum(w_main, w5, n: int):
    # Sum_{k=0}^{n-1} C(n+k,k) * w_main^k * w5^(n-k-1)
    return mp.fsum(
        mp.binomial(n + k, k) * (w_main ** k) * (w5 ** (n - k - 1))
        for k in range(n)
    )

def I1_eps_int(x1: float, x2: float, n: int, c: float):
    w1, w2, w3, w4, w5 = _w_select(x1, x2, c, 1, 2, 3, 4, 5)

    term1 = 1 / (w1**n * w2**n)

    x1 = 1 - 2 * w3 / w5
    x2 = 1 - 2 * w4 / w5
    term2 =  _jacobi_P(n, x1) / (w2**n * w5**n) + _jacobi_P(n, x2) / (w1**n * w5**n)

    return term1 - term2

def I2_eps_int(x1: float, x2: float, n: int, c: float):
    w2, w3, w5 = _w_select(x1, x2, c, 2, 3, 5)

    sum_w2w5 = _int_branch_series_sum(w2, w5, n)

    return -(w3 / (w2**n * w5**(2*n))) * sum_w2w5

def I3_eps_int(x1: float, x2: float, n: int, c: float):
    w1, w4, w5 = _w_select(x1, x2, c, 1, 4, 5)

    sum_w1w5 = _int_branch_series_sum(w1, w5, n)

    return -(w4 / (w1**n * w5**(2*n))) * sum_w1w5

def I4_eps_int(x1: float, x2: float, n: int, c: float):
    (w5,) = _w_select(x1, x2, c, 5)

    return mp.binomial(2*n, n) / (w5**(2*n))

# ---------------- final solutions ----------------
EPS_TOL = mp.mpf("1e-12")

def _raise_unsupported_eps(eps):
    raise ValueError(
        "sol_chain.py currently supports only "
        "eps=0, eps=+n, or eps=-n; "
        f"got eps={eps}."
    )


def I1_fin(x1: float, x2: float, eps: float, c: float):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I1_eps0(x1, x2, c)

    n_pos_int = _eps_to_n_pos_int(eps)
    if n_pos_int is not None:
        return I1_eps_pos_int(x1, x2, n_pos_int, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I1_eps_int(x1, x2, n_int, c)

    _raise_unsupported_eps(eps)


def I2_fin(x1: float, x2: float, eps: float, c: float):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I2_eps0(x1, x2, c)

    n_pos_int = _eps_to_n_pos_int(eps)
    if n_pos_int is not None:
        return I2_eps_pos_int(x1, x2, n_pos_int, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I2_eps_int(x1, x2, n_int, c)

    _raise_unsupported_eps(eps)


def I3_fin(x1: float, x2: float, eps: float, c: float):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I3_eps0(x1, x2, c)

    n_pos_int = _eps_to_n_pos_int(eps)
    if n_pos_int is not None:
        return I3_eps_pos_int(x1, x2, n_pos_int, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I3_eps_int(x1, x2, n_int, c)

    _raise_unsupported_eps(eps)


def I4_fin(x1: float, x2: float, eps: float, c: float):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I4_eps0(x1, x2, c)

    n_pos_int = _eps_to_n_pos_int(eps)
    if n_pos_int is not None:
        return I4_eps_pos_int(x1, x2, n_pos_int, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I4_eps_int(x1, x2, n_int, c)

    _raise_unsupported_eps(eps)
