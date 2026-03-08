import mpmath as mp
from two_site_chain.sol_chain import eps_to_n_int, eps_to_n_half, alpha0, alpha1, alpha2

# ---------------- 1-loop letters ----------------
def _ws(x1: float, x2: float, y1: float, c: float):
    x1m, x2m, y1m, cm = mp.mpf(x1), mp.mpf(x2), mp.mpf(y1), mp.mpf(c)

    w1 = x1m + y1m + cm
    w2 = x2m + y1m + cm

    w3 = x1m + y1m - cm
    w4 = x2m + y1m - cm

    w5 = x1m - y1m + cm
    w6 = x2m - y1m + cm

    w7 = x1m - y1m - cm
    w8 = x2m - y1m - cm

    w9 = x1m + x2m + 2 * y1m
    w10 = x1m + x2m + 2 * cm
    w11 = x1m + x2m
    return w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11

def _w_select(x1: float, x2: float, y1: float, c: float, *idx):
    ws = _ws(x1, x2, y1, c)
    return tuple(ws[i - 1] for i in idx)

# ------------------------------------------------------
# ---------------- general eps solutions (I_k order TBD do not use)---------------
# ------------------------------------------------------
def I1_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp0=None, _alp2=None):
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = _ws(x1, x2, y1, c)

    if _alp0 is None:
        _alp0 = alpha0(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = _alp0 * (w1 ** eps) * (w2 ** eps)

    # 2F1(1, -2eps; 1-eps; wi / wj)
    def _hg(z):
        return mp.hyper([1.0, -2.0 * eps], [1.0 - eps], z)

    term9 = _alp2 * (w9 ** (2.0 * eps)) * (_hg(w3 / w9) + _hg(w4 / w9))
    term10 = _alp2 * (w10 ** (2.0 * eps)) * (_hg(w5 / w10) + _hg(w6 / w10))
    term11 = _alp2 * (w11 ** (2.0 * eps)) * (_hg(w7 / w11) + _hg(w8 / w11))

    return term0 - (term9 + term10 - term11)

def I2_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w2, w3, w9 = _w_select(x1, x2, y1, c, 2, 3, 9)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = _alp1 * (w2 ** eps) * (w3 ** eps)

    u1 = w3 / w2
    u2 = w9 / w2
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term1 = _alp2 * (u1 ** eps) * (w9 ** (2.0 * eps)) * hg

    return term0 + term1

def I3_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w1, w4, w9 = _w_select(x1, x2, y1, c, 1, 4, 9)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = _alp1 * (w1 ** eps) * (w4 ** eps)

    u1 = w4 / w1
    u2 = w9 / w1
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term1 = _alp2 * (u1 ** eps) * (w9 ** (2.0 * eps)) * hg

    return term0 + term1

def I4_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w2, w5, w10 = _w_select(x1, x2, y1, c, 2, 5, 10)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = _alp1 * (w2 ** eps) * (w5 ** eps)

    u1 = w5 / w2
    u2 = w10 / w2
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term1 = _alp2 * (u1 ** eps) * (w10 ** (2.0 * eps)) * hg

    return term0 + term1

def I5_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w1, w6, w10 = _w_select(x1, x2, y1, c, 1, 6, 10)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = _alp1 * (w1 ** eps) * (w6 ** eps)

    u1 = w6 / w1
    u2 = w10 / w1
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term1 = _alp2 * (u1 ** eps) * (w10 ** (2.0 * eps)) * hg

    return term0 + term1

def I6_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w2, w7, w11 = _w_select(x1, x2, y1, c, 2, 7, 11)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = -_alp1 * (w2 ** eps) * (w7 ** eps)

    u1 = w7 / w2
    u2 = w11 / w2
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term1 = -_alp2 * (u1 ** eps) * (w11 ** (2.0 * eps)) * hg

    return term0 + term1

def I7_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w1, w8, w11 = _w_select(x1, x2, y1, c, 1, 8, 11)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term0 = -_alp1 * (w1 ** eps) * (w8 ** eps)

    u1 = w8 / w1
    u2 = w11 / w1
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term1 = -_alp2 * (u1 ** eps) * (w11 ** (2.0 * eps)) * hg

    return term0 + term1

def I8_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    (w9,) = _w_select(x1, x2, y1, c, 9)
    if _alp2 is None:
        _alp2 = alpha2(eps)
    return 2.0 * _alp2 * (w9 ** (2.0 * eps))

def I9_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    (w10,) = _w_select(x1, x2, y1, c, 10)
    if _alp2 is None:
        _alp2 = alpha2(eps)
    return 2.0 * _alp2 * (w10 ** (2.0 * eps))

def I10_eps(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    (w11,) = _w_select(x1, x2, y1, c, 11)
    if _alp2 is None:
        _alp2 = alpha2(eps)
    return -2.0 * _alp2 * (w11 ** (2.0 * eps))

# -----------------------------------------
# ---------------- eps = 0 ----------------
# -----------------------------------------
def I1_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w2, w3, w4, w5, w6, w7, w8, _, _, _ = _ws(x1, x2, y1, c)

    return (-(mp.pi**2) / 6
            + mp.polylog(2, w3 / w1)
            + mp.polylog(2, w4 / w2)
            - mp.polylog(2, (w3 * w4) / (w1 * w2))
            + mp.polylog(2, w5 / w1)
            + mp.polylog(2, w6 / w2)
            - mp.polylog(2, (w5 * w6) / (w1 * w2))
            - mp.polylog(2, w7 / w1)
            - mp.polylog(2, w8 / w2)
            + mp.polylog(2, (w7 * w8) / (w1 * w2)))

def I2_eps0(x1: float, x2: float, y1: float, c: float):
    w2, w9 = _w_select(x1, x2, y1, c, 2, 9)
    return mp.log(w9 / w2)

def I5_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w9 = _w_select(x1, x2, y1, c, 1, 9)
    return mp.log(w9 / w1)

def I3_eps0(x1: float, x2: float, y1: float, c: float):
    w2, w10 = _w_select(x1, x2, y1, c, 2, 10)
    return mp.log(w10 / w2)

def I6_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w10 = _w_select(x1, x2, y1, c, 1, 10)
    return mp.log(w10 / w1)

def I4_eps0(x1: float, x2: float, y1: float, c: float):
    w2, w11 = _w_select(x1, x2, y1, c, 2, 11)
    return -mp.log(w11 / w2)

def I7_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w11 = _w_select(x1, x2, y1, c, 1, 11)
    return -mp.log(w11 / w1)

def I8_eps0(_x1: float, _x2: float, _y1: float, _c: float):
    return mp.mpf("1")

def I9_eps0(_x1: float, _x2: float, _y1: float, _c: float):
    return mp.mpf("1")

def I10_eps0(_x1: float, _x2: float, _y1: float, _c: float):
    return mp.mpf("-1")

# ------------------------------------------
# ---------------- eps = -n ----------------
# ------------------------------------------
def _jacobi_P_int(n: int, x):
    z = (1 - x) / 2
    return mp.binomial(2*n - 1, n - 1) * mp.hyper([1 - n, n], [n + 1], z)

def _int_branch_series_sum(w_main, w_main2, n: int):
    # Sum_{k=0}^{n-1} C(n+k,k) * w_main^k * w_main2^(n-k-1)
    return mp.fsum(
        mp.binomial(n + k, k) * (w_main ** k) * (w_main2 ** (n - k - 1))
        for k in range(n)
    )

def I1_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = _ws(x1, x2, y1, c)

    term0 = 1 / (w1**n * w2**n)

    def _branch(wi, wj, wn):
        x = 1 - 2 * wi / wn
        return ( (wj**n) * _jacobi_P_int(n, x) ) / (wn**(2*n))

    b9 = (_branch(w3, w1, w9) + _branch(w4, w2, w9))
    b10 = (_branch(w5, w1, w10) + _branch(w6, w2, w10))
    b11 = (_branch(w7, w1, w11) + _branch(w8, w2, w11))

    return term0 - (b9 / (w1**n * w2**n)) - (b10 / (w1**n * w2**n)) + (b11 / (w1**n * w2**n))

def I2_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w3, w9 = _w_select(x1, x2, y1, c, 2, 3, 9)
    s29 = _int_branch_series_sum(w2, w9, n)
    return -(w3 / (w2**n * w9**(2*n))) * s29

def I5_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w4, w9 = _w_select(x1, x2, y1, c, 1, 4, 9)
    s19 = _int_branch_series_sum(w1, w9, n)
    return -(w4 / (w1**n * w9**(2*n))) * s19

def I3_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w5, w10 = _w_select(x1, x2, y1, c, 2, 5, 10)
    s210 = _int_branch_series_sum(w2, w10, n)
    return -(w5 / (w2**n * w10**(2*n))) * s210

def I6_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w6, w10 = _w_select(x1, x2, y1, c, 1, 6, 10)
    s110 = _int_branch_series_sum(w1, w10, n)
    return -(w6 / (w1**n * w10**(2*n))) * s110

def I4_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w7, w11 = _w_select(x1, x2, y1, c, 2, 7, 11)
    s211 = _int_branch_series_sum(w2, w11, n)
    return (w7 / (w2**n * w11**(2*n))) * s211

def I7_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w8, w11 = _w_select(x1, x2, y1, c, 1, 8, 11)
    s111 = _int_branch_series_sum(w1, w11, n)
    return (w8 / (w1**n * w11**(2*n))) * s111

def I8_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w9,) = _w_select(x1, x2, y1, c, 9)
    return mp.binomial(2*n, n) / (w9 ** (2*n))

def I9_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w10,) = _w_select(x1, x2, y1, c, 10)
    return mp.binomial(2*n, n) / (w10 ** (2*n))

def I10_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w11,) = _w_select(x1, x2, y1, c, 11)
    return -mp.binomial(2*n, n) / (w11 ** (2*n))

# ---------------------------------------------------------------------------
# ---------------- eps = -(2n-1)/2 (I_k order TBD do not use)----------------
# ---------------------------------------------------------------------------
def _odd_double_factorial(m: int):
    out = mp.mpf("1")
    for k in range(1, m + 1, 2):
        out *= k
    return out

def _Poly_half(n: int, a, b):
    a = mp.mpf(a) if isinstance(a, (int, float)) else a
    b = mp.mpf(b) if isinstance(b, (int, float)) else b

    if n == 1:
        return mp.mpf("1")
    if n == 2:
        return 3*(a**2) + 8*a*b - 3*(b**2)
    if n == 3:
        return 15*(a**4) + 70*(a**3)*b + 128*(a**2)*(b**2) - 70*a*(b**3) - 15*(b**4)
    if n == 4:
        return (105*(a**6) + 700*(a**5)*b + 1981*(a**4)*(b**2) + 3072*(a**3)*(b**3)
                - 1981*(a**2)*(b**4) - 700*a*(b**5) - 105*(b**6))
    raise ValueError("Half-integer branch currently supports n=1..4 only (Pn provided up to P4).")

def _eps_half_core(a, b, wn, n: int, *, pref=-2.0*mp.pi):
    df = _odd_double_factorial(2 * n - 1)
    pow_ab = a ** (n - 0.5) * b ** (n - 0.5)
    term1 = (1 / pow_ab) * mp.atan(mp.sqrt(b / a))

    poly = _Poly_half(n, a, b)
    term2 = poly / (df * (pow_ab * wn ** (2 * n - 1)))

    return pref * (term1 - term2)

def I2_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w3, w9 = _w_select(x1, x2, y1, c, 2, 3, 9)
    return _eps_half_core(w2, w3, w9, n, pref=-2.0*mp.pi)

def I3_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w4, w9 = _w_select(x1, x2, y1, c, 1, 4, 9)
    return _eps_half_core(w1, w4, w9, n, pref=-2.0*mp.pi)

def I4_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w5, w10 = _w_select(x1, x2, y1, c, 2, 5, 10)
    return _eps_half_core(w2, w5, w10, n, pref=-2.0*mp.pi)

def I5_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w6, w10 = _w_select(x1, x2, y1, c, 1, 6, 10)
    return _eps_half_core(w1, w6, w10, n, pref=-2.0*mp.pi)

def I6_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w7, w11 = _w_select(x1, x2, y1, c, 2, 7, 11)
    return _eps_half_core(w2, w7, w11, n, pref=2.0*mp.pi)

def I7_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w8, w11 = _w_select(x1, x2, y1, c, 1, 8, 11)
    return _eps_half_core(w1, w8, w11, n, pref=2.0*mp.pi)

def I8_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    (w9,) = _w_select(x1, x2, y1, c, 9)
    coeff = (2**(4*n - 1)) * mp.pi / (n * mp.binomial(2*n, n))
    return coeff / (w9 ** (2*n - 1))

def I9_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    (w10,) = _w_select(x1, x2, y1, c, 10)
    coeff = (2**(4*n - 1)) * mp.pi / (n * mp.binomial(2*n, n))
    return coeff / (w10 ** (2*n - 1))

def I10_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    (w11,) = _w_select(x1, x2, y1, c, 11)
    coeff = (2**(4*n - 1)) * mp.pi / (n * mp.binomial(2*n, n))
    return -coeff / (w11 ** (2*n - 1))

def I1_eps_half(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w2 = _w_select(x1, x2, y1, c, 1, 2)

    return ((mp.pi**2) / (w1 ** (n - 0.5) * w2 ** (n - 0.5))
            + I2_eps_half(x1, x2, y1, n, c)
            + I3_eps_half(x1, x2, y1, n, c)
            - I8_eps_half(x1, x2, y1, n, c)
            + I4_eps_half(x1, x2, y1, n, c)
            + I5_eps_half(x1, x2, y1, n, c)
            - I9_eps_half(x1, x2, y1, n, c)
            + I6_eps_half(x1, x2, y1, n, c)
            + I7_eps_half(x1, x2, y1, n, c)
            - I10_eps_half(x1, x2, y1, n, c))

# ---------------- final solutions ----------------
EPS_TOL = mp.mpf("1e-12")

def I1_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp0=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I1_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I1_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I1_eps_half(x1, x2, y1, n_half, c)

    return I1_eps(x1, x2, y1, eps, c, _alp0=_alp0, _alp2=_alp2)

def I2_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I2_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I2_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I2_eps_half(x1, x2, y1, n_half, c)

    return I2_eps(x1, x2, y1, eps, c, _alp1=_alp1, _alp2=_alp2)

def I3_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I3_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I3_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I3_eps_half(x1, x2, y1, n_half, c)

    return I3_eps(x1, x2, y1, eps, c, _alp1=_alp1, _alp2=_alp2)

def I4_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I4_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I4_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I4_eps_half(x1, x2, y1, n_half, c)

    return I4_eps(x1, x2, y1, eps, c, _alp1=_alp1, _alp2=_alp2)

def I5_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I5_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I5_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I5_eps_half(x1, x2, y1, n_half, c)

    return I5_eps(x1, x2, y1, eps, c, _alp1=_alp1, _alp2=_alp2)

def I6_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I6_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I6_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I6_eps_half(x1, x2, y1, n_half, c)

    return I6_eps(x1, x2, y1, eps, c, _alp1=_alp1, _alp2=_alp2)

def I7_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I7_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I7_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I7_eps_half(x1, x2, y1, n_half, c)

    return I7_eps(x1, x2, y1, eps, c, _alp1=_alp1, _alp2=_alp2)

def I8_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I8_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I8_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I8_eps_half(x1, x2, y1, n_half, c)

    return I8_eps(x1, x2, y1, eps, c, _alp2=_alp2)

def I9_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I9_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I9_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I9_eps_half(x1, x2, y1, n_half, c)

    return I9_eps(x1, x2, y1, eps, c, _alp2=_alp2)

def I10_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I10_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I10_eps_int(x1, x2, y1, n_int, c)

    n_half = eps_to_n_half(eps)
    if n_half is not None:
        return I10_eps_half(x1, x2, y1, n_half, c)

    return I10_eps(x1, x2, y1, eps, c, _alp2=_alp2)
