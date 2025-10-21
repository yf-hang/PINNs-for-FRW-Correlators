import torch
import math
import numpy as np
from scipy.special import gamma, hyp2f1

# ---------------- analytic solutions of integral family ----------------
def cd_of_eps(eps: float) -> float:
    sin_pi = math.sin(math.pi * eps)
    csc_pi = 1.0 / sin_pi
    return (math.pi ** 2) * (csc_pi ** 2)

def cf_of_eps(eps: float) -> float:
    sin_pi = math.sin(math.pi * eps)
    sin_2pi = math.sin(2.0 * math.pi * eps)
    return - (math.pi ** 2) * (1.0 / sin_pi) * (1.0 / sin_2pi)

def cz_of_eps(eps: float) -> float:
    sin_pi = math.sin(math.pi * eps)
    csc_pi = 1.0 / sin_pi
    return (4.0 ** (-eps)) * math.sqrt(math.pi) * csc_pi * gamma(eps) * gamma(0.5 - eps)

def p_sol(x1: float, x2: float, eps: float, cy_val: float,
          cz_val=None, cd_val=None, y1_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    if cd_val is None:
        cd_val = cd_of_eps(eps)
    if y1_val is None:
        y1_val = y1
    term1 = cd_val * (complex(x1 + cy_val) ** eps) * (complex(x2 + cy_val) ** eps)
    base = complex(x1 + x2) ** (2.0 * eps)
    # hypergeometric args for p
    u_p1 = (cy_val - x2) / (cy_val + x1)
    u_p2 = (cy_val - x1) / (cy_val + x2)
    hg1 = hyp2f1(1.0, eps, 1.0 - eps, u_p1)
    hg2 = hyp2f1(1.0, eps, 1.0 - eps, u_p2)
    term2 = cz_val * base * (1.0 - 2.0 * hg1 - 2.0 * hg2)
    return ((y1_val / cy_val) ** (2 * eps)) * (term1 + term2)

def f_sol(x1: float, x2: float, eps: float, cy_val: float,
          cz_val=None, cf_val=None, y1_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    if cf_val is None:
        cf_val = cf_of_eps(eps)
    if y1_val is None:
        y1_val = y1
    a1 = complex(x1 - cy_val)
    a2 = complex(x2 + cy_val)
    term1 = cf_val * (a1 ** eps) * (a2 ** eps)
    # hypergeometric args for f
    u_f = (x1 + x2) / (x2 + cy_val)
    hyper = hyp2f1(eps, 2.0 * eps, 1.0 + 2.0 * eps, u_f)
    term2 = 0.5 * cz_val * ((a1 / a2) ** eps) * (complex(x1 + x2) ** (2.0 * eps)) * hyper
    return ((y1_val / cy_val) ** (2 * eps)) * (term1 + term2)

def ft_sol(x1: float, x2: float, eps: float, cy_val: float,
           cz_val=None, cf_val=None, y1_val=None) -> complex:
    # swap x1,x2 to obtain ft
    return f_sol(x2, x1, eps, cy_val, cz_val, cf_val, y1_val)

def q_sol(x1: float, x2: float, eps: float, cy_val: float,
          cz_val=None, y1_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    if y1_val is None:
        y1_val = y1
    return ((y1_val / cy_val) ** (2 * eps)) * cz_val * (complex(x1 + x2) ** (2.0 * eps))

def compute_boundary_values_rescaled(x_pair, eps, cy_val, y1_val):
    x1, x2 = float(x_pair[0]), float(x_pair[1])
    cz_val, cd_val, cf_val = cz_of_eps(eps), cd_of_eps(eps), cf_of_eps(eps)
    pv = p_sol(x1, x2, eps, cy_val, cz_val, cd_val, y1_val)
    fv = f_sol(x1, x2, eps, cy_val, cz_val, cf_val, y1_val)
    ftv = ft_sol(x1, x2, eps, cy_val, cz_val, cf_val, y1_val)
    qv = q_sol(x1, x2, eps, cy_val, cz_val, y1_val)
    return np.array([pv, fv, ftv, qv], dtype=complex)

# ---------------- hypergeometric arg safety (Euclidean region) ----------------
def is_safe_point(x1, x2, cy_val, min_letter_abs=1e-8, arg_max_safe=0.95):
    # letters w1 - w5
    letters = np.stack([
        x1 - cy_val,
        x1 + cy_val,
        x2 - cy_val,
        x2 + cy_val,
        x1 + x2
    ], axis=0)
    min_letter = min_letter_abs
    letter_ok = np.all(np.abs(letters) > min_letter, axis=0)

    # args
    up1 = (cy_val - x2) / (x1 + cy_val)
    up2 = (cy_val - x1) / (x2 + cy_val)
    uf = (x1 + x2) / (x2 + cy_val)
    uft = (x1 + x2) / (x1 + cy_val)
    arg_max = arg_max_safe
    hyp_ok = (np.abs(up1) < arg_max) & \
             (np.abs(up2) < arg_max) & \
             (np.abs(uf) < arg_max) & \
             (np.abs(uft) < arg_max)

    return letter_ok & hyp_ok

def generate_safe_collocation(n_pts, x1_lo, x1_hi, x2_lo, x2_hi, cy_val, device):
    batch = 2_000
    rng = np.random.default_rng(0)
    pts = []
    trials = 0
    while len(pts) < n_pts:
        x1 = rng.uniform(x1_lo, x1_hi, batch)
        x2 = rng.uniform(x2_lo, x2_hi, batch)
        trials += batch

        mask = is_safe_point(x1, x2, cy_val)
        if np.any(mask):
            safe_points = np.stack((x1[mask], x2[mask]), axis=1)
            pts.append(safe_points)

        if sum(len(a) for a in pts) >= n_pts:
            break

    pts = np.concatenate(pts, axis=0)[:n_pts]
    eff = n_pts / trials
    print(f"[collocation-fast] generated {n_pts} safe points (total trials={trials}, efficiency={eff:.3f})")

    return torch.tensor(pts, dtype=torch.float32, device=device)

def find_safe_boundary(x1_lo, x1_hi, x2_lo, x2_hi, cy_val):
    batch = 1_000
    max_trials = 2_000
    rng = np.random.default_rng(1)
    trials = 0

    while trials < max_trials:
        x1 = rng.uniform(x1_lo, x1_hi, batch)
        x2 = rng.uniform(x2_lo, x2_hi, batch)
        trials += batch

        mask = is_safe_point(x1, x2, cy_val)

        if np.any(mask):
            x1_safe = x1[mask][0]
            x2_safe = x2[mask][0]
            print(f"[boundary-fast] Found safe boundary after {trials} trials")
            return np.array([x1_safe, x2_safe], dtype=float)

    print("[boundary-fast] Warning: fallback to domain center")
    # defensive programming return to center point
    return np.array([0.5 * (x1_lo + x1_hi), 0.5 * (x2_lo + x2_hi)], dtype=float)

# ---------------- collocation + boundary generation ----------------
def build_inputs_and_boundary(n_coll_pts, x1_lo, x1_hi, x2_lo, x2_hi,
                              cy_val, y1_val, eps_val,
                              use_eps_as_input, device):
    # collocation in Euclidean region (safe)
    x_coll_xy = generate_safe_collocation(n_coll_pts, x1_lo, x1_hi, x2_lo, x2_hi, cy_val, device)  # (N,2)
    if use_eps_as_input:
        eps_col = (torch.rand(x_coll_xy.shape[0], 1, device=device) - 0.5) * 0.02  # small jitter
        x_coll = torch.cat([x_coll_xy, eps_col], dim=1)  # (N,3)
    else:
        x_coll = x_coll_xy  # (N,2)

    # boundary seed (safe)
    x_b = find_safe_boundary(x1_lo, x1_hi, x2_lo, x2_hi, cy_val)
    eps_seed = float(eps_val)
    # analytic boundary target
    bc_complex = compute_boundary_values_rescaled(x_b, eps_seed, cy_val, y1_val)
    bc_re = np.real(bc_complex)
    bc_im = np.imag(bc_complex)
    bc_target = torch.tensor(np.concatenate([bc_re, bc_im]),
                             dtype=torch.float32, device=device).unsqueeze(0)

    if use_eps_as_input:
        x_b_tensor = torch.tensor([[x_b[0], x_b[1], eps_seed]],
                                  dtype=torch.float32, device=device)
    else:
        x_b_tensor = torch.tensor([[x_b[0], x_b[1]]],
                                  dtype=torch.float32, device=device)
    return x_coll, x_b_tensor, bc_target