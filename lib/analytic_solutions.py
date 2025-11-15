import torch
import numpy as np
import mpmath as mp
from plot_tools.plot_bc_points import visualize_bc_points, visualize_collocation_points

# ---------------- analytic coefficients ----------------
def cd_of_eps(eps: float) -> float:
    sin_pi = mp.sin(mp.pi * eps)
    csc_pi = 1.0 / sin_pi
    return (mp.pi ** 2) * (csc_pi ** 2)

def cf_of_eps(eps: float) -> float:
    sin_pi = mp.sin(mp.pi * eps)
    sin_2pi = mp.sin(2.0 * mp.pi * eps)
    return - (mp.pi ** 2) * (1.0 / sin_pi) * (1.0 / sin_2pi)

def cz_of_eps(eps: float) -> float:
    sin_pi = mp.sin(mp.pi * eps)
    csc_pi = 1.0 / sin_pi
    return (4.0 ** (-eps)) * mp.sqrt(mp.pi) * csc_pi * mp.gamma(eps) * mp.gamma(0.5 - eps)

# ---------------- analytic solutions of integral family ----------------
def p_sol(x1: float, x2: float, eps: float,
          cy_val: float,
          cz_val=None, cd_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    if cd_val is None:
        cd_val = cd_of_eps(eps)

    term1 = cd_val * (complex(x1 + cy_val) ** eps) * (complex(x2 + cy_val) ** eps)
    base = complex(x1 + x2) ** (2.0 * eps)
    # hypergeometric args for p
    u_p1 = (cy_val - x2) / (cy_val + x1)
    u_p2 = (cy_val - x1) / (cy_val + x2)
    hg1 = mp.hyper([1.0, eps], [1.0 - eps], u_p1)
    hg2 = mp.hyper([1.0, eps], [1.0 - eps], u_p2)
    term2 = cz_val * base * (1.0 - 2.0 * hg1 - 2.0 * hg2)

    return complex(term1 + term2)

def f_sol(x1: float, x2: float, eps: float,
          cy_val: float,
          cz_val=None, cf_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    if cf_val is None:
        cf_val = cf_of_eps(eps)

    a1 = complex(x1 - cy_val)
    a2 = complex(x2 + cy_val)
    term1 = cf_val * (a1 ** eps) * (a2 ** eps)

    u_f = (x1 + x2) / (x2 + cy_val)
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], complex(u_f, 1e-12))
    term2 = (0.5 * cz_val
             * ((a1 / a2) ** eps)
             * (complex(x1 + x2) ** (2.0 * eps))
             * hg)
    return complex(term1 + term2)

def ft_sol(x1: float, x2: float, eps: float,
           cy_val: float,
           cz_val=None, cf_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    if cf_val is None:
        cf_val = cf_of_eps(eps)

    a1 = complex(x2 - cy_val)
    a2 = complex(x1 + cy_val)
    term1 = cf_val * (a1 ** eps) * (a2 ** eps)

    u_f = (x1 + x2) / (x1 + cy_val)
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], complex(u_f, 1e-12))
    term2 = (0.5 * cz_val
             * ((a1 / a2) ** eps)
             * (complex(x1 + x2) ** (2.0 * eps))
             * hg)
    return complex(term1 + term2)

def q_sol(x1: float, x2: float, eps: float, cz_val=None) -> complex:
    if cz_val is None:
        cz_val = cz_of_eps(eps)
    return cz_val * (complex(x1 + x2) ** (2.0 * eps))

# ---------------- vectorized boundary computation ----------------
def compute_boundary_values_rescaled(x_pair, eps, cy_val):
    """
    Compute analytic boundary values for multiple (x1, x2) pairs.
    Input: x_pair (N,2)
    Output: complex array (N,4) for [p, f, ft, q]
    """
    x_pair = np.array(x_pair)
    if x_pair.ndim == 1:
        assert x_pair.shape[0] == 2, "single point must be shape (2,)"
        x_pair = x_pair.reshape(1, 2)
    elif x_pair.ndim == 2:
        if x_pair.shape[1] != 2 and x_pair.shape[0] == 2:
            x_pair = x_pair.T
    else:
        raise ValueError(f"Unsupported ndim for x_pair: {x_pair.ndim}")

    cz_val, cd_val, cf_val = cz_of_eps(eps), cd_of_eps(eps), cf_of_eps(eps)
    results = []
    for x1, x2 in x_pair:
        pv = p_sol(float(x1), float(x2), eps, cy_val, cz_val, cd_val)
        fv = f_sol(float(x1), float(x2), eps, cy_val, cz_val, cf_val)
        ftv = ft_sol(float(x1), float(x2), eps, cy_val, cz_val, cf_val)
        qv = q_sol(float(x1), float(x2), eps, cz_val)
        results.append([pv, fv, ftv, qv])
    return np.array(results, dtype=complex)

# ---------------- collocation + boundary generation ----------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)

def build_inputs_and_boundary(
    n_coll_pts,
    x1_lo, x1_hi, x2_lo, x2_hi,
    cy_val, eps_val,
    use_eps_as_input,
    device):
    """
    Generate collocation points, boundary points, and analytic targets.
    Returns:
      x_coll: (N, 2) or (N, 3)
      x_b_tensor: (M, 2) or (M, 3)
      bc_target: (M, 8)
      function_target: (N, 8)
    """
    # ---------- collocation points with focus region ----------
    """
    focus_ratio = 0.0  # x % of collocation points in focus region
    focus_x1_max = x1_lo + 1.5
    focus_x2_max = x2_lo + 1.5

    n_focus = int(n_coll_pts * focus_ratio)
    n_global = n_coll_pts - n_focus

    # global uniform points
    x1_global = np.random.uniform(x1_lo, x1_hi, n_global)
    x2_global = np.random.uniform(x2_lo, x2_hi, n_global)

    # focus region (bottom-left)
    x1_focus = np.random.uniform(x1_lo, focus_x1_max, n_focus)
    x2_focus = np.random.uniform(x2_lo, focus_x2_max, n_focus)

    # merge and shuffle
    x1_all = np.concatenate([x1_global, x1_focus])
    x2_all = np.concatenate([x2_global, x2_focus])
    idx = np.random.permutation(len(x1_all))
    x1_all, x2_all = x1_all[idx], x2_all[idx]

    x_coll_xy = np.stack((x1_all, x2_all), axis=1)
    x_coll_xy = torch.tensor(x_coll_xy, dtype=torch.float32, device=device)
    """
    # ---------- collocation points with reduced density in bottom-left ----------
    reduce_ratio = 0.0 # drop x% point at the given region below
    focus_x1_max = x1_lo + 1.5
    focus_x2_max = x2_lo + 1.5

    x1_all = np.random.uniform(x1_lo, x1_hi, n_coll_pts)
    x2_all = np.random.uniform(x2_lo, x2_hi, n_coll_pts)

    # find points at the given region
    mask_focus = (x1_all < focus_x1_max) & (x2_all < focus_x2_max)

    # drop points at the given region
    mask_keep = np.ones_like(mask_focus, dtype=bool)
    drop_idx = np.where(mask_focus)[0]
    n_drop = int(len(drop_idx) * reduce_ratio)
    if n_drop > 0:
        drop_choice = np.random.choice(drop_idx, size=n_drop, replace=False)
        mask_keep[drop_choice] = False

    # keep last points
    x1_all = x1_all[mask_keep]
    x2_all = x2_all[mask_keep]

    # keep all points unchanged
    n_needed = n_coll_pts - len(x1_all)
    if n_needed > 0:
        x1_extra = np.random.uniform(focus_x1_max, x1_hi, n_needed)
        x2_extra = np.random.uniform(focus_x2_max, x2_hi, n_needed)
        x1_all = np.concatenate([x1_all, x1_extra])
        x2_all = np.concatenate([x2_all, x2_extra])

    idx = np.random.permutation(len(x1_all))
    x1_all, x2_all = x1_all[idx], x2_all[idx]

    x_coll_xy = np.stack((x1_all, x2_all), axis=1)
    x_coll_xy = torch.tensor(x_coll_xy, dtype=torch.float32, device=device)

    # ----------------------------
    visualize_collocation_points(x1_all, x2_all,
                                 focus_x1_max, focus_x2_max)

    # ----------------------------
    if use_eps_as_input:
        eps_col = (torch.rand(n_coll_pts, 1, device=device) - 0.5) * 0.02
        x_coll = torch.cat([x_coll_xy, eps_col], dim=1)
    else:
        x_coll = x_coll_xy

    # ---------- boundary points (improved coverage) ----------
    n_bc_edge = 12 # n_bc_edge points for each edge
    rng = np.random.default_rng(seed=0)

    x1_grid = np.linspace(x1_lo, x1_hi, n_bc_edge)
    x2_grid = np.linspace(x2_lo, x2_hi, n_bc_edge)

    bc_bottom = np.stack([x1_grid, np.full_like(x1_grid, x2_lo)], axis=1)
    bc_top    = np.stack([x1_grid, np.full_like(x1_grid, x2_hi)], axis=1)
    bc_left   = np.stack([np.full_like(x2_grid, x1_lo), x2_grid], axis=1)
    bc_right  = np.stack([np.full_like(x2_grid, x1_hi), x2_grid], axis=1)

    x_b_all = np.concatenate([bc_bottom, bc_top, bc_left, bc_right], axis=0)

    # Add mild noise to break symmetry
    jitter = 0.0
    w1, w2 = (x1_hi - x1_lo), (x2_hi - x2_lo)
    noise = rng.uniform(-jitter, jitter, size=x_b_all.shape)
    x_b_all[:, 0] = np.clip(x_b_all[:, 0] + noise[:, 0] * w1, x1_lo, x1_hi)
    x_b_all[:, 1] = np.clip(x_b_all[:, 1] + noise[:, 1] * w2, x2_lo, x2_hi)

    # Add anchors inside domain
    anchors = np.array([
        [(x1_lo + x1_hi) / 2, (x2_lo + x2_hi) / 2]])
    """
    anchors = np.array([
        [(x1_lo + x1_hi) / 2, (x2_lo + x2_hi) / 2],
        [x1_lo + 1.0, x2_hi - 1.0],
        [x1_hi - 1.0, x2_lo + 1.0],
        [x1_lo + 1.0, x2_lo + 1.0],
        [x1_hi - 1.0, x2_hi - 1.0],
        [x1_lo + 2.0, x2_hi - 2.0],
        [x1_hi - 2.0, x2_lo + 2.0],
        [x1_lo + 2.0, x2_lo + 2.0],
        [x1_hi - 2.0, x2_hi - 2.0],
        [x1_lo + 3.0, x2_hi - 3.0],
        [x1_hi - 3.0, x2_lo + 3.0],
        [x1_lo + 3.0, x2_lo + 3.0],
        [x1_hi - 3.0, x2_hi - 3.0],
    ])
    """
    # ---- Add extra random points in the lower-left subregion ----
    #n_extra = 20  # number of additional points
    #extra_x1 = np.random.uniform(x1_lo, x1_lo + 1.5, n_extra)
    #extra_x2 = np.random.uniform(x2_lo, x2_lo + 1.5, n_extra)
    #extra_points = np.stack([extra_x1, extra_x2], axis=1)  # (n_extra, 2)

    # ---- Extra BC points along diagonals (and near-diagonals) ----
    add_diag_points = True
    n_per_line = 8
    d_list = [0.2, -0.2]  # deviation distance

    extra_points_list = []

    if add_diag_points:
        t = np.linspace(0.0, 1.0, n_per_line)

        # main diagonal x2 = x1 and its ±d orthogonal deviation
        x_main = x1_lo + t * (x1_hi - x1_lo)
        y_main = x_main.copy()

        # main diagonal
        diag_main = np.stack([x_main, y_main], axis=1)
        extra_points_list.append(diag_main)

        # orthogonal to main diagonal n = (1, -1)/√2
        for d in d_list:
            dx = d / np.sqrt(2.0)
            diag_offset = np.stack([x_main + dx, y_main - dx], axis=1)
            extra_points_list.append(diag_offset)

        # diagonal x2 = -x1 + c its ±d orthogonal deviation
        # c is the constant
        c = (x1_lo + x1_hi) / 2 + (x2_lo + x2_hi) / 2

        x_ortho = x1_lo + t * (x1_hi - x1_lo)
        y_ortho = c - x_ortho
        diag_ortho = np.stack([x_ortho, y_ortho], axis=1)
        extra_points_list.append(diag_ortho)

        # n_ortho = (1, 1)/\sqrt{2}
        for d in d_list:
            dx = d / np.sqrt(2.0)
            diag_offset = np.stack([x_ortho + dx, y_ortho + dx], axis=1)
            extra_points_list.append(diag_offset)

        extra_points = np.concatenate(extra_points_list, axis=0)
        extra_points[:, 0] = np.clip(extra_points[:, 0], x1_lo, x1_hi)
        extra_points[:, 1] = np.clip(extra_points[:, 1], x2_lo, x2_hi)

    x_b_all = np.concatenate([x_b_all,
                              anchors,
                              extra_points
                              ],
                             axis=0) # (M, 2)

    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)
    eps_seed = float(eps_val)

    # ---------- function target (N,8) ----------
    xy_np = to_numpy(x_coll)[:, :2]
    function_complex = compute_boundary_values_rescaled(xy_np, eps_seed, cy_val)
    function_concat = np.concatenate(
        [np.real(function_complex), np.imag(function_complex)], axis=1)
    function_target = torch.tensor(function_concat, dtype=torch.float32, device=device)

    # ---------- boundary target (M,8) ----------
    bc_complex = compute_boundary_values_rescaled(x_b_all, eps_seed, cy_val)
    bc_concat = np.concatenate([np.real(bc_complex), np.imag(bc_complex)], axis=1)
    bc_target = torch.tensor(bc_concat, dtype=torch.float32, device=device)

    if use_eps_as_input:
        eps_col_b = torch.full((x_b_tensor.shape[0], 1), eps_seed, device=device)
        x_b_tensor = torch.cat([x_b_tensor, eps_col_b], dim=1)

    visualize_bc_points(x1_lo, x1_hi, x2_lo, x2_hi, x_b_all)

    return x_coll, x_b_tensor, bc_target, function_target
