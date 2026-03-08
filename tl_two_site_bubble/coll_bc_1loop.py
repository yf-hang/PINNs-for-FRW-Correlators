import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

from tl_two_site_bubble.sol_1loop import (
    I1_fin,
    I2_fin,
    I3_fin,
    I4_fin,
    I5_fin,
    I6_fin,
    I7_fin,
    I8_fin,
    I9_fin,
    I10_fin,
)


def _normalize_output_part(value, default="both"):
    if value is None:
        value = default
    s = str(value).strip().lower()
    if s in {"both", "all", "reim", "complex"}:
        return "both"
    if s in {"re", "real"}:
        return "re"
    if s in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported output part: {value!r}. Expected one of Re/Im/Both.")


def _complex_to_output_channels(function_complex: np.ndarray, output_part="both") -> np.ndarray:
    part = _normalize_output_part(output_part)
    if part == "both":
        return np.concatenate([np.real(function_complex), np.imag(function_complex)], axis=1)
    if part == "re":
        return np.real(function_complex)
    return np.imag(function_complex)


def _eval_1loop_chunk(x_chunk, cy_val):
    x_arr = np.asarray(x_chunk, dtype=float)
    out = np.empty((x_arr.shape[0], 10), dtype=complex)
    for i, (x1, x2, y1, eps) in enumerate(x_arr):
        x1f, x2f, y1f, epsf = float(x1), float(x2), float(y1), float(eps)
        out[i, 0] = I1_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 1] = I2_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 2] = I3_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 3] = I4_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 4] = I5_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 5] = I6_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 6] = I7_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 7] = I8_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 8] = I9_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 9] = I10_fin(x1f, x2f, y1f, epsf, cy_val)
    return out


def compute_boundary_values_rescaled_1loop(
    x_quadruplet,
    cy_val,
    *,
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
):
    """
    Compute analytic values for multiple (x1, x2, y1, eps) points.
    Input shape: (N,4)
    Output: complex array (N,10) for [I1, ..., I10]
    """
    x_all = np.asarray(x_quadruplet, dtype=float)
    if x_all.ndim == 1:
        if x_all.shape[0] != 4:
            raise ValueError("single point must be shape (4,) = (x1,x2,y1,eps)")
        x_all = x_all.reshape(1, 4)

    if x_all.shape[1] != 4:
        raise ValueError(f"x_quadruplet must have 4 columns (x1,x2,y1,eps), got shape {x_all.shape}")

    n_pts = int(x_all.shape[0])
    nw = max(int(num_workers), 1)
    cs = max(int(chunk_size), 1)
    nmin = max(int(parallel_min_points), 1)

    if (nw == 1) or (n_pts < nmin):
        return _eval_1loop_chunk(x_all, cy_val)

    chunks = [x_all[i:i + cs] for i in range(0, n_pts, cs)]
    cy_vals = [float(cy_val)] * len(chunks)
    try:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            parts = list(pool.map(_eval_1loop_chunk, chunks, cy_vals))
        return np.concatenate(parts, axis=0)
    except (PermissionError, OSError):
        print("[warn] 1-loop target parallel disabled by runtime; fallback to single-process.")
        return _eval_1loop_chunk(x_all, cy_val)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def compute_function_target_from_xcoll_1loop(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    output_part="both",
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
) -> torch.Tensor:
    x_np = to_numpy(x_coll)

    if x_np.ndim != 2:
        raise ValueError(f"x_coll must be 2D tensor/array, got shape {x_np.shape}")

    if x_np.shape[1] == 3:
        quad_np = np.concatenate(
            [x_np[:, :3], np.full((x_np.shape[0], 1), float(eps_val))],
            axis=1,
        )
    elif x_np.shape[1] == 4:
        quad_np = x_np[:, :4]
    else:
        raise ValueError(f"x_coll must have 3 or 4 columns, got {x_np.shape}")

    function_complex = compute_boundary_values_rescaled_1loop(
        quad_np,
        cy_val,
        num_workers=num_workers,
        chunk_size=chunk_size,
        parallel_min_points=parallel_min_points,
    )
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)

    return torch.tensor(function_concat, dtype=torch.float32, device=x_coll.device)


def build_inputs_and_boundary_1loop(
    n_coll_pts,
    x1_lo,
    x1_hi,
    x2_lo,
    x2_hi,
    y1_lo,
    y1_hi,
    cy_val,
    eps_val,
    device,
    compute_function_target=False,
    output_part="both",
    target_total_bc=500,
    n_bc_edge=6,
    n_face_pts=40,
    n_corner_extra=5,
    bc_abs_cap=1e8,
):
    # ---------- collocation points ----------
    n = int(n_coll_pts)
    x1_all = np.random.uniform(x1_lo, x1_hi, size=n).astype(np.float64)
    x2_all = np.random.uniform(x2_lo, x2_hi, size=n).astype(np.float64)
    y1_all = np.random.uniform(y1_lo, y1_hi, size=n).astype(np.float64)

    x_coll = torch.tensor(
        np.stack((x1_all, x2_all, y1_all), axis=1),
        dtype=torch.float32,
        device=device,
    )

    # ---------- boundary points ----------
    vertices = np.array(
        [
            [x1_lo, x2_lo, y1_lo],
            [x1_hi, x2_lo, y1_lo],
            [x1_lo, x2_hi, y1_lo],
            [x1_hi, x2_hi, y1_lo],
            [x1_lo, x2_lo, y1_hi],
            [x1_hi, x2_lo, y1_hi],
            [x1_lo, x2_hi, y1_hi],
            [x1_hi, x2_hi, y1_hi],
        ],
        dtype=np.float64,
    )

    def lin_edge(start, end, n_pts):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        ts = np.linspace(0.0, 1.0, int(n_pts))
        return start[None, :] + (end - start)[None, :] * ts[:, None]

    edges = np.concatenate(
        [
            # x1 direction
            lin_edge([x1_lo, x2_lo, y1_lo], [x1_hi, x2_lo, y1_lo], n_bc_edge),
            lin_edge([x1_lo, x2_hi, y1_lo], [x1_hi, x2_hi, y1_lo], n_bc_edge),
            lin_edge([x1_lo, x2_lo, y1_hi], [x1_hi, x2_lo, y1_hi], n_bc_edge),
            lin_edge([x1_lo, x2_hi, y1_hi], [x1_hi, x2_hi, y1_hi], n_bc_edge),
            # x2 direction
            lin_edge([x1_lo, x2_lo, y1_lo], [x1_lo, x2_hi, y1_lo], n_bc_edge),
            lin_edge([x1_hi, x2_lo, y1_lo], [x1_hi, x2_hi, y1_lo], n_bc_edge),
            lin_edge([x1_lo, x2_lo, y1_hi], [x1_lo, x2_hi, y1_hi], n_bc_edge),
            lin_edge([x1_hi, x2_lo, y1_hi], [x1_hi, x2_hi, y1_hi], n_bc_edge),
            # y1 direction
            lin_edge([x1_lo, x2_lo, y1_lo], [x1_lo, x2_lo, y1_hi], n_bc_edge),
            lin_edge([x1_hi, x2_lo, y1_lo], [x1_hi, x2_lo, y1_hi], n_bc_edge),
            lin_edge([x1_lo, x2_hi, y1_lo], [x1_lo, x2_hi, y1_hi], n_bc_edge),
            lin_edge([x1_hi, x2_hi, y1_lo], [x1_hi, x2_hi, y1_hi], n_bc_edge),
        ],
        axis=0,
    )

    def sample_face_x1(x1_fixed, n_pts):
        xs = np.full(n_pts, x1_fixed)
        x2s = np.random.uniform(x2_lo, x2_hi, size=n_pts)
        y1s = np.random.uniform(y1_lo, y1_hi, size=n_pts)
        return np.stack([xs, x2s, y1s], axis=1)

    def sample_face_x2(x2_fixed, n_pts):
        x1s = np.random.uniform(x1_lo, x1_hi, size=n_pts)
        x2s = np.full(n_pts, x2_fixed)
        y1s = np.random.uniform(y1_lo, y1_hi, size=n_pts)
        return np.stack([x1s, x2s, y1s], axis=1)

    def sample_face_y1(y1_fixed, n_pts):
        x1s = np.random.uniform(x1_lo, x1_hi, size=n_pts)
        x2s = np.random.uniform(x2_lo, x2_hi, size=n_pts)
        y1s = np.full(n_pts, y1_fixed)
        return np.stack([x1s, x2s, y1s], axis=1)

    faces = np.concatenate(
        [
            sample_face_x1(x1_lo, n_face_pts),
            sample_face_x1(x1_hi, n_face_pts),
            sample_face_x2(x2_lo, n_face_pts),
            sample_face_x2(x2_hi, n_face_pts),
            sample_face_y1(y1_lo, n_face_pts),
            sample_face_y1(y1_hi, n_face_pts),
        ],
        axis=0,
    )

    dx = 0.1 * max(x1_hi - x1_lo, 1e-12)
    dy = 0.1 * max(x2_hi - x2_lo, 1e-12)
    dz = 0.1 * max(y1_hi - y1_lo, 1e-12)

    corner_extra = []
    for vx1, vx2, vy1 in vertices:
        xl, xh = (vx1, min(vx1 + dx, x1_hi)) if vx1 == x1_lo else (max(vx1 - dx, x1_lo), vx1)
        yl, yh = (vx2, min(vx2 + dy, x2_hi)) if vx2 == x2_lo else (max(vx2 - dy, x2_lo), vx2)
        zl, zh = (vy1, min(vy1 + dz, y1_hi)) if vy1 == y1_lo else (max(vy1 - dz, y1_lo), vy1)
        pts = np.random.uniform([xl, yl, zl], [xh, yh, zh], size=(n_corner_extra, 3))
        corner_extra.append(pts)
    corner_extra = np.concatenate(corner_extra, axis=0)

    n_fixed = vertices.shape[0] + edges.shape[0] + faces.shape[0] + corner_extra.shape[0]
    n_inner = max(int(target_total_bc) - n_fixed, 0)
    inner_points = np.random.uniform(
        [x1_lo, x2_lo, y1_lo],
        [x1_hi, x2_hi, y1_hi],
        size=(n_inner, 3),
    )

    x_b_all = np.concatenate([vertices, edges, faces, corner_extra, inner_points], axis=0)
    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    function_target = None
    if compute_function_target:
        function_target = compute_function_target_from_xcoll_1loop(
            x_coll,
            cy_val=cy_val,
            eps_val=eps_val,
            output_part=output_part,
        )

    quad_b = np.concatenate(
        [x_b_all, np.full((x_b_all.shape[0], 1), float(eps_val), dtype=float)],
        axis=1,
    )
    bc_complex = compute_boundary_values_rescaled_1loop(quad_b, cy_val)

    # Filter out boundary points where analytic values are non-finite
    # (e.g. points on singular letters such as w7=0 or w8=0).
    re_bc = np.real(bc_complex)
    im_bc = np.imag(bc_complex)
    finite_mask = np.isfinite(re_bc).all(axis=1) & np.isfinite(im_bc).all(axis=1)
    bounded_mask = (np.abs(re_bc) <= float(bc_abs_cap)).all(axis=1) & (
        np.abs(im_bc) <= float(bc_abs_cap)
    ).all(axis=1)
    keep_mask = finite_mask & bounded_mask

    n_bad_nonfinite = int((~finite_mask).sum())
    n_bad_large = int((finite_mask & (~bounded_mask)).sum())
    n_bad_total = int((~keep_mask).sum())
    if n_bad_total > 0:
        print(
            "[warn] build_inputs_and_boundary_1loop: "
            f"filtered {n_bad_total} / {bc_complex.shape[0]} boundary points "
            f"(non-finite={n_bad_nonfinite}, |value|>{bc_abs_cap:g}={n_bad_large})."
        )
        x_b_all = x_b_all[keep_mask]
        bc_complex = bc_complex[keep_mask]

    bc_concat = _complex_to_output_channels(bc_complex, output_part=output_part)
    bc_target = torch.tensor(bc_concat, dtype=torch.float32, device=device)
    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    return x_coll, x_b_tensor, bc_target, function_target
