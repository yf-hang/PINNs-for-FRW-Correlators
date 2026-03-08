import itertools
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

from tl_two_site_sunset.sol_2loop import (
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
    I11_fin,
    I12_fin,
    I13_fin,
    I14_fin,
    I15_fin,
    I16_fin,
    I17_fin,
    I18_fin,
    I19_fin,
    I20_fin,
    I21_fin,
    I22_fin,
)

_ANALYTIC_FUNCS = (
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
    I11_fin,
    I12_fin,
    I13_fin,
    I14_fin,
    I15_fin,
    I16_fin,
    I17_fin,
    I18_fin,
    I19_fin,
    I20_fin,
    I21_fin,
    I22_fin,
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


def _resolve_n_basis(n_basis):
    if n_basis is None:
        return len(_ANALYTIC_FUNCS)
    n = int(n_basis)
    if n <= 0 or n > len(_ANALYTIC_FUNCS):
        raise ValueError(
            f"n_basis must be in [1, {len(_ANALYTIC_FUNCS)}], got {n_basis}."
        )
    return n


def _eval_2loop_chunk(x_chunk, cy_val, n_basis):
    x_arr = np.asarray(x_chunk, dtype=float)
    n_basis = _resolve_n_basis(n_basis)
    out = np.empty((x_arr.shape[0], n_basis), dtype=complex)
    funcs = _ANALYTIC_FUNCS[:n_basis]

    for i, (x1, x2, y1, y2, eps) in enumerate(x_arr):
        x1f = float(x1)
        x2f = float(x2)
        y1f = float(y1)
        y2f = float(y2)
        epsf = float(eps)
        for j, fn in enumerate(funcs):
            out[i, j] = fn(x1f, x2f, y1f, y2f, epsf, cy_val)
    return out


def compute_boundary_values_rescaled_2loop(
    x_quintuplet,
    cy_val,
    *,
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
    n_basis=22,
):
    """
    Compute analytic values for multiple (x1, x2, y1, y2, eps) points.
    Input shape: (N,5)
    Output: complex array (N, n_basis) for [I1, ..., I_n_basis].
    """
    x_all = np.asarray(x_quintuplet, dtype=float)
    n_basis = _resolve_n_basis(n_basis)

    if x_all.ndim == 1:
        if x_all.shape[0] != 5:
            raise ValueError("single point must be shape (5,) = (x1,x2,y1,y2,eps)")
        x_all = x_all.reshape(1, 5)

    if x_all.shape[1] != 5:
        raise ValueError(
            f"x_quintuplet must have 5 columns (x1,x2,y1,y2,eps), got shape {x_all.shape}"
        )

    n_pts = int(x_all.shape[0])
    nw = max(int(num_workers), 1)
    cs = max(int(chunk_size), 1)
    nmin = max(int(parallel_min_points), 1)

    if (nw == 1) or (n_pts < nmin):
        return _eval_2loop_chunk(x_all, cy_val, n_basis)

    chunks = [x_all[i : i + cs] for i in range(0, n_pts, cs)]
    cy_vals = [float(cy_val)] * len(chunks)
    n_basis_vals = [int(n_basis)] * len(chunks)
    try:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            parts = list(pool.map(_eval_2loop_chunk, chunks, cy_vals, n_basis_vals))
        return np.concatenate(parts, axis=0)
    except (PermissionError, OSError):
        print("[warn] 2-loop target parallel disabled by runtime; fallback to single-process.")
        return _eval_2loop_chunk(x_all, cy_val, n_basis)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def compute_function_target_from_xcoll_2loop(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    output_part="both",
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
    n_basis=22,
) -> torch.Tensor:
    x_np = to_numpy(x_coll)

    if x_np.ndim != 2:
        raise ValueError(f"x_coll must be 2D tensor/array, got shape {x_np.shape}")

    if x_np.shape[1] == 4:
        quint_np = np.concatenate(
            [x_np[:, :4], np.full((x_np.shape[0], 1), float(eps_val))],
            axis=1,
        )
    elif x_np.shape[1] == 5:
        quint_np = x_np[:, :5]
    else:
        raise ValueError(f"x_coll must have 4 or 5 columns, got {x_np.shape}")

    function_complex = compute_boundary_values_rescaled_2loop(
        quint_np,
        cy_val,
        num_workers=num_workers,
        chunk_size=chunk_size,
        parallel_min_points=parallel_min_points,
        n_basis=n_basis,
    )
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)

    return torch.tensor(function_concat, dtype=torch.float32, device=x_coll.device)


def _lin_edge(start, end, n_pts):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, int(n_pts))
    return start[None, :] + (end - start)[None, :] * ts[:, None]


def _sample_boundary_4d(
    *,
    lo,
    hi,
    target_total_bc,
    n_bc_edge,
    n_face_pts,
    n_cell_pts,
    n_corner_extra,
):
    lo = np.asarray(lo, dtype=np.float64).reshape(4)
    hi = np.asarray(hi, dtype=np.float64).reshape(4)

    # 16 vertices.
    vertices = np.array(
        list(itertools.product([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]], [lo[3], hi[3]])),
        dtype=np.float64,
    )

    # 32 edges (4 directions * 2^3 fixed combinations).
    edges_all = []
    for axis in range(4):
        fixed_axes = [a for a in range(4) if a != axis]
        for bits in itertools.product([0, 1], repeat=3):
            start = np.empty(4, dtype=np.float64)
            end = np.empty(4, dtype=np.float64)
            for i, ax in enumerate(fixed_axes):
                val = lo[ax] if bits[i] == 0 else hi[ax]
                start[ax] = val
                end[ax] = val
            start[axis] = lo[axis]
            end[axis] = hi[axis]
            edges_all.append(_lin_edge(start, end, n_bc_edge))
    edges = np.concatenate(edges_all, axis=0)

    # 24 square faces (choose 2 varying dims, 2 fixed dims on lo/hi).
    faces_all = []
    for varying in itertools.combinations(range(4), 2):
        fixed = [a for a in range(4) if a not in varying]
        for bits in itertools.product([0, 1], repeat=2):
            pts = np.empty((int(n_face_pts), 4), dtype=np.float64)
            for ax in varying:
                pts[:, ax] = np.random.uniform(lo[ax], hi[ax], size=int(n_face_pts))
            for i, ax in enumerate(fixed):
                pts[:, ax] = lo[ax] if bits[i] == 0 else hi[ax]
            faces_all.append(pts)
    faces = np.concatenate(faces_all, axis=0)

    # 8 cubic cells on the boundary (fix one dim to lo/hi, vary other 3 dims).
    cells_all = []
    for fixed_axis in range(4):
        varying = [a for a in range(4) if a != fixed_axis]
        for fixed_val in (lo[fixed_axis], hi[fixed_axis]):
            pts = np.empty((int(n_cell_pts), 4), dtype=np.float64)
            for ax in varying:
                pts[:, ax] = np.random.uniform(lo[ax], hi[ax], size=int(n_cell_pts))
            pts[:, fixed_axis] = fixed_val
            cells_all.append(pts)
    cells = np.concatenate(cells_all, axis=0)

    # Extra points around each corner.
    delta = 0.1 * np.maximum(hi - lo, 1e-12)
    corner_extra_list = []
    for v in vertices:
        low_local = np.empty(4, dtype=np.float64)
        high_local = np.empty(4, dtype=np.float64)
        for d in range(4):
            if np.isclose(v[d], lo[d]):
                low_local[d] = v[d]
                high_local[d] = min(v[d] + delta[d], hi[d])
            else:
                low_local[d] = max(v[d] - delta[d], lo[d])
                high_local[d] = v[d]
        pts = np.random.uniform(low_local, high_local, size=(int(n_corner_extra), 4))
        corner_extra_list.append(pts)
    corner_extra = np.concatenate(corner_extra_list, axis=0)

    n_fixed = (
        vertices.shape[0]
        + edges.shape[0]
        + faces.shape[0]
        + cells.shape[0]
        + corner_extra.shape[0]
    )
    n_inner = max(int(target_total_bc) - int(n_fixed), 0)
    inner_points = np.random.uniform(lo, hi, size=(n_inner, 4))

    x_b_all = np.concatenate(
        [vertices, edges, faces, cells, corner_extra, inner_points],
        axis=0,
    )
    return x_b_all


def build_inputs_and_boundary_2loop(
    n_coll_pts,
    x1_lo,
    x1_hi,
    x2_lo,
    x2_hi,
    y1_lo,
    y1_hi,
    y2_lo,
    y2_hi,
    cy_val,
    eps_val,
    device,
    compute_function_target=False,
    output_part="both",
    target_total_bc=500,
    n_bc_edge=6,
    n_face_pts=40,
    n_cell_pts=40,
    n_corner_extra=5,
    bc_abs_cap=1e8,
    n_basis=22,
):
    # ---------- collocation points ----------
    n = int(n_coll_pts)
    x1_all = np.random.uniform(x1_lo, x1_hi, size=n).astype(np.float64)
    x2_all = np.random.uniform(x2_lo, x2_hi, size=n).astype(np.float64)
    y1_all = np.random.uniform(y1_lo, y1_hi, size=n).astype(np.float64)
    y2_all = np.random.uniform(y2_lo, y2_hi, size=n).astype(np.float64)

    x_coll = torch.tensor(
        np.stack((x1_all, x2_all, y1_all, y2_all), axis=1),
        dtype=torch.float32,
        device=device,
    )

    # ---------- boundary points ----------
    x_b_all = _sample_boundary_4d(
        lo=[x1_lo, x2_lo, y1_lo, y2_lo],
        hi=[x1_hi, x2_hi, y1_hi, y2_hi],
        target_total_bc=target_total_bc,
        n_bc_edge=n_bc_edge,
        n_face_pts=n_face_pts,
        n_cell_pts=n_cell_pts,
        n_corner_extra=n_corner_extra,
    )

    function_target = None
    if compute_function_target:
        function_target = compute_function_target_from_xcoll_2loop(
            x_coll,
            cy_val=cy_val,
            eps_val=eps_val,
            output_part=output_part,
            n_basis=n_basis,
        )

    quint_b = np.concatenate(
        [x_b_all, np.full((x_b_all.shape[0], 1), float(eps_val), dtype=float)],
        axis=1,
    )
    bc_complex = compute_boundary_values_rescaled_2loop(
        quint_b,
        cy_val,
        n_basis=n_basis,
    )

    # Filter out boundary points with non-finite or extremely large analytic values.
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
            "[warn] build_inputs_and_boundary_2loop: "
            f"filtered {n_bad_total} / {bc_complex.shape[0]} boundary points "
            f"(non-finite={n_bad_nonfinite}, |value|>{bc_abs_cap:g}={n_bad_large})."
        )
        x_b_all = x_b_all[keep_mask]
        bc_complex = bc_complex[keep_mask]

    bc_concat = _complex_to_output_channels(bc_complex, output_part=output_part)
    bc_target = torch.tensor(bc_concat, dtype=torch.float32, device=device)
    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    return x_coll, x_b_tensor, bc_target, function_target
