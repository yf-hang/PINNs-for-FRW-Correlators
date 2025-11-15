import torch
import torch.nn as nn

# ----------  Base Class: shared implementation ---------
class ConnectionAMatricesBase(nn.Module):
    """
    Base A-matrix
    A_i(x) = sum_k A_k * pd_{x_i} log(w_k)
    Letters: w_k = {x1+cy, x1-cy, x2+cy, x2-cy, x1+x2}
    """
    def __init__(self, n_basis=4, n_letters=5, cy_val=None):
        super().__init__()
        self.n_basis = n_basis
        self.n_letters = n_letters
        self.cy_val = cy_val

    # ---------- Static: compute pd log(w)/ pd x_i ----------
    @staticmethod
    def dlog_partials(x1: torch.Tensor, x2: torch.Tensor, cy_val: float):
        """
        x1, x2 shape (n_coll, 1)
        Compute pd log(w_k) / pd x1 and pd log(w_k) / pd x2
        Returns: dlog_dx1, dlog_dx2 : shape (n_coll, n_letters, 1)
        """
        zeros_x1 = torch.zeros_like(x1)
        zeros_x2 = torch.zeros_like(x2)

        w1, w3 = x1 + cy_val, x1 - cy_val
        w2, w4 = x2 + cy_val, x2 - cy_val
        w5 = x1 + x2

        dlog_dx1 = torch.stack([1/w1, zeros_x1, 1/w3, zeros_x1, 1/w5], dim=1)
        dlog_dx2 = torch.stack([zeros_x2, 1/w2, zeros_x2, 1/w4, 1/w5], dim=1)
        return dlog_dx1, dlog_dx2

    # ---------- Core Shared Forward ----------
    def build_matrices(self, x_in, ak_list, cy_val=None):

        cy_used = self.cy_val if cy_val is None else cy_val
        if cy_used is None:
            raise ValueError("cy_val must be set either in constructor or in forward().")

        x1, x2 = x_in[:, 0:1], x_in[:, 1:2]
        dlog_dx1, dlog_dx2 = self.dlog_partials(x1, x2, cy_used)

        device, dtype = x_in.device, x_in.dtype
        a_x1 = torch.zeros(x_in.shape[0], self.n_basis, self.n_basis, device=device, dtype=dtype)
        a_x2 = torch.zeros_like(a_x1)

        for k, ak in enumerate(ak_list):
            coef_a1 = dlog_dx1[:, k:k+1]
            coef_a2 = dlog_dx2[:, k:k+1]
            a_x1 += coef_a1 * ak.unsqueeze(0)
            a_x2 += coef_a2 * ak.unsqueeze(0)
            #print(dlog_dx1[:, k:k+1].shape)
            #print(ak.shape)
        return a_x1, a_x2


# ---------- Learnable version — for PINN / CDE training ----------
class ConnectionAMatrices(ConnectionAMatricesBase):
    """
    Learnable A-matrix builder with trainable coefficients a_k.
    A_i(x) = sum_k a_k * pd_{x_i} log(w_k)
    """
    def __init__(self, n_basis_local=4, n_letters=5, cy_val=None):
        super().__init__(n_basis_local, n_letters, cy_val)
        self.ak_list = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(n_basis_local, n_basis_local))
            # 0.01: Initialize the matrices to be nearly smooth (close to zero),
            # but retain slight perturbations to trigger learning.
            for _ in range(n_letters)
        ])

    def forward(self, x_in, cy_val=None):
        return self.build_matrices(x_in, self.ak_list, cy_val)


# ----------  Fixed version — A-matrix from known constant matrices ----------
class ConnectionAMatricesFixed(ConnectionAMatricesBase):
    # Fixed A-matrix builder using constant matrices a1 - a5 see my note

    def __init__(self, n_basis_local=4, n_letters=5, ak_list=None, cy_val=None):
        super().__init__(n_basis_local, n_letters, cy_val)

        if not ak_list or len(ak_list) != n_letters:
            raise ValueError(f"ak_list must contain exactly {n_letters} matrices.")

        for k, ak in enumerate(ak_list):
            self.register_buffer(f"a_{k}", ak.detach().clone())

    def forward(self, x_in, cy_val=None):
        ak_list = [getattr(self, f"a_{k}") for k in range(self.n_letters)]
        return self.build_matrices(x_in, ak_list, cy_val)
