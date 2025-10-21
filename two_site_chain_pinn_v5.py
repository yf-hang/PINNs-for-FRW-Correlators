# two_site_chain_pinn_v5.py
# Rescaled variables only (x1, x2). Euclidean region.
# Safe collocation + automatic boundary + Canonical DE residual.
# Two-phase training baseline (Phase 1) + two Phase 2 variants (A: random A, B: copied A).
# Outputs two PDFs: phase2A and phase2B losses.

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import json
from lib.conn_matrices import ConnectionAMatrices, ConnectionAMatricesFixed
from lib.analytic_solutions import (
    cd_of_eps, cf_of_eps, cz_of_eps,
    p_sol, f_sol, ft_sol, q_sol,
    build_inputs_and_boundary)
from tools.plot_losses import plot_and_save
# ---------------- user settings ----------------

class Config:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.__dict__.update(data)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)

# Initialize config
cfg = Config("config.json")

#print("Loaded configuration:")
#print(cfg)
print(f"Using device: {cfg.device}")

device = torch.device(cfg.device)

# ---------------- PINN Model ----------------
# ---------------- Feed-Forward NN ----------------
class PinnModel(nn.Module):
    def __init__(self, hidden_size_=128, n_hidden_layers_=4, use_eps_input=True):
        super().__init__()
        in_dim = 3 if use_eps_input else 2
        layers = [nn.Linear(in_dim, hidden_size_), nn.Tanh()]
        for _ in range(n_hidden_layers_ - 1):
            layers += [nn.Linear(hidden_size_, hidden_size_), nn.Tanh()]
        layers += [nn.Linear(hidden_size_, 2 * cfg.n_basis)]  # real + imag stacked
        self.net = nn.Sequential(*layers)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.net(x_in)

# ---------------- gradients for complex outputs ----------------
def compute_gradients_complex(model: nn.Module, x_input: torch.Tensor):
    """
    Returns:
      re_out, im_out: (N, n_basis)
      d_re_dx1, d_re_dx2, d_im_dx1, d_im_dx2: (N, n_basis)
    """
    x_req = x_input.clone().detach().requires_grad_(True)
    outputs = model(x_req)           # (N, 2*n_basis)
    re_out = outputs[:, :cfg.n_basis]
    im_out = outputs[:, cfg.n_basis:]

    grads_re_list, grads_im_list = [], []
    for idx_f in range(cfg.n_basis):
        g_re = torch.autograd.grad(re_out[:, idx_f].sum(), x_req, create_graph=True)[0]  # (N,in_dim)
        g_im = torch.autograd.grad(im_out[:, idx_f].sum(), x_req, create_graph=True)[0]
        grads_re_list.append(g_re[:, 0:2].unsqueeze(2))  # keep x1,x2 dims -> (N,2,1)
        grads_im_list.append(g_im[:, 0:2].unsqueeze(2))

    grads_re = torch.cat(grads_re_list, dim=2)   # (N,2,n_basis)
    grads_im = torch.cat(grads_im_list, dim=2)
    d_re_dx1 = grads_re[:, 0, :]
    d_re_dx2 = grads_re[:, 1, :]
    d_im_dx1 = grads_im[:, 0, :]
    d_im_dx2 = grads_im[:, 1, :]
    return re_out, im_out, d_re_dx1, d_re_dx2, d_im_dx1, d_im_dx2

# ---------------- losses ----------------
def pde_residual_loss(model: nn.Module, a_builder: nn.Module,
                      x_batch: torch.Tensor, eps_val: float):
    """
    Canonical DE residual:
      dI/dx1 = eps * A1(x) I
      dI/dx2 = eps * A2(x) I
    I is complex -> train real+imag stacked outputs.
    """
    re_out, im_out, d_re_dx1, d_re_dx2, d_im_dx1, d_im_dx2 = compute_gradients_complex(model, x_batch)

    # A matrices at x_batch
    a1, a2 = a_builder(x_batch[:, :2])  # use [x1,x2]

    # batched real "complex" matmul with real A
    def batched_matmul_complex(a_mat, v_re, v_im):
        out_re = torch.einsum('bij,bj->bi', a_mat, v_re)
        out_im = torch.einsum('bij,bj->bi', a_mat, v_im)
        return out_re, out_im

    rhs1_re, rhs1_im = batched_matmul_complex(a1, re_out, im_out)
    rhs2_re, rhs2_im = batched_matmul_complex(a2, re_out, im_out)

    # residuals: dF/dxi - eps*A_i*F
    res1_re = d_re_dx1 - eps_val * rhs1_re
    res1_im = d_im_dx1 - eps_val * rhs1_im
    res2_re = d_re_dx2 - eps_val * rhs2_re
    res2_im = d_im_dx2 - eps_val * rhs2_im

    # MSE over all components
    loss = (res1_re.pow(2).mean() + res1_im.pow(2).mean() +
            res2_re.pow(2).mean() + res2_im.pow(2).mean())
    return loss

def boundary_loss(model: nn.Module, x_b_tensor: torch.Tensor, bc_target: torch.Tensor):
    pred_b = model(x_b_tensor)  # (1, 8)
    return nn.MSELoss()(pred_b, bc_target)

# ---------------- LR scheduler: warmup + cosine ----------------
def make_warmup_cosine_scheduler(optimizer, total_epochs: int, warmup_epochs_: int):
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs_:
            return float(epoch_idx + 1) / float(warmup_epochs_)
        # cosine from warmup to total
        progress = (epoch_idx - warmup_epochs_) / max(1, (total_epochs - warmup_epochs_))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------- training phases ----------------
def train_phase(model, a_builder, x_coll, x_b_tensor, bc_target,
                lr_init, total_epochs, warmup_len,
                freeze_a=False,
                pde_w=1.0, bc_w=200.0,
                phase_name="phase"):
    # freeze/unfreeze A
    for p in a_builder.parameters():
        p.requires_grad = (not freeze_a)

    # optimizer (+ scheduler)
    params = list(model.parameters()) + (list(a_builder.parameters()) if not freeze_a else [])
    optimizer = optim.Adam(params, lr=lr_init)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_epochs, warmup_len)

    loss_total_hist, loss_pde_hist, loss_bc_hist = [], [], []
    for step in range(1, total_epochs + 1):
        optimizer.zero_grad()

        loss_pde = pde_residual_loss(model, a_builder, x_coll, cfg.vareps_global)
        loss_bc = boundary_loss(model, x_b_tensor, bc_target)
        loss_total = pde_w * loss_pde + bc_w * loss_bc

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        loss_total_hist.append(loss_total.item())
        loss_pde_hist.append(loss_pde.item())
        loss_bc_hist.append(loss_bc.item())

        if step % cfg.print_every == 0 or step == 1 or step == total_epochs:
            print(f"[{phase_name} {step:04d}] total={loss_total.item():.3e}  pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}  lr={scheduler.get_last_lr()[0]:.2e}")

    return loss_total_hist, loss_pde_hist, loss_bc_hist

# ---------------- main ----------------
def main():
    # numpy seed
    np.random.seed(0)
    # PyTorch CPU seed
    torch.manual_seed(0)
    # PyTorch GPU seed (only for GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # data
    x_coll, x_b_tensor, bc_target = build_inputs_and_boundary(cfg.n_coll, cfg.x1_min, cfg.x1_max,
                                                              cfg.x2_min, cfg.x2_max,
                                                              cfg.cy, cfg.y1,
                                                              cfg.vareps_global,
                                                              cfg.use_eps_as_input,
                                                              device)

    # model base (for Phase 1)
    model_base = PinnModel(cfg.hidden_size,
                           cfg.n_hidden_layers,
                           cfg.use_eps_as_input).to(device)

    # ---- Define fixed a_k matrices ----
    # dtype = torch.float32
    a1 = torch.tensor([[1.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, -1.0],
                       [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    a2 = torch.tensor([[1.0, 0.0, -1.0, 0.0],
                       [0.0, 1.0, 0.0, -1.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    a3 = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    a4 = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    a5 = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 2.0]], dtype=torch.float32, device=device)

    ak_list = [a1, a2, a3, a4, a5]

    # ---- Phase 1: fixed A, train F only ----
    print("\n---- Phase 1: Fixed connection matrix A (train Functions only) ----")
    a_builder_fixed = ConnectionAMatricesFixed(ak_list=ak_list, cy_val=cfg.cy).to(device)
    tot1, pde1, bc1 = train_phase(
        model_base, a_builder_fixed, x_coll, x_b_tensor, bc_target,
        lr_init=cfg.learning_rate_p1, total_epochs=cfg.phase1_epochs, warmup_len=cfg.warmup_epochs,
        freeze_a=True, pde_w=cfg.pde_weight, bc_w=cfg.bc_weight, phase_name="Phase 1"
    )
    # Snapshot Phase 1 model weights
    phase1_state = {k: v.detach().clone() for k, v in model_base.state_dict().items()}

    # -------------- Phase 2A: Random-initialized A --------------
    print("\n---- Phase 2A: Unfreeze A (random initialization, exploration mode) ----")
    model_a = PinnModel(cfg.hidden_size, cfg.n_hidden_layers, cfg.use_eps_as_input).to(device)
    model_a.load_state_dict(phase1_state)  # start from Phase 1 solution
    a_builder_a = ConnectionAMatrices(n_basis_local=cfg.n_basis, n_letters=5, cy_val=cfg.cy).to(device)

    tot2a, pde2a, bc2a = train_phase(
        model_a, a_builder_a, x_coll, x_b_tensor, bc_target,
        lr_init=cfg.learning_rate_p2, total_epochs=cfg.phase2_epochs, warmup_len=cfg.warmup_epochs,
        freeze_a=False, pde_w=cfg.pde_weight, bc_w=cfg.bc_weight, phase_name="Phase 2A"
    )

    # -------------- Phase 2B: Copy-from-fixed A --------------
    print("\n---- Phase 2B: Unfreeze A (copied from fixed A matrices, fine-tune mode) ----")
    model_b = PinnModel(cfg.hidden_size, cfg.n_hidden_layers, cfg.use_eps_as_input).to(device)
    model_b.load_state_dict(phase1_state)  # same Phase 1 start
    a_builder_b = ConnectionAMatrices(n_basis_local=cfg.n_basis, n_letters=5, cy_val=cfg.cy).to(device)
    with torch.no_grad():
        for i in range(5):
            a_builder_b.ak_list[i].copy_(ak_list[i])

    tot2b, pde2b, bc2b = train_phase(
        model_b, a_builder_b, x_coll, x_b_tensor, bc_target,
        lr_init=cfg.learning_rate_p2, total_epochs=cfg.phase2_epochs, warmup_len=cfg.warmup_epochs,
        freeze_a=False, pde_w=cfg.pde_weight, bc_w=cfg.bc_weight, phase_name="Phase 2B"
    )

    # --------------------- Plotting ---------------------
    plot_and_save(
        tot1, pde1, bc1,  # Phase 1
        tot2a, pde2a, bc2a,  # Phase 2A
        title="2-site chain: P1 → P2A random",
        fname="p1_to_p2A.pdf",
        phase1_epochs=cfg.phase1_epochs,
        phase2_epochs=cfg.phase2_epochs
    )

    plot_and_save(
        tot1, pde1, bc1,  # Phase 1
        tot2b, pde2b, bc2b,  # Phase 2B
        title="2-site chain: P1 → P2B copied",
        fname="p1_to_p2B.pdf",
        phase1_epochs=cfg.phase1_epochs,
        phase2_epochs=cfg.phase2_epochs
    )
    print("\nBoth Phase 2 schemes completed.")

if __name__ == "__main__":
    main()