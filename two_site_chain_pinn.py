# two_site_chain_pinn version 11
# Rescaled variables (x1, x2).
# One-phase training baseline
# Boundary selection: 4 edges + points on the diagonal lines
# Adaptive weights for CDE and BC + EMA

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import json
from lib.conn_matrices import ConnectionAMatrices, ConnectionAMatricesFixed
from lib.analytic_solutions import (
    cd_of_eps, cf_of_eps, cz_of_eps,
    p_sol, f_sol, ft_sol, q_sol,
    build_inputs_and_boundary, to_numpy)
from plot_tools.plot_losses import plot_and_save_p1
from plot_tools.plot_error import plot_error_dis, plot_x1_x2, plot_cosine_similarity

# ---------------- Settings ----------------
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
#cfg = Config("config_local.json")

print("[Configuration]")
print(cfg)
print(f"\n[Using device]: {cfg.device}")

device = torch.device(cfg.device)

# ---------------- PINN Model ----------------
# ---------------- Feed-Forward NN ----------------
class PinnModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size_ = config.hidden_size
        n_hidden_layers_ = config.n_hidden_layers
        use_eps_input = config.use_eps_as_input
        n_basis_ = config.n_basis
        activation_name = config.activation_f

        activation = self.get_activation(activation_name)

        in_dim = 3 if use_eps_input else 2
        layers = [nn.Linear(in_dim, hidden_size_), activation]
        for _ in range(n_hidden_layers_ - 1):
            layers += [nn.Linear(hidden_size_, hidden_size_), activation]
        layers += [nn.Linear(hidden_size_, 2 * n_basis_)]  # real + imag
        self.net = nn.Sequential(*layers)
        print(f"[Activation]: {activation_name}")

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.net(x_in)

    @staticmethod
    def get_activation(name: str):
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

# ----------------------------------------------------------------
# ---------------- Gradients for complex outputs ----------------
# ----------------------------------------------------------------
def compute_gradients_complex(model: nn.Module, x_input: torch.Tensor):

    #Returns: re_out, im_out: (N, n_basis)
            # d_re_dx1, d_re_dx2,
            # d_im_dx1, d_im_dx2: (N, n_basis)

    x_req = x_input.clone().detach().requires_grad_(True)
    outputs = model(x_req) # (N, 2 * n_basis)
    re_out = outputs[:, :cfg.n_basis]
    im_out = outputs[:, cfg.n_basis:]

    grads_re_list, grads_im_list = [], []

    for idx_f in range(cfg.n_basis):
        g_re = torch.autograd.grad(re_out[:, idx_f].sum(), x_req, create_graph=True)[0]  # (N, in_dim)
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

# --------------------------------------------------
# ---------------- Losses (CDE + BC)----------------
# --------------------------------------------------
def cde_residual_loss(model: nn.Module, a_builder: nn.Module,
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

    def batched_matmul_complex(a_mat, v_re, v_im):
        out_re = torch.einsum('bij,bj->bi', a_mat, v_re)
        out_im = torch.einsum('bij,bj->bi', a_mat, v_im)
        return out_re, out_im

    rhs1_re, rhs1_im = batched_matmul_complex(a1, re_out, im_out)
    rhs2_re, rhs2_im = batched_matmul_complex(a2, re_out, im_out)

    res1_re = d_re_dx1 - eps_val * rhs1_re #(N,4)
    res1_im = d_im_dx1 - eps_val * rhs1_im #(N,4)
    res2_re = d_re_dx2 - eps_val * rhs2_re #(N,4)
    res2_im = d_im_dx2 - eps_val * rhs2_im #(N,4)

    # MSE over all components
    loss_cde = 0.5 * (res1_re.pow(2).mean() + res1_im.pow(2).mean() +
            res2_re.pow(2).mean() + res2_im.pow(2).mean())
            # 0.5 -> average two components x1, x2
    return loss_cde

def boundary_loss(model: nn.Module, x_b_tensor: torch.Tensor, bc_target: torch.Tensor):
    pred_b = model(x_b_tensor)  # (1, 8), Re + Im = 8
    loss_bc = nn.MSELoss()(pred_b, bc_target) * 2.0 # * 2.0 -> average 4 functions (Re+Im)
    return loss_bc

def integrability_loss(a_builder, x_batch):
    #Compute ||[A1, A2]|| for integrability penalty.
    A1, A2 = a_builder(x_batch)
    comm = torch.bmm(A1, A2) - torch.bmm(A2, A1)
    return torch.linalg.norm(comm, dim=(1,2)).mean()

# -------------------------------------------------
# ---------------- Training phases ----------------
# -------------------------------------------------
def train_phase(model, a_builder, x_coll, x_b_tensor,
                bc_target, function_target,
                lr_init, total_epochs, warmup_len,
                freeze_a=False,
                cde_w=1.0, bc_w=1.0,
                enforce_integrability=False,
                lambda_int=0.0,
                phase_name="phase"):

    ema_ratio = None
    # freeze/unfreeze A
    for p in a_builder.parameters():
        p.requires_grad = (not freeze_a)

    # ---- Optimizer ----
    params = list(model.parameters()) + (list(a_builder.parameters()) if not freeze_a else [])
    optimizer = optim.Adam(params, lr=lr_init)

    # ---- Scheduler: Warmup + Cosine ----
    def warmup_fn(epoch):
        if epoch < warmup_len:
            return float(epoch + 1) / float(warmup_len)
        else:
            return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)

    # ---- Main scheduler selection ----
    scheduler_type = cfg.scheduler_type.lower() if cfg is not None else "cosine"

    if scheduler_type == "cos":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max = total_epochs - warmup_len,
            eta_min =  cfg.cosine_min_lr
        )
        use_plateau = False
        print(f"[Scheduler] Warmup + CosineAnnealingLR")

    elif scheduler_type == "cos_warm":
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = cfg.cosine_t0,
            eta_min =  cfg.cosine_min_lr
        )
        use_plateau = False
        print(f"[Scheduler] Warmup + CosineAnnealingWarmRestarts")

    elif scheduler_type == "pla":
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor = cfg.plateau_factor,
            patience = cfg.plateau_patience,
            threshold = cfg.plateau_threshold,
            cooldown = cfg.plateau_cooldown,
            min_lr = cfg.plateau_min_lr
        )
        use_plateau = True
        print(f"[Scheduler] Warmup + ReduceLROnPlateau")

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    loss_total_hist, loss_cde_hist, loss_bc_hist = [], [], []

    for step in range(1, total_epochs + 1):
        optimizer.zero_grad()

        loss_cde = cde_residual_loss(model, a_builder, x_coll, cfg.eps_global)
        loss_bc = boundary_loss(model, x_b_tensor, bc_target)
        loss_total = cde_w * loss_cde + bc_w * loss_bc

        # ------------------------------------
        # ----- integrability constraint -----
        # ------------------------------------
        loss_int = torch.tensor(0.0, device=x_coll.device)  # safe default
        if enforce_integrability:
            loss_int = integrability_loss(a_builder, x_coll)
            loss_total += lambda_int * loss_int
        # -------------------------------------

        loss_total.backward()
        optimizer.step()

        # Scheduler step
        if step <= warmup_len:
            warmup_scheduler.step()
        else:
            if use_plateau:
                val_loss = loss_total.item()
                main_scheduler.step(val_loss)
            else:
                main_scheduler.step()

        loss_total_hist.append(loss_total.item())
        loss_cde_hist.append(loss_cde.item())
        loss_bc_hist.append(loss_bc.item())

#---------------------------------------------------------------
        if step % cfg.print_every == 0 or step == 1 or step == total_epochs:
            lr = optimizer.param_groups[0]['lr']
            ratio_raw = loss_cde.item() / (loss_bc.item() + 1e-12)
            #print(f"[{phase_name} {step:04d}] | tot= {loss_total.item():.3e} | "
            #      f"cde= {loss_cde.item():.3e} | bc= {loss_bc.item():.3e} | "
            #      f"cde/bc= {ratio_raw:.2e} | lr= {lr:.2e}"
            #      )

            # Base message
            msg = (f"[{phase_name} {step:04d}] | "
                   f"tot= {loss_total.item():.3e} | "
                   f"cde= {loss_cde.item():.3e} | "
                   f"bc= {loss_bc.item():.3e} | "
                   f"cde/bc= {ratio_raw:.2e}")

            # Append integrability info if active
            if enforce_integrability:
                msg += f" | int= {loss_int.item():.3e}"

            msg += f" | lr= {lr:.2e}"
            print(msg)

            # -------- Auto-Balance --------
            target_ratio = 5.0
            alpha = 0.3
            min_cde_w, max_cde_w = 1e-6, 1.0
            ema_beta = 0.8

            # initial EMA
            if ema_ratio is None:
                ema_ratio = ratio_raw
            else:
                ema_ratio = ema_beta * ema_ratio + (1 - ema_beta) * ratio_raw
            ema_ratio = max(ema_ratio, 1e-12)

            # update EMA
            ema_ratio = ema_beta * ema_ratio + (1 - ema_beta) * ratio_raw
            ema_ratio = max(ema_ratio, 1e-12)

            # Adjust after warmup
            if step > warmup_len:
                adjust = (target_ratio / ema_ratio) ** alpha
                cde_w *= adjust
                cde_w = max(min(cde_w, max_cde_w), min_cde_w)

                def colorize(val, txt):
                    if val > 10 * target_ratio:
                        return f"\033[91m{txt}\033[0m"  # red too large
                    elif val < 0.1 * target_ratio:
                        return f"\033[94m{txt}\033[0m"  # blue too small
                    else:
                        return f"\033[92m{txt}\033[0m"  # green normal

                ratio_txt = colorize(ema_ratio, f"{ema_ratio:.2e}")
                cde_txt = colorize(cde_w, f"{cde_w:.2e}")

                print(f"[AutoBal] | " #step= {step:04d} "
                      f"cde/bc(raw)= {ratio_raw:.2e} | "
                      f"cde/bc(EMA)= {ratio_txt} "
                      f"→ cde_w= {cde_txt} (adjust= {adjust:.2f})")
    # ---------- train end ----------
    print(f"\n[{phase_name}] training complete.")
    print(f"Final cde_weight = {cde_w:.3e}")

    plot_error_dis(model, x_coll, function_target,
                   phase_name=phase_name)

    return loss_total_hist, loss_cde_hist, loss_bc_hist

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
    x_coll, x_b_tensor, bc_target, function_target = build_inputs_and_boundary(
        cfg.n_coll, cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max,
        cfg.cy, cfg.eps_global, cfg.use_eps_as_input,
        device)

    # model base (for Phase 1)
    model_base = PinnModel(cfg).to(device)

    # ---- Define fixed a_k matrices ----
    # dtype = torch.float32
    a1 = torch.tensor([[1.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, -1.0],
                       [0.0, 0.0, 0.0, 0.0]],
                      dtype=torch.float32, device=device)

    a2 = torch.tensor([[1.0, 0.0, -1.0, 0.0],
                       [0.0, 1.0, 0.0, -1.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]],
                      dtype=torch.float32, device=device)

    a3 = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]],
                      dtype=torch.float32, device=device)

    a4 = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]],
                      dtype=torch.float32, device=device)

    a5 = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 2.0]],
                      dtype=torch.float32, device=device)

    ak_list = [a1, a2, a3, a4, a5]

    #act_name = cfg.activation_f
    #bc_wt = cfg.bc_weight
    #cde_wt = cfg.cde_weight
    eps_here = cfg.eps_global
    cy_here = cfg.cy
    #scheduler_name = cfg.scheduler_type

    # ---- Phase 1: fixed A, train functions only ----
    print("\n---- Phase 1: Fixed connection matrix A (train functions only) ----")
    a_builder_fixed = ConnectionAMatricesFixed(ak_list=ak_list, cy_val=cfg.cy).to(device)
    tot1, cde1, bc1 = train_phase(
        model_base, a_builder_fixed,
        x_coll, x_b_tensor,
        bc_target, function_target,
        lr_init=cfg.learning_rate_p1,
        total_epochs=cfg.phase1_epochs,
        warmup_len=cfg.warmup_epochs_p1,
        freeze_a=True,
        cde_w=cfg.cde_weight, bc_w=cfg.bc_weight,
        phase_name="P1"
    )
    # Snapshot Phase 1 model weights
    #phase1_state = {k: v.detach().clone() for k, v in model_base.state_dict().items()}

    plot_and_save_p1(
        tot1, cde1, bc1,
       # title=rf"2-site chain P1 ($\varepsilon$:{eps_here}; bc:{bc_wt}; cde:{cde_wt}; {act_name}; {scheduler_name})",
        title=rf"P1 ($\varepsilon$ = {eps_here}, $c$ = {cy_here})",
        #fname=f"P1_{act_name}_{scheduler_name}.png",
        fname=f"P1_losses.png",
        fname2=f"P1_loss_tot.png",
        phase1_epochs=cfg.phase1_epochs
    )
    plot_x1_x2(
        to_numpy(x_coll),
        to_numpy(function_target),
        to_numpy(model_base(x_coll).detach()),
        phase_name="P1"
    )
    plot_cosine_similarity(
        model_base,
        x_coll,
        function_target,
        phase_name="P1"
    )
    print("\ncomplete")

if __name__ == "__main__":
    main()