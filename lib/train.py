import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as nn_utils


def _grad_norm_l2(model):
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return sq ** 0.5


def train_model_fixed_eps(
    model,
    a_builder,
    x_coll,
    x_b_tensor,
    bc_target,
    *,
    cde_loss_fixed_fn,
    bc_loss_fn,
    n_basis,
    eps_val,
    lr_init,
    warmup_len,
    total_epochs,
    lam1,
    lam2,
    cosine_min_lr=0.0,
    print_every=100,
    phase_name="P1",
    log_fn=None,
    use_grad_norm_probe=False,
    grad_clip_max_norm=10.0,
):
    def _emit(msg: str):
        print(msg)
        if log_fn is not None:
            log_fn(msg)

    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_len, 1),
        eta_min=cosine_min_lr,
    )

    loss_tot_hist = []
    loss_cde_hist = []
    loss_bc_hist = []

    for step in range(1, total_epochs + 1):
        optimizer.zero_grad()

        loss_cde, Nc = cde_loss_fixed_fn(
            model, a_builder, x_coll, n_basis, eps_val=eps_val
        )
        loss_bc = bc_loss_fn(model, x_b_tensor, bc_target)

        loss_total = lam1 * loss_cde + lam2 * loss_bc
        loss_total.backward()

        if use_grad_norm_probe:
            # grad-norm probe + optional clipping
            gn = float(nn_utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip_max_norm),
            ))

            # compute post-clip norm (L2)
            gn_post = _grad_norm_l2(model)
            gn_post_text = f" | gn_post={float(gn_post):.2e}"
        else:
            # raw L2 grad norm without clipping
            gn = _grad_norm_l2(model)
            gn_post_text = ""

        optimizer.step()

        if step <= warmup_len:
            scale = step / float(warmup_len)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_init * scale
        else:
            cosine_scheduler.step()

        loss_tot_hist.append(loss_total.item())
        loss_cde_hist.append(loss_cde.item())
        loss_bc_hist.append(loss_bc.item())

        if step % print_every == 0 or step == 1 or step == total_epochs:
            lr = optimizer.param_groups[0]["lr"]
            tot_val = float(loss_total.item())
            bc_val = float(loss_bc.item())
            bc_weighted_val = float(lam2) * bc_val
            if abs(tot_val) > 1e-30:
                bc_over_tot_pct = 100.0 * bc_weighted_val / tot_val
            else:
                bc_over_tot_pct = float("nan")
            msg = (
                f"[{step:04d}] "
                f"tot={tot_val:.3e} | "
                f"cde={loss_cde.item():.3e} | "
                f"bc={bc_val:.3e} | "
                f"lam2*bc/tot={bc_over_tot_pct:.2f}% | "
                f"lr={lr:.2e} | "
                f"gn={float(gn):.2e}"
                f"{gn_post_text}"
            )
            if step == 1:
                nb = int(x_b_tensor.shape[0])
                msg = (
                    f"Nc={int(Nc.item())}, Nb={nb}, "
                    f"lambda1={float(lam1):g}, lambda2={float(lam2):g}\n"
                    + msg
                )
            _emit(msg)

    _emit(f" ------ {phase_name} fixed-eps training complete ------\n")

    return (
        model,
        loss_tot_hist,
        loss_cde_hist,
        loss_bc_hist,
    )
