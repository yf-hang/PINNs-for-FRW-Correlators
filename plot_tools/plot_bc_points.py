import os
import json
import matplotlib.pyplot as plt

def get_eps_dir(config_path="config.json"):
    if not os.path.exists(config_path):
        print(f"[warn] config.json not found: {config_path}")
        eps_dir = "eps_unknown"
    else:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        eps_val = cfg.get("eps_global", None)
        if eps_val is None:
            eps_dir = "eps_unknown"
        else:
            prefix = "p" if eps_val >= 0 else "m"
            eps_str = str(abs(eps_val)).replace(".", "_")
            eps_dir = f"eps_{prefix}{eps_str}"
    os.makedirs(eps_dir, exist_ok=True)
    return eps_dir


def visualize_bc_points(
    x1_lo, x1_hi, x2_lo, x2_hi, x_b_all,
    save_path=None, save_name=None):

    eps_dir = get_eps_dir("config.json")

    n_bc = len(x_b_all)
    auto_name = f"nb_{n_bc}.png"

    if save_path is not None:
        save_file = os.path.basename(save_path)
    elif save_name is not None:
        save_file = save_name
    else:
        save_file = auto_name

    save_path_full = os.path.join(eps_dir, save_file)

    plt.figure(figsize=(6, 4))
    plt.scatter(x_b_all[:, 0], x_b_all[:, 1], c="tab:red", s=5, label="BC Points")
    plt.xlim(x1_lo - 0.5, x1_hi + 0.5)
    plt.ylim(x2_lo - 0.5, x2_hi + 0.5)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(f"$N_b$ = {len(x_b_all)}")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", "box")

    plt.tight_layout()
    plt.savefig(save_path_full, dpi=200)
    plt.close()
    print(f"[saved] {save_path_full}")


def visualize_collocation_points(
    x1_all, x2_all, focus_x1_max, focus_x2_max,
    save_path=None, save_name=None):
    eps_dir = get_eps_dir("config.json")

    n_coll = x1_all.shape[0]
    auto_name = f"ncoll_{n_coll}.png"

    if save_path is not None:
        save_file = os.path.basename(save_path)
    elif save_name is not None:
        save_file = save_name
    else:
        save_file = auto_name

    save_path_full = os.path.join(eps_dir, save_file)

    plt.figure(figsize=(6, 4))
    mask_focus = (x1_all < focus_x1_max) & (x2_all < focus_x2_max)

    plt.scatter(x1_all[~mask_focus], x2_all[~mask_focus],
                s=6, c='tab:blue', alpha=0.5, label="Global region")
    plt.scatter(x1_all[mask_focus], x2_all[mask_focus],
                s=8, c='tab:orange', alpha=0.6, label="Focus region")

    plt.axvline(focus_x1_max, color='gray', linestyle='--', lw=1)
    plt.axhline(focus_x2_max, color='gray', linestyle='--', lw=1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(f"$N_c$ = {len(x1_all)}")
    plt.tight_layout()
    plt.savefig(save_path_full, dpi=200)
    plt.close()
    print(f"[saved] {save_path_full}")
