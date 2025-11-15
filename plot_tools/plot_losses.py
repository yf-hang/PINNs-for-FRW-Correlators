#import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json

def get_nested_save_dir(base_save_dir, config_path="config.json"):

    if not os.path.exists(config_path):
        print(f"[warn] config.json not found: {config_path}")
        return os.path.join("eps_unknown", "cy_unknown", base_save_dir)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # eps
    eps_val = cfg.get("eps_global", None)
    if eps_val is None:
        eps_folder = "eps_unknown"
    else:
        eps_prefix = "p" if eps_val >= 0 else "m"
        eps_folder = f"eps_{eps_prefix}{str(abs(eps_val)).replace('.', '_')}"

    # cy
    cy_val = cfg.get("cy", None) or cfg.get("cy_val", None)
    if cy_val is None:
        cy_folder = "cy_unknown"
    else:
        cy_prefix = "p" if cy_val >= 0 else "m"
        cy_folder = f"cy_{cy_prefix}{str(abs(cy_val)).replace('.', '_')}"

    final_dir = os.path.join(eps_folder, cy_folder, base_save_dir)
    os.makedirs(final_dir, exist_ok=True)
    return final_dir

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "font.size": 12,
    "legend.frameon": False,
    "legend.handlelength": 2.8,
    "legend.fontsize": 11,
})

plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"



def plot_and_save_p1(
        phase1_vals, phase1_cde, phase1_bc,
        title, save_dir="1_losses",
        fname="phase",
        fname2="phase2",
        phase1_epochs=0):

    save_dir = get_nested_save_dir(save_dir, "config.json")

    plt.figure(figsize=(6, 4))

    # --- Combine x-axis ---
    x1 = np.arange(1, phase1_epochs + 1)

    colors = {
        "total": "tab:blue",
        "cde": "tab:orange",
        "bc": "tab:green"
    }

    plt.semilogy(x1, phase1_vals, "-", color=colors["total"], label="Total")
    plt.semilogy(x1, phase1_cde, "-", color=colors["cde"], alpha=0.7, label="CDE")
    plt.semilogy(x1, phase1_bc, "-", color=colors["bc"], alpha=0.7, label="BC")

    plt.legend(
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()


    plt.semilogy(x1, phase1_vals, "-", color=colors["total"], label="Total")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname2), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[saved] {fname} to /{save_dir}")
    print(f"[saved] {fname2} to /{save_dir}")