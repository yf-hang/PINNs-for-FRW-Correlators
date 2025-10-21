import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

def plot_and_save(
        phase1_vals, phase1_pde, phase1_bc,
        phase2_vals, phase2_pde, phase2_bc,
        title, fname,
        phase1_epochs, phase2_epochs):

    plt.figure(figsize=(6, 4))

    # --- Combine x-axis ---
    x1 = np.arange(1, phase1_epochs + 1)
    x2 = np.arange(phase1_epochs + 1, phase1_epochs + phase2_epochs + 1)

    # --- Color ---
    colors = {
        "total": "tab:blue",
        "pde": "tab:orange",
        "bc": "tab:green"
    }

    # --- Phase 1 (dashed line) ---
    plt.semilogy(x1, phase1_vals, "--", color=colors["total"], label="P1 total loss")
    plt.semilogy(x1, phase1_pde, "--", color=colors["pde"], alpha=0.7, label="P1 pde loss")
    plt.semilogy(x1, phase1_bc, "--", color=colors["bc"], alpha=0.7, label="P1 bc loss")

    # --- Phase 2 (solid line) ---
    plt.semilogy(x2, phase2_vals, "-", color=colors["total"], label="P2 total loss")
    plt.semilogy(x2, phase2_pde, "-", color=colors["pde"], alpha=0.7, label="P2 pde loss")
    plt.semilogy(x2, phase2_bc, "-", color=colors["bc"], alpha=0.7, label="P2 bc loss")

    # --- Separation ---
    plt.axvline(phase1_epochs, color="gray", linestyle=":", linewidth=1)
    # plt.text(phase1_epochs + 3, plt.ylim()[1] * 0.9, "→ Phase 2", fontsize=8, color="gray")

    plt.legend(fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()

    os.makedirs("loss_results", exist_ok=True)
    plt.savefig(os.path.join("loss_results", fname), dpi=900, bbox_inches="tight")
    plt.close()
    print(f"[saved] loss_results/{fname}")