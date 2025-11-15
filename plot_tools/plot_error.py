import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from numpy.linalg import norm
import json
from matplotlib.ticker import FuncFormatter

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

def get_nested_save_dir(base_save_dir, config_path="config.json"):

    if not os.path.exists(config_path):
        print(f"[warn] config.json not found: {config_path}")
        return os.path.join("eps_unknown", "cy_unknown", base_save_dir)

    with open(config_path, "r") as f:
        cfg_ = json.load(f)

    # eps
    eps_val = cfg_.get("eps_global", None)
    if eps_val is None:
        eps_folder = "eps_unknown"
    else:
        eps_prefix = "p" if eps_val >= 0 else "m"
        eps_folder = f"eps_{eps_prefix}{str(abs(eps_val)).replace('.', '_')}"

    # cy
    cy_val = cfg_.get("cy", None) or cfg_.get("cy_val", None)
    if cy_val is None:
        cy_folder = "cy_unknown"
    else:
        cy_prefix = "p" if cy_val >= 0 else "m"
        cy_folder = f"cy_{cy_prefix}{str(abs(cy_val)).replace('.', '_')}"

    final_dir = os.path.join(eps_folder, cy_folder, base_save_dir)
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def sci_cb(cbar, vmin, vmax):
    max_val = max(abs(vmin), abs(vmax))
    if max_val == 0:
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(max_val)))

    # tick formatter
    def coeff_formatter(y, pos):
        return f"{y / (10 ** exponent):.1f}"

    cbar.formatter = FuncFormatter(coeff_formatter)
    cbar.update_ticks()

    # ---- put exponent at top of colorbar ----
    if exponent != 0:
        # create title
        title = cbar.ax.set_title(
            rf"$\times 10^{{{exponent}}}$",
            fontsize=12,
            pad=4,
        )
        # shift title to the right (adjust x value as desired)
        title.set_position((2.0, 1.02))


# ----------------------------------------------------------
# ---------- /2_true_pred ---------------------------------
# ----------------------------------------------------------
def plot_error_dis(model, x_coll,
                   function_target,
                   phase_name="phase",
                   save_dir="2_true_pred"):

    save_dir = get_nested_save_dir(save_dir, "config.json")

    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------
    # ----- Safe histogram for Re & Im in percentage form ------
    # ----------------------------------------------------------
    def safe_hist(_re_data, _im_data, _bins, xlabel, ylabel, filename, title=None):
        """Plot Re/Im absolute diff histogram with percentage weights."""
        re_vals = np.asarray(_re_data)
        im_vals = np.asarray(_im_data)
        re_vals = re_vals[np.isfinite(re_vals)]
        im_vals = im_vals[np.isfinite(im_vals)]

        if len(re_vals) == 0 or len(im_vals) == 0:
            print(f"[warn] Empty or invalid data for {filename}")
            return

        # convert to percentage weights
        w_re = np.ones_like(re_vals) * (100.0 / len(re_vals))
        w_im = np.ones_like(im_vals) * (100.0 / len(im_vals))

        plt.figure(figsize=(6, 4))
        plt.hist(re_vals, bins=_bins, weights=w_re, histtype='step', color='tab:blue', lw=1.8)
        plt.hist(im_vals, bins=_bins, weights=w_im, histtype='step', color='tab:red', alpha=0.8, lw=1.8)
        plt.xscale('log')

        # legend with line handles
        legend_lines = [
            Line2D([0], [0], color='tab:blue', lw=1.8, label='Re'),
            Line2D([0], [0], color='tab:red', lw=1.8, alpha=0.8, label='Im')
        ]

        if title is not None:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(
            fontsize=9,
            handles=legend_lines,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    # ---------------------------------
    # ---------------------------------
    model.eval()
    with torch.no_grad():
        pred = model(x_coll)
        true = function_target

    print("true.mean abs:", true.abs().mean().item())
    print("pred.mean abs:", pred.abs().mean().item())

    n_basis = pred.shape[1] // 2
    pred_re, pred_im = pred[:, :n_basis], pred[:, n_basis:]
    true_re, true_im = true[:, :n_basis], true[:, n_basis:]

    func_labels = ["I1", "I2", "I3", "I4"] if n_basis == 4 else [f"Func{i + 1}" for i in range(n_basis)]
    func_labels2 = ["$\mathcal{I}_1$", "$\mathcal{I}_2$", "$\mathcal{I}_3$", "$\mathcal{I}_4$"] \
        if n_basis == 4 else [f"Func{i + 1}" for i in range(n_basis)]
    #colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # -------------------------------------------
    # ---------- Re vs Im True + Pred ----------
    # -------------------------------------------
    pred_re_np = pred_re.cpu().numpy()
    pred_im_np = pred_im.cpu().numpy()
    true_re_np = true_re.cpu().numpy()
    true_im_np = true_im.cpu().numpy()

    for j in range(n_basis):
        plt.figure(figsize=(6, 4))
        plt.scatter(true_re_np[:, j], true_im_np[:, j], s=16, c='tab:blue', alpha=1.0, marker='o', label="True")
        plt.scatter(pred_re_np[:, j], pred_im_np[:, j], s=12, c='tab:red', alpha=0.7, marker='x', label="Pred")
        plt.xlabel(f"Re({func_labels2[j]})")
        plt.ylabel(f"Im({func_labels2[j]})")
        #plt.title(f"{func_labels2[j]} : Re vs Im ({phase_name}, $\\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})")
        plt.title(f"{phase_name} ($\\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})")
        plt.legend(
            fontsize=9,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0
        )
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{phase_name}_ReIm_{func_labels[j]}.png"), dpi=200)
        plt.close()

    print(f"[saved] True_Pred_Re_Im to /{save_dir}")

    # ------------------------------------------------------------
    # ---------- Average Re vs Im (new figure) -------------------
    # ------------------------------------------------------------

    # avg over basis dimension = average over I1,I2,I3,I4
    true_re_avg = true_re_np.mean(axis=1)  # (N,)
    true_im_avg = true_im_np.mean(axis=1)

    pred_re_avg = pred_re_np.mean(axis=1)
    pred_im_avg = pred_im_np.mean(axis=1)

    plt.figure(figsize=(6, 4))

    plt.scatter(true_re_avg, true_im_avg,
                s=16, c='tab:blue', alpha=1.0, marker='o', label="True")
    plt.scatter(pred_re_avg, pred_im_avg,
                s=12, c='tab:red', alpha=0.7, marker='x', label="Pred")

    plt.xlabel(r"Re($\overline{\mathcal{I}}$)")
    plt.ylabel(r"Im($\overline{\mathcal{I}}$)")
    plt.title(f"{phase_name} ($\\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})")

    plt.legend(
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0
    )
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{phase_name}_ReIm_Ave.png"), dpi=200)
    plt.close()

    print(f"[saved] True_Pred_ReIm_Ave to /{save_dir}")

    # ---------------------------------------------------------
    # ---------- Absolute difference per function ----------
    # ---------------------------------------------------------
    bins = None
    for j in range(n_basis):
        abs_re_j = torch.abs(pred_re[:, j] - true_re[:, j]).cpu().numpy()
        abs_im_j = torch.abs(pred_im[:, j] - true_im[:, j]).cpu().numpy()
        if bins is None:
            bins = np.logspace(-10, 0, 60)

        sym = func_labels2[j].strip('$') if func_labels2[j].startswith('$') else func_labels2[j]

        safe_hist(
            abs_re_j, abs_im_j, bins,
            #xlabel=rf'$|{sym}^{{\mathrm{{pred}}}} - {sym}^{{\mathrm{{true}}}}|$',
            xlabel=rf'$|\Delta{sym}|$',
            ylabel=r'$\%$',
            filename=os.path.join(save_dir, f"{phase_name}_AbsDiff_{func_labels[j]}.png"),
            title=rf"{phase_name} ($\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})"
        )

    print(f"[saved] |Re_Diff| & |Im_Diff| I1-I4 to /{save_dir}")

    # ------------------------------------------------------------
    # ---------- Absolute difference Ave ----------
    # ------------------------------------------------------------
    abs_re = torch.abs(pred_re - true_re).flatten().cpu().numpy() # (4N,)
    abs_im = torch.abs(pred_im - true_im).flatten().cpu().numpy() # (4N,)

    safe_hist(
        abs_re, abs_im, bins,
        #xlabel=r'$|{\vec{\mathcal{I}}}_{\mathrm{pred}} - \vec{\mathcal{I}}_{\mathrm{true}}|$',
        xlabel=r'$|\overline{\Delta{\mathcal{I}}}|$',
        ylabel=r'$\%$',
        filename=os.path.join(save_dir, f"{phase_name}_AbsDiff_Ave.png"),
        title = rf"{phase_name} ($\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})"
    )
    print(f"[saved] |Re_Diff| & |Im_Diff| Ave(I) to /{save_dir}")

# ----------------------------------------------------------
# ---------- /3_true_pred_x1_x2 ---------------------------------
# ---------- Re/Im-part signed difference plotting ----------
# -----------------------------------------------------------
def plot_signed_diff_map(x1, x2, diff_values, title, filename):
    vmax = np.max(np.abs(diff_values))

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x1, x2, c=diff_values, cmap="RdBu_r", s=20)

    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.clim(-vmax, vmax)
    sci_cb(cbar, -vmax, vmax)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
# -------------------------------------------
# ---------- Re/Im signed difference----------
# -------------------------------------------
def plot_x1_x2(x_coll, true_vals, pred_vals,
               func_labels=None,
               save_dir="3_true_pred_x1_x2",
               phase_name="phase"):

    save_dir = get_nested_save_dir(save_dir, "config.json")
    os.makedirs(save_dir, exist_ok=True)

    n_basis = pred_vals.shape[1] // 2

    pred_re = pred_vals[:, :n_basis]
    pred_im = pred_vals[:, n_basis:]
    true_re = true_vals[:, :n_basis]
    true_im = true_vals[:, n_basis:]

    func_labels = func_labels or [f"I{i+1}" for i in range(n_basis)]
    func_tex = [rf"\mathcal{{I}}_{i+1}" for i in range(n_basis)]

    x1, x2 = x_coll[:, 0], x_coll[:, 1]
    # ------------------------------
    # 1) Plot each basis component
    # ------------------------------
    for j in range(n_basis):

        diff_re_j = pred_re[:, j] - true_re[:, j]
        diff_im_j = pred_im[:, j] - true_im[:, j]

        # ---- Re part ----
        plot_signed_diff_map(
            x1, x2, diff_re_j,
            title=rf"$\mathrm{{Re}}(\Delta{func_tex[j]})$ "
                  rf"({phase_name}, $\varepsilon$={cfg.eps_global}, $c$={cfg.cy})",
            filename=os.path.join(save_dir, f"{phase_name}_ReDiff_{func_labels[j]}.png")
        )

        # ---- Im part ----
        plot_signed_diff_map(
            x1, x2, diff_im_j,
            title=rf"$\mathrm{{Im}}(\Delta{func_tex[j]})$ "
                  rf"({phase_name}, $\varepsilon$={cfg.eps_global}, $c$={cfg.cy})",
            filename=os.path.join(save_dir, f"{phase_name}_ImDiff_{func_labels[j]}.png")
        )
    # -------------------------------------------------
    # 2) Total average difference across all basis
    # -------------------------------------------------
    total_diff_re = np.mean(pred_re - true_re, axis=1)
    total_diff_im = np.mean(pred_im - true_im, axis=1)

    plot_signed_diff_map(
        x1, x2, total_diff_re,
        title=rf"$\mathrm{{Re}}(\overline{{\Delta\mathcal{{I}}}})$ "
              rf"({phase_name}, $\varepsilon$={cfg.eps_global}, $c$={cfg.cy})",
        filename=os.path.join(save_dir, f"{phase_name}_ReDiff_Ave.png")
    )

    plot_signed_diff_map(
        x1, x2, total_diff_im,
        title=rf"$\mathrm{{Im}}(\overline{{\Delta\mathcal{{I}}}})$ "
              rf"({phase_name}, $\varepsilon$={cfg.eps_global}, $c$={cfg.cy})",
        filename=os.path.join(save_dir, f"{phase_name}_ImDiff_Ave.png")
    )

    print(f"[saved] signed-difference I1-I4 & Ave to /{save_dir}")


# -----------------------------------------------
# ---------- Cosine similarity plotting ----------
# -----------------------------------------------
def plot_cosine_similarity(model, x_coll, function_target,
                           func_labels=None,
                           save_dir="4_cosine_similarity",
                           phase_name="phase"):
    save_dir = get_nested_save_dir(save_dir, "config.json")

    os.makedirs(save_dir, exist_ok=True)

    # ---------- compute prediction ----------
    model.eval()
    with torch.no_grad():
        pred = model(x_coll).detach().cpu().numpy()

    true = function_target.detach().cpu().numpy() if hasattr(function_target, "detach") else function_target

    # ---------- split Re/Im ----------
    n_basis = pred.shape[1] // 2
    true_re, true_im = true[:, :n_basis], true[:, n_basis:]
    pred_re, pred_im = pred[:, :n_basis], pred[:, n_basis:]

    func_labels = func_labels or ["I1", "I2", "I3", "I4"]
    func_labels2 = [
        "\\mathcal{I}_1", "\\mathcal{I}_2", "\\mathcal{I}_3", "\\mathcal{I}_4"
    ] if n_basis == 4 else [f"Func{i + 1}" for i in range(n_basis)]

    eps = 1e-15

    for j in range(n_basis):
        re_true, re_pred = true_re[:, j], pred_re[:, j]
        im_true, im_pred = true_im[:, j], pred_im[:, j]

        # ---- continuous cosine similarity ----
        cos_re_vals = (re_true * re_pred) / (
                np.sqrt(re_true ** 2 + eps) * np.sqrt(re_pred ** 2 + eps)
        )
        cos_im_vals = (im_true * im_pred) / (
                np.sqrt(im_true ** 2 + eps) * np.sqrt(im_pred ** 2 + eps)
        )

        cos_re_vals = np.clip(cos_re_vals, -1, 1)
        cos_im_vals = np.clip(cos_im_vals, -1, 1)

        # ---- histogram ----
        bins = np.linspace(-1, 1, 60)
        w_re = np.ones_like(cos_re_vals) * (100.0 / len(cos_re_vals))
        w_im = np.ones_like(cos_im_vals) * (100.0 / len(cos_im_vals))

        plt.figure(figsize=(6, 4))
        plt.hist(cos_re_vals, bins=bins, weights=w_re, histtype='step',
                 lw=1.8, color='tab:blue', label='Re')
        plt.hist(cos_im_vals, bins=bins, weights=w_im, histtype='step',
                 lw=1.8, color='tab:red', alpha=0.8, label='Im')

        # ---- mean cosine similarity ----
        mean_re = np.nanmean(cos_re_vals)
        mean_im = np.nanmean(cos_im_vals)
        plt.axvline(mean_re, color='tab:blue', linestyle='--', alpha=0.7)
        plt.axvline(mean_im, color='tab:red', linestyle='--', alpha=0.7)

        ylim = plt.ylim()[1]
        plt.text(mean_re + 0.03, ylim * 0.85,
                 f"{mean_re:+.2f}", color='tab:blue', fontsize=9, ha='left')
        plt.text(mean_im + 0.03, ylim * 0.75,
                 f"{mean_im:+.2f}", color='tab:red', alpha=0.8, fontsize=9, ha='left')

        plt.xlim(-1.1, 1.1)
        plt.xticks(np.linspace(-1, 1, 5))
        plt.xlabel(r"$\cos\theta$")
        plt.ylabel(r"$\%$")
        plt.title(rf"${func_labels2[j]}$ ({phase_name}, $\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # ---- legend with line handles ----
        legend_lines = [
            Line2D([0], [0], color='tab:blue', lw=1.8, label='Re'),
            Line2D([0], [0], color='tab:red', alpha=0.8, lw=1.8, label='Im')
        ]
        plt.legend(handles=legend_lines, frameon=False)

        plt.savefig(
            os.path.join(save_dir, f"{phase_name}_CosSim_{func_labels[j]}.png"),
            dpi=200
        )
        plt.close()

        # -----------------------------------
        # Total / Average cosine similarity
        # -----------------------------------
        # Combine across all functions
        total_cos_re_vals = []
        total_cos_im_vals = []

        for j in range(n_basis):
            re_true, re_pred = true_re[:, j], pred_re[:, j]
            im_true, im_pred = true_im[:, j], pred_im[:, j]

            cos_re_vals = (re_true * re_pred) / (
                    np.sqrt(re_true ** 2 + eps) * np.sqrt(re_pred ** 2 + eps)
            )
            cos_im_vals = (im_true * im_pred) / (
                    np.sqrt(im_true ** 2 + eps) * np.sqrt(im_pred ** 2 + eps)
            )
            total_cos_re_vals.append(cos_re_vals)
            total_cos_im_vals.append(cos_im_vals)

        # Stack then average across basis dimension
        total_cos_re_vals = np.nanmean(np.stack(total_cos_re_vals, axis=1), axis=1)
        total_cos_im_vals = np.nanmean(np.stack(total_cos_im_vals, axis=1), axis=1)

        # ---- histogram for total ----
        bins = np.linspace(-1.0, 1.0, 60)
        w_re = np.ones_like(total_cos_re_vals) * (100.0 / len(total_cos_re_vals))
        w_im = np.ones_like(total_cos_im_vals) * (100.0 / len(total_cos_im_vals))

        plt.figure(figsize=(6, 4))
        plt.hist(total_cos_re_vals, bins=bins, weights=w_re, histtype='step',
                 lw=1.8, color='tab:blue', label='Re (avg)')
        plt.hist(total_cos_im_vals, bins=bins, weights=w_im, histtype='step',
                 lw=1.8, color='tab:red', alpha=0.8, label='Im (avg)')

        mean_re_total = np.nanmean(total_cos_re_vals)
        mean_im_total = np.nanmean(total_cos_im_vals)
        plt.axvline(mean_re_total, color='tab:blue', linestyle='--', alpha=0.7)
        plt.axvline(mean_im_total, color='tab:red', linestyle='--', alpha=0.7)

        ylim = plt.ylim()[1]
        plt.text(mean_re_total + 0.03, ylim * 0.85,
                 f"{mean_re_total:+.2f}", color='tab:blue', fontsize=9, ha='left')
        plt.text(mean_im_total + 0.03, ylim * 0.75,
                 f"{mean_im_total:+.2f}", color='tab:red', alpha=0.8, fontsize=9, ha='left')

        plt.xlim(-1.1, 1.1)
        plt.xticks(np.linspace(-1, 1, 5))
        plt.xlabel(r"$\cos\theta$")
        plt.ylabel(r"$\%$")
        plt.title(
            rf"$\overline{{\mathcal{{I}}}}$ ({phase_name}, $\varepsilon$ = {cfg.eps_global}, $c$ = {cfg.cy})"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        legend_lines = [
            Line2D([0], [0], color='tab:blue', lw=1.8, label='Re'),
            Line2D([0], [0], color='tab:red', lw=1.8, alpha=0.8, label='Im')
        ]
        plt.legend(handles=legend_lines, frameon=False)

        plt.savefig(
            os.path.join(save_dir, f"{phase_name}_CosSim_Ave.png"),
            dpi=200
        )
        plt.close()

    print(f"[saved] cosine similarity to /{save_dir}")