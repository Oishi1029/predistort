# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Second figure: robustness, and the trade that hardening buys.

The point of this figure is a negative result, so it is drawn to make the
negative result unmissable rather than tucked into a corner: the kappa-robust
solution wins on the axis it was hardened against and loses badly on the one it
was not.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pulses as P  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figures"

C_NAIVE = "#c0392b"
C_CLASSICAL = "#e08e0b"
C_E2E = "#1f6feb"
C_ROBUST = "#7d3c98"
FLOOR = 1e-14
PLOT_FLOOR = 2e-7          # where a "< floor" point is drawn
COH = 3.2e-4


def _series(block, key, order):
    """Values for one arm, with sub-floor entries lifted to the plot floor."""
    out = []
    for k in order:
        v = block[k][key]
        out.append(v if v > PLOT_FLOOR else PLOT_FLOOR)
    return np.array(out)


def main():
    d = json.loads((RESULTS / "robustness.json").read_text())
    w = np.load(RESULTS / "robustness_waveforms.npz")
    FIGS.mkdir(exist_ok=True)

    arms = [("naive", C_NAIVE, "DRAG, electronics ignored", "o"),
            ("classical", C_CLASSICAL, "classical pre-distortion", "s"),
            ("end_to_end", C_E2E, "end-to-end", "D"),
            ("e2e_robust", C_ROBUST, "end-to-end, $\\kappa$-robust", "^")]

    fig = plt.figure(figsize=(13.0, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[1, 1, 0.95])

    # ---------------------------------------------- (a) bandwidth error
    ax = fig.add_subplot(gs[0, 0])
    order = sorted(d["bandwidth_error"], key=float)
    xs = np.array([float(k) for k in order]) * 100
    for key, c, lbl, m in arms:
        ax.semilogy(xs, _series(d["bandwidth_error"], key, order), m + "-",
                    color=c, lw=1.9, ms=5.5, label=lbl)
    ax.axhline(COH, color="k", ls="--", lw=1.1)
    ax.text(xs[0], COH * 1.25, "relaxation floor", fontsize=8, color="#333")
    ax.set_xlabel("error in the assumed line bandwidth (%)")
    ax.set_ylabel("gate infidelity  $1-\\bar{F}$")
    ax.set_title("mis-measure the line", fontsize=11)
    ax.grid(alpha=0.24, which="both")
    ax.set_ylim(PLOT_FLOOR / 1.5, 2e-2)

    # ---------------------------------------------- (b) kappa drift
    ax = fig.add_subplot(gs[0, 1])
    order = sorted(d["kappa_error"], key=float)
    xs = np.array([float(k) for k in order]) * 100
    for key, c, lbl, m in arms:
        ax.semilogy(xs, _series(d["kappa_error"], key, order), m + "-",
                    color=c, lw=1.9, ms=5.5, label=lbl)
    ax.axhline(COH, color="k", ls="--", lw=1.1)
    ax.set_xlabel("drift in the drive scale $\\kappa$ (%)")
    ax.set_title("drift the drive amplitude", fontsize=11)
    ax.grid(alpha=0.24, which="both")
    ax.set_ylim(PLOT_FLOOR / 1.5, 2e-2)
    ax.legend(fontsize=8.2, loc="lower center", framealpha=0.95)

    # ---------------------------------------------- (c) the trade
    ax = fig.add_subplot(gs[0, 2])
    kap = [k for k in sorted(d["kappa_error"], key=float) if abs(float(k)) > 1e-9]
    bws = [k for k in sorted(d["bandwidth_error"], key=float) if abs(float(k)) > 1e-9]
    gain_k = np.mean([d["kappa_error"][k]["end_to_end"]
                      / d["kappa_error"][k]["e2e_robust"] for k in kap])
    loss_b = np.mean([d["bandwidth_error"][k]["e2e_robust"]
                      / d["bandwidth_error"][k]["end_to_end"] for k in bws])

    ax.bar([0], [gain_k], color=C_ROBUST, width=0.55)
    ax.bar([1], [loss_b], color=C_NAIVE, width=0.55)
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["better under\n$\\kappa$ drift", "worse under\nline error"],
                       fontsize=9.5)
    ax.set_ylabel("factor vs plain end-to-end")
    ax.set_title("robustness is not a scalar", fontsize=11)
    for xpos, v in ((0, gain_k), (1, loss_b)):
        ax.text(xpos, v * 1.15, f"{v:.0f}×", ha="center", fontsize=12,
                fontweight="bold")
    ax.set_ylim(0.5, loss_b * 4)
    ax.grid(alpha=0.24, axis="y", which="both")
    ax.text(0.5, 0.62, "hardening against one\nuncertainty exposed another",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.95))

    fig.suptitle("Designed against the nominal 80 MHz model, scored against a "
                 "perturbed one", fontsize=12.5, y=1.02)
    fig.savefig(FIGS / "robustness.png", dpi=185, bbox_inches="tight")
    fig.savefig(FIGS / "robustness.pdf", bbox_inches="tight")
    print(f"wrote {FIGS/'robustness.png'}  "
          f"(kappa gain {gain_k:.2f}x, bandwidth loss {loss_b:.1f}x)")


if __name__ == "__main__":
    main()
