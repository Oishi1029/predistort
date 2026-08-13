# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submission's spine figure, built from results/sweep.json."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pulses as P  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figures"

C_FLOOR = "#8c8c8c"
C_NAIVE = "#c0392b"
C_CLASSICAL = "#e08e0b"
C_E2E = "#1f6feb"


def main():
    d = json.loads((RESULTS / "sweep.json").read_text())
    w = np.load(RESULTS / "sweep_waveforms.npz")
    rows = sorted(d["sweep"], key=lambda r: r["bw_mhz"])
    FIGS.mkdir(exist_ok=True)

    bw = np.array([r["bw_mhz"] for r in rows])
    a0 = np.array([r["arm0"] for r in rows])
    a1 = np.array([r["arm1"] for r in rows])
    a2 = np.array([r["arm2"] for r in rows])
    a3 = np.array([max(r["arm3"], 1e-17) for r in rows])  # log axis needs > 0
    demand = np.array([r["predistortion_demand"] for r in rows])
    coh = d["coherence_reference"]["value"]

    fig = plt.figure(figsize=(13.0, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.33, wspace=0.22)

    # ------------------------------------------------ (a) the sweep
    ax = fig.add_subplot(gs[0, :])
    ax.axhspan(1e-18, coh, color="#d8d8d8", alpha=0.55, zorder=0)
    ax.text(bw.max(), coh * 0.55,
            f"  below relaxation error of a {d['gate_window_ns']:.0f} ns gate "
            f"at $T_1={d['coherence_reference']['T1_us']:.0f}\\,\\mu$s "
            f"($t/T_1={coh:.1e}$)",
            ha="right", va="top", fontsize=8.5, color="#555")

    ax.semilogy(bw, a1, "o-", color=C_NAIVE, lw=2,
                label="DRAG, electronics ignored")
    ax.semilogy(bw, a2, "s-", color=C_CLASSICAL, lw=2,
                label="classical pre-distortion + recalibration (strong baseline)")
    ax.semilogy(bw, a3, "D-", color=C_E2E, lw=2.4,
                label="end-to-end gradients through both Tesseracts")
    ax.semilogy(bw, a0, ":", color=C_FLOOR, lw=1.6,
                label="DRAG on a perfect line (unreachable floor)")

    clipped = np.array([r["clipped"] for r in rows])
    if clipped.any():
        ax.axvline(bw[clipped].max(), color="k", ls="--", lw=1.1, alpha=0.55)
        ax.text(bw[clipped].max(), a1.max(),
                " DAC box exceeded\n by the classical inverse →",
                ha="right", va="top", fontsize=8.5)

    ax.invert_xaxis()
    ax.set_xlabel("control-line analogue bandwidth (MHz)   —   harder to the right")
    ax.set_ylabel("gate infidelity  $1-\\bar{F}$")
    ax.set_title("X90 on a three-level transmon: where classical pre-distortion "
                 "stops working", fontsize=12.5, pad=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # ------------------------------------------- (b) why it stops working
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(bw, demand, "s-", color=C_CLASSICAL, lw=2,
            label="peak DAC code the linear inverse demands")
    ax.axhline(d["u_max"], color="k", ls="--", lw=1.4,
               label=f"DAC box $u_{{\\max}}={d['u_max']}$")
    ax.fill_between(bw, d["u_max"], np.maximum(demand, d["u_max"]),
                    color=C_NAIVE, alpha=0.22, label="clipped — inverse is destroyed")
    ax.invert_xaxis()
    ax.set_xlabel("bandwidth (MHz)")
    ax.set_ylabel("peak commanded DAC code")
    ax.set_title("the mechanism: pre-emphasis outruns the DAC", fontsize=10.5)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper right")

    # ------------------------------------------- (c) waveforms, hardest point
    hardest = rows[0]["bw_mhz"] if bw[0] < bw[-1] else rows[-1]["bw_mhz"]
    hardest = int(min(bw))
    t = (np.arange(P.N_AWG) + 0.5) * P.DT_AWG
    ax = fig.add_subplot(gs[1, 1])
    ax.step(t, w[f"ui2_{hardest}"], where="mid", color=C_CLASSICAL, lw=1.7,
            label="classical, I")
    ax.step(t, w[f"ui3_{hardest}"], where="mid", color=C_E2E, lw=2.0,
            label="end-to-end, I")
    ax.step(t, w[f"uq3_{hardest}"], where="mid", color=C_E2E, lw=1.3, ls=":",
            label="end-to-end, Q")
    ax.axhline(d["u_max"], color="k", ls="--", lw=1.0)
    ax.axhline(-d["u_max"], color="k", ls="--", lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("commanded DAC code")
    ax.set_title(f"commanded pulses at {hardest} MHz", fontsize=10.5)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    fig.savefig(FIGS / "spine.png", dpi=185, bbox_inches="tight")
    fig.savefig(FIGS / "spine.pdf", bbox_inches="tight")
    print(f"wrote {FIGS/'spine.png'}")


if __name__ == "__main__":
    main()
