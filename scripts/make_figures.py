# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submission's spine figure, built from results/ produced by run_experiment.py."""

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
    res = json.loads((RESULTS / "results.json").read_text())
    w = np.load(RESULTS / "waveforms.npz")
    FIGS.mkdir(exist_ok=True)

    hist = w["hist"]
    t_awg = (np.arange(P.N_AWG) + 0.5) * P.DT_AWG
    t_sim = (np.arange(P.N_SIM) + 0.5) * P.DT_SIM

    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.24)

    # ---------------- (a) convergence, the headline ----------------------
    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(np.minimum.accumulate(hist), color=C_E2E, lw=2.2,
                label="end-to-end through both Tesseracts")
    for key, colour, label in [
        ("arm1_drag_real_line", C_NAIVE,
         "DRAG calibrated ignoring the electronics"),
        ("arm2_classical_predistortion", C_CLASSICAL,
         "classical pre-distortion (mixer + AM/AM + Tikhonov + recal)"),
        ("arm0_drag_perfect_line", C_FLOOR,
         "DRAG on a perfect line (unreachable floor)"),
    ]:
        ax.axhline(res[key], color=colour, ls="--", lw=1.6, label=label)

    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("gate infidelity  $1-\\bar{F}$")
    ax.set_title(
        "X90 on a three-level transmon: gradients across a Julia/JAX container boundary",
        fontsize=12, pad=10,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    gain = res["gain_over_strong_baseline"]
    ax.annotate(
        f"{gain:.1f}× better than the strong classical baseline\n"
        f"{res['arm2_classical_predistortion']:.2e}  →  {res['arm3_end_to_end']:.2e}",
        xy=(0.015, 0.06), xycoords="axes fraction", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=C_E2E, alpha=0.95),
    )

    # ---------------- (b) commanded DAC codes ----------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.step(t_awg, w["ui2"], where="mid", color=C_CLASSICAL, lw=1.6,
            label="classical, I")
    ax.step(t_awg, w["uq2"], where="mid", color=C_CLASSICAL, lw=1.2, ls=":",
            label="classical, Q")
    ax.step(t_awg, w["ui3"], where="mid", color=C_E2E, lw=1.9, label="end-to-end, I")
    ax.step(t_awg, w["uq3"], where="mid", color=C_E2E, lw=1.4, ls=":",
            label="end-to-end, Q")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("commanded DAC code")
    ax.set_title("what the AWG is told to play", fontsize=10.5)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)

    # ---------------- (c) delivered drive --------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_sim, P.KAPPA * P.zoh(w["ui0"]), color=C_FLOOR, lw=1.3,
            label="ideal DRAG (what you wanted)")
    ax.plot(t_sim, P.lti_forward(w["ui0"]) * P.KAPPA, color=C_NAIVE, lw=1.5,
            label="DRAG after the line (what arrives)")
    ax.plot(t_sim, P.lti_forward(w["ui3"]) * P.KAPPA, color=C_E2E, lw=1.7,
            label="pre-distorted after the line")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("in-phase drive (rad/ns)")
    ax.set_title("in-phase drive at the qubit", fontsize=10.5)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    fig.savefig(FIGS / "spine.png", dpi=190, bbox_inches="tight")
    fig.savefig(FIGS / "spine.pdf", bbox_inches="tight")
    print(f"wrote {FIGS/'spine.png'}")


if __name__ == "__main__":
    main()
