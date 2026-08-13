# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submission's spine figure, built from results/sweep.json.

A presentation note that is really an honesty note: the end-to-end arm sits at
the float64 floor of the metric (|1-F| < 1e-14, and sometimes a few times 1e-16
NEGATIVE, which is rounding noise about zero) at every bandwidth. Plotting those
values on a log axis would stretch it over sixteen decades and squash the only
informative part of the figure into a sliver -- and would imply a precision the
number does not carry. So the end-to-end arm is drawn as a floor band with its
bound stated, not as a curve pretending to resolve 1e-16.
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

C_FLOOR = "#8c8c8c"
C_NAIVE = "#c0392b"
C_CLASSICAL = "#e08e0b"
C_E2E = "#1f6feb"
Y_LO, Y_HI = 3e-7, 2e-2
FLOOR_BOUND = 1e-14


def main():
    d = json.loads((RESULTS / "sweep.json").read_text())
    w = np.load(RESULTS / "sweep_waveforms.npz")
    rows = sorted(d["sweep"], key=lambda r: -r["bw_mhz"])
    FIGS.mkdir(exist_ok=True)

    bw = np.array([r["bw_mhz"] for r in rows], float)
    a0 = np.array([r["arm0"] for r in rows])
    a1 = np.array([r["arm1"] for r in rows])
    a2 = np.array([r["arm2"] for r in rows])
    demand = np.array([r["predistortion_demand"] for r in rows])
    clipped = np.array([r["clipped"] for r in rows])
    coh = d["coherence_reference"]["value"]
    umax = d["u_max"]

    # bandwidth at which the linear inverse first leaves the DAC box
    # demand increases monotonically as bandwidth falls, so it is the
    # valid interpolant abscissa; bw is decreasing and must not be.
    cross = float(np.interp(umax, demand, bw))

    fig = plt.figure(figsize=(12.8, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1.0], hspace=0.36, wspace=0.23)

    # ------------------------------------------------------ (a) the sweep
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(bw))  # even spacing; the axis is categorical, not linear

    x_cross = float(np.interp(umax, demand, x))
    ax.axvspan(x_cross, x[-1] + 0.4,
               color="#c0392b", alpha=0.06, zorder=0)

    ax.plot(x, a1, "o-", color=C_NAIVE, lw=2.1, ms=6,
            label="DRAG, electronics ignored")
    ax.plot(x, a2, "s-", color=C_CLASSICAL, lw=2.4, ms=7,
            label="classical pre-distortion + recalibration  (strong baseline)")
    ax.plot(x, a0, ":", color=C_FLOOR, lw=1.8,
            label="DRAG on a perfect line  (unreachable floor)")

    # end-to-end: a bound, not a curve
    ax.axhspan(Y_LO, Y_LO * 2.2, color=C_E2E, alpha=0.30, zorder=1)
    ax.plot(x, np.full_like(x, Y_LO * 1.5, dtype=float), "D", color=C_E2E, ms=7,
            zorder=3,
            label=f"end-to-end through both Tesseracts  ($1-\\bar{{F}}$ < "
                  f"{FLOOR_BOUND:.0e}, the metric's float64 floor)")

    ax.axhline(coh, color="k", ls="--", lw=1.3)
    ax.text(0.15, coh * 1.35,
            f"relaxation error of a {d['gate_window_ns']:.0f} ns gate at "
            f"$T_1={d['coherence_reference']['T1_us']:.0f}\\,\\mu$s  "
            f"($t/T_1={coh:.1e}$) — below this line, differences are not observable",
            fontsize=8.6, color="#333")

    ax.axvline(x_cross, color="k", ls="--", lw=1.2,
               alpha=0.7)
    ax.text(x_cross + 0.06, 1.1e-2,
            f"  linear inverse leaves the\n  DAC box (~{cross:.0f} MHz)",
            fontsize=9, va="top")

    ax.set_yscale("log")
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bw])
    ax.set_xlim(-0.4, len(bw) - 0.6)
    ax.set_xlabel("control-line analogue bandwidth (MHz)      —      harder →")
    ax.set_ylabel("gate infidelity   $1-\\bar{F}$")
    ax.set_title("X90 on a three-level transmon: where classical pre-distortion "
                 "stops working", fontsize=13, pad=12)
    ax.grid(alpha=0.25, which="both", axis="y")
    ax.legend(loc="center left", fontsize=9, framealpha=0.96)

    worst = rows[-1]
    ax.annotate(
        f"at {int(worst['bw_mhz'])} MHz the classical stack is WORSE than doing\n"
        f"nothing: {worst['arm2']:.2e} vs {worst['arm1']:.2e}\n"
        f"— a clipped inverse is not an inverse",
        xy=(x[-1], worst["arm2"]), xytext=(x[-1] - 2.6, 4.5e-3),
        fontsize=9.2, ha="left",
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b", alpha=0.95))

    # -------------------------------------------- (b) the mechanism
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, demand, "s-", color=C_CLASSICAL, lw=2, ms=6,
            label="peak DAC code the linear inverse demands")
    ax.axhline(umax, color="k", ls="--", lw=1.4, label=f"DAC box $u_{{\\max}}={umax}$")
    ax.fill_between(x, umax, np.maximum(demand, umax), color=C_NAIVE, alpha=0.25,
                    label="unrealisable — clipped")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bw])
    ax.set_xlim(-0.4, len(bw) - 0.6)
    ax.set_xlabel("bandwidth (MHz)")
    ax.set_ylabel("peak commanded DAC code")
    ax.set_title("the mechanism: pre-emphasis outruns the converter", fontsize=10.8)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.2, loc="upper left")

    # -------------------------------------------- (c) pulses at the hard end
    hard = int(bw[-1])
    t = (np.arange(P.N_AWG) + 0.5) * P.DT_AWG
    ax = fig.add_subplot(gs[1, 1])
    ax.step(t, w[f"ui2_{hard}"], where="mid", color=C_CLASSICAL, lw=1.8,
            label="classical, I (clipped)")
    ax.step(t, w[f"ui3_{hard}"], where="mid", color=C_E2E, lw=2.0,
            label="end-to-end, I")
    ax.step(t, w[f"uq3_{hard}"], where="mid", color=C_E2E, lw=1.2, ls=":",
            label="end-to-end, Q")
    for s in (umax, -umax):
        ax.axhline(s, color="k", ls="--", lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("commanded DAC code")
    ax.set_title(f"commanded pulses at {hard} MHz", fontsize=10.8)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.2, loc="lower right")

    fig.savefig(FIGS / "spine.png", dpi=185, bbox_inches="tight")
    fig.savefig(FIGS / "spine.pdf", bbox_inches="tight")
    print(f"wrote {FIGS/'spine.png'}   (clip onset ~{cross:.0f} MHz)")


if __name__ == "__main__":
    main()
