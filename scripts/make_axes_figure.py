# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both hardness axes side by side: they fail for one shared reason.

Bandwidth and compression are independent knobs and the classical stack has an
exact inverse for each in isolation -- a Tikhonov pre-emphasis for the line, a
closed-form AM/AM inverse for the compressor. Yet both axes end the same way,
and the bottom row shows why: the inverse the classical stack needs is not
inside the hardware's reachable set. Once the commanded code the inverse demands
leaves the DAC box, clipping destroys it. An optimiser that parameterises the
box away never leaves the reachable set in the first place.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figures"

C_NAIVE = "#c0392b"
C_CLASSICAL = "#e08e0b"
C_E2E = "#1f6feb"
C_FLOOR = "#8c8c8c"
PLOT_FLOOR = 3e-7
COH = 3.2e-4


def panel(ax_top, ax_bot, rows, keyname, xlabel, title, umax, invert):
    k = np.array([r["key"] for r in rows], float)
    x = np.arange(len(k))
    a0 = np.array([r["arm0"] for r in rows])
    a1 = np.array([r["arm1"] for r in rows])
    a2 = np.array([r["arm2"] for r in rows])
    dem = np.array([r["predistortion_demand"] for r in rows])

    ax_top.axhspan(PLOT_FLOOR / 1.4, PLOT_FLOOR * 2.0, color=C_E2E, alpha=0.28)
    ax_top.plot(x, np.full_like(x, PLOT_FLOOR * 1.4, dtype=float), "D",
                color=C_E2E, ms=6.5, zorder=3,
                label="end-to-end  ($<10^{-14}$, metric floor)")
    ax_top.plot(x, a1, "o-", color=C_NAIVE, lw=2, ms=5.5,
                label="DRAG, electronics ignored")
    ax_top.plot(x, a2, "s-", color=C_CLASSICAL, lw=2.3, ms=6,
                label="classical pre-distortion")
    ax_top.plot(x, a0, ":", color=C_FLOOR, lw=1.6, label="perfect-line floor")
    ax_top.axhline(COH, color="k", ls="--", lw=1.1)
    ax_top.text(0.05, COH * 1.3, "relaxation floor", fontsize=8, color="#333")
    ax_top.set_yscale("log")
    ax_top.set_ylim(PLOT_FLOOR / 1.4, 3e-1)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{v:g}" for v in k])
    ax_top.set_title(title, fontsize=11.5)
    ax_top.set_ylabel("gate infidelity  $1-\\bar{F}$")
    ax_top.grid(alpha=0.24, which="both", axis="y")

    ax_bot.plot(x, dem, "s-", color=C_CLASSICAL, lw=2, ms=6,
                label="DAC code the classical inverse demands")
    ax_bot.axhline(umax, color="k", ls="--", lw=1.3,
                   label=f"DAC box $u_{{\\max}}={umax}$")
    ax_bot.fill_between(x, umax, np.maximum(dem, umax), color=C_NAIVE, alpha=0.22,
                        label="unrealisable")
    if dem.max() / dem.min() > 20:
        ax_bot.set_yscale("log")
    else:
        ax_bot.set_ylim(0.7, max(1.45, dem.max() * 1.06))
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"{v:g}" for v in k])
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("peak commanded code")
    ax_bot.grid(alpha=0.24, which="both", axis="y")


def main():
    bwd = json.loads((RESULTS / "sweep.json").read_text())
    cmp_ = json.loads((RESULTS / "sweep_compression.json").read_text())
    FIGS.mkdir(exist_ok=True)

    bw_rows = sorted(bwd["sweep"], key=lambda r: -r["bw_mhz"])
    for r in bw_rows:
        r["key"] = r["bw_mhz"]
    cm_rows = sorted(cmp_["sweep"], key=lambda r: -r["xsat"])
    for r in cm_rows:
        r["key"] = r["xsat"]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0),
                             gridspec_kw={"height_ratios": [1.5, 1.0],
                                          "hspace": 0.16, "wspace": 0.20},
                             sharex="col")

    panel(axes[0, 0], axes[1, 0], bw_rows, "bw_mhz",
          "control-line bandwidth (MHz)      harder →",
          "narrower line", bwd["u_max"], True)
    panel(axes[0, 1], axes[1, 1], cm_rows, "xsat",
          "amplifier saturation level $x_{sat}$ (DAC units)      harder →",
          "harder compression", cmp_["u_max"], True)

    axes[0, 0].legend(fontsize=8.4, loc="upper left", framealpha=0.96)
    axes[1, 1].legend(fontsize=8.2, loc="upper left", framealpha=0.96)

    axes[0, 1].annotate(
        "the AM/AM inverse stops existing:\nit asks for 1654× the DAC range",
        xy=(len(cm_rows) - 1, cm_rows[-1]["arm2"]),
        xytext=(len(cm_rows) - 5.6, 4e-3), fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c0392b", alpha=0.95))

    fig.suptitle("Two independent hardness axes, one shared failure: the classical "
                 "inverse leaves the hardware's reachable set",
                 fontsize=13, y=0.965)
    fig.savefig(FIGS / "axes.png", dpi=185, bbox_inches="tight")
    fig.savefig(FIGS / "axes.pdf", bbox_inches="tight")
    print(f"wrote {FIGS/'axes.png'}")


if __name__ == "__main__":
    main()
