# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The experiment: how the four arms behave as the control line gets harder.

A single headline ratio invites the question "did you pick the regime that
flattered you?". So the result here is a sweep, not a point. The physical axis is
the analogue bandwidth of the control line, which is the thing that actually
varies between fridges, cable runs and filter stacks.

  arm 0  DRAG on a PERFECT line                    the unreachable floor
  arm 1  DRAG calibrated ignoring the electronics  what you get by pretending
  arm 2  classical pre-distortion + recalibration  the strong baseline
  arm 3  end-to-end gradients through both         this work
         Tesseracts

Every arm is scored with the identical virtual-Z-optimal, leakage-aware
infidelity from the transmon Tesseract, under the identical DAC amplitude box
and the identical 12 ns support inside a 16 ns window.

Expected physics: inverting a low-pass line means boosting high-frequency
content, so the narrower the line, the larger the DAC codes classical
pre-emphasis demands. Once those codes leave the box they are clipped, and a
clipped inverse is not an inverse. End-to-end optimisation never has that
failure mode because the box is inside its parameterisation rather than applied
after it.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.signal import bessel
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

import pulses as P

jax.config.update("jax_enable_x64", True)

U_MAX = 0.90
TARGET_ANGLE = np.pi / 2  # X90
FS_SIM = 16e9
BANDWIDTHS_MHZ = [400, 300, 250, 200, 160, 130, 100, 80]

# Reference scale, stated as a transparent formula rather than a claim:
# a 16 ns gate on a qubit with T1 = 50 us has t_gate/T1 = 3.2e-4 of relaxation.
T1_US = 50.0
T_WINDOW_NS = P.N_AWG * P.DT_AWG
COHERENCE_SCALE = (T_WINDOW_NS * 1e-3) / T1_US

OUTDIR = Path.home() / "dev" / "tesseract-hack" / "_outputs"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def sos_for(bw_hz):
    return bessel(3, bw_hz, btype="low", analog=False, output="sos",
                  fs=FS_SIM, norm="mag")


def codes_from_theta(theta):
    u = U_MAX * jnp.tanh(theta)
    mask = jnp.zeros(P.N_AWG).at[P.SUPPORT].set(1.0)
    return u[: P.N_AWG] * mask, u[P.N_AWG :] * mask


def theta_from_codes(ui, uq):
    z = np.concatenate([ui, uq]) / U_MAX
    return np.arctanh(np.clip(z, -0.999, 0.999))


class Pipeline:
    def __init__(self, te, tq, sos):
        self.te, self.tq, self.sos = te, tq, jnp.asarray(sos)

    def _qubit(self, di, dq):
        return apply_tesseract(self.tq, {
            "drive_i": di, "drive_q": dq,
            "dt": jnp.float64(P.DT_SIM),
            "anharmonicity": jnp.float64(P.ALPHA),
            "detunings": jnp.zeros(1),
            "target_angle": jnp.float64(TARGET_ANGLE)})

    def real(self, ui, uq):
        d = apply_tesseract(self.te, {"envelope_i": ui, "envelope_q": uq,
                                      "sos": self.sos})
        return self._qubit(d["drive_i"], d["drive_q"])

    def ideal(self, ui, uq):
        return self._qubit(P.KAPPA * jnp.repeat(ui, P.UPSAMPLE),
                           P.KAPPA * jnp.repeat(uq, P.UPSAMPLE))

    def f_real(self, ui, uq):
        return float(self.real(jnp.asarray(ui), jnp.asarray(uq))["infidelity"])

    def f_ideal(self, ui, uq):
        return float(self.ideal(jnp.asarray(ui), jnp.asarray(uq))["infidelity"])


def nm(obj, x0, maxfev=400):
    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-16, "maxfev": maxfev})
    return r.x, float(r.fun)


def run_point(pipe, sos, maxiter):
    # ---- arms 0 and 1: DRAG fitted against a perfect line ----------------
    def obj_ideal(x):
        amp, beta = x
        if not (0.0 < amp <= U_MAX):
            return 1.0
        ui, uq = P.place(*P.drag_pair(amp, beta))
        if max(np.abs(ui).max(), np.abs(uq).max()) > U_MAX:
            return 1.0
        return pipe.f_ideal(ui, uq)

    x_id, f0 = nm(obj_ideal, np.array([0.60, 0.5]))
    ui0, uq0 = P.place(*P.drag_pair(*x_id))
    f1 = pipe.f_real(ui0, uq0)

    # ---- arm 2: classical pre-distortion, recalibrated -------------------
    def build2(x):
        amp, phase, beta = x
        i_s, q_s = P.drag_pair(amp, beta, phase=phase)
        ui, uq = P.place(i_s, q_s)
        di, dq = P.KAPPA * P.zoh(ui), P.KAPPA * P.zoh(uq)
        pi_, pq_ = P.classical_predistort(di, dq, sos=sos)
        mask = np.zeros(P.N_AWG)
        mask[P.SUPPORT] = 1.0
        # the DAC cannot deliver what the inverse asks for; it clips
        return (np.clip(pi_ * mask, -U_MAX, U_MAX),
                np.clip(pq_ * mask, -U_MAX, U_MAX))

    def obj2(x):
        if not (0.0 < x[0] <= U_MAX):
            return 1.0
        return pipe.f_real(*build2(x))

    x2, f2 = nm(obj2, np.array([x_id[0], 0.0, x_id[1]]))
    ui2, uq2 = build2(x2)

    # how much pre-emphasis did the inverse ASK for, before clipping?
    i_s, q_s = P.drag_pair(x2[0], x2[2], phase=x2[1])
    uid, uqd = P.place(i_s, q_s)
    pre_i, pre_q = P.classical_predistort(P.KAPPA * P.zoh(uid),
                                          P.KAPPA * P.zoh(uqd), sos=sos)
    demand = float(np.abs(np.concatenate([pre_i, pre_q])).max())

    # ---- arm 3: end-to-end gradients -------------------------------------
    loss = lambda th: pipe.real(*codes_from_theta(th))["infidelity"]  # noqa: E731
    vg = jax.value_and_grad(loss)
    hist = []

    def fg(x):
        v, g = vg(jnp.asarray(x))
        hist.append(float(v))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(fg, theta_from_codes(ui2, uq2), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-14})
    ui3, uq3 = (np.asarray(a) for a in codes_from_theta(jnp.asarray(res.x)))
    f3 = pipe.f_real(ui3, uq3)

    return dict(arm0=f0, arm1=f1, arm2=f2, arm3=f3,
                predistortion_demand=demand, clipped=demand > U_MAX,
                evals=len(hist), hist=hist,
                ui0=ui0, uq0=uq0, ui2=ui2, uq2=uq2, ui3=ui3, uq3=uq3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=250)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows, waves = [], {}
    with (
        Tesseract.from_image("electronics:latest", output_path=OUTDIR) as te,
        Tesseract.from_image("transmon:latest", output_path=OUTDIR) as tq,
    ):
        print(f"{'BW/MHz':>7} {'arm0 ideal':>12} {'arm1 naive':>12} "
              f"{'arm2 classical':>15} {'arm3 e2e':>12} {'demand':>8} {'clip':>5}")
        for bw in BANDWIDTHS_MHZ:
            sos = sos_for(bw * 1e6)
            t0 = time.perf_counter()
            r = run_point(Pipeline(te, tq, sos), sos, args.maxiter)
            r["bw_mhz"] = bw
            r["seconds"] = time.perf_counter() - t0
            print(f"{bw:>7} {r['arm0']:>12.3e} {r['arm1']:>12.3e} "
                  f"{r['arm2']:>15.3e} {r['arm3']:>12.3e} "
                  f"{r['predistortion_demand']:>8.3f} "
                  f"{'YES' if r['clipped'] else 'no':>5}")
            for k in ("ui0", "uq0", "ui2", "uq2", "ui3", "uq3", "hist"):
                waves[f"{k}_{bw}"] = np.asarray(r.pop(k))
            rows.append(r)

    out = {
        "target": "X90 on a three-level transmon",
        "metric": "virtual-Z-optimal average gate infidelity, leakage-aware",
        "gate_window_ns": T_WINDOW_NS,
        "drive_support_ns": (P.SUPPORT.stop - P.SUPPORT.start) * P.DT_AWG,
        "u_max": U_MAX,
        "coherence_reference": {
            "note": "transparent reference scale, not a measurement: t_window/T1",
            "T1_us": T1_US,
            "value": COHERENCE_SCALE,
        },
        "sweep": rows,
    }
    (RESULTS / "sweep.json").write_text(json.dumps(out, indent=2))
    np.savez(RESULTS / "sweep_waveforms.npz", **waves)
    print(f"\nwrote {RESULTS/'sweep.json'}")


if __name__ == "__main__":
    main()
