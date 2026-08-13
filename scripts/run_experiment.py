# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The experiment: four arms, one metric, one number.

  arm 0  DRAG on a PERFECT line                      the unreachable floor
  arm 1  DRAG calibrated ignoring the electronics,   what you get if you
         then played through the real line           pretend the line is ideal
  arm 2  DRAG + the full classical pre-distortion    the strong baseline: every
         stack + recalibrated amp / phase / beta     correction a lab applies
  arm 3  end-to-end gradient optimisation through    this project
         both Tesseracts

Every arm is scored with the identical virtual-Z-optimal, leakage-aware gate
infidelity returned by the transmon Tesseract, and every arm obeys the same DAC
amplitude box and the same 12 ns support inside a 16 ns window.

The honest comparison is arm 3 against arm 2, not arm 3 against arm 1.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

import pulses as P

jax.config.update("jax_enable_x64", True)

U_MAX = 0.90
TARGET_ANGLE = np.pi / 2  # X90
OUTDIR = Path.home() / "dev" / "tesseract-hack" / "_outputs"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def codes_from_theta(theta):
    u = U_MAX * jnp.tanh(theta)
    mask = jnp.zeros(P.N_AWG).at[P.SUPPORT].set(1.0)
    return u[: P.N_AWG] * mask, u[P.N_AWG :] * mask


def theta_from_codes(ui, uq):
    """Inverse of codes_from_theta, for warm-starting arm 3 from arm 2."""
    z = np.concatenate([ui, uq]) / U_MAX
    return np.arctanh(np.clip(z, -0.999, 0.999))


class Pipeline:
    def __init__(self, t_elec, t_qubit):
        self.e, self.q = t_elec, t_qubit

    def _qubit(self, di, dq):
        return apply_tesseract(
            self.q,
            {
                "drive_i": di,
                "drive_q": dq,
                "dt": jnp.float64(P.DT_SIM),
                "anharmonicity": jnp.float64(P.ALPHA),
                "detunings": jnp.zeros(1),
                "target_angle": jnp.float64(TARGET_ANGLE),
            },
        )

    def real(self, ui, uq):
        """Through the electronics, i.e. what the qubit actually sees."""
        d = apply_tesseract(
            self.e, {"envelope_i": ui, "envelope_q": uq}
        )
        return self._qubit(d["drive_i"], d["drive_q"])

    def ideal(self, ui, uq):
        """A perfect line: zero-order hold and drive scaling only."""
        di = P.KAPPA * jnp.repeat(ui, P.UPSAMPLE)
        dq = P.KAPPA * jnp.repeat(uq, P.UPSAMPLE)
        return self._qubit(di, dq)

    def infid_real(self, ui, uq):
        return float(self.real(jnp.asarray(ui), jnp.asarray(uq))["infidelity"])

    def infid_ideal(self, ui, uq):
        return float(self.ideal(jnp.asarray(ui), jnp.asarray(uq))["infidelity"])


def calibrate_drag(objective, x0, label, bounds=None):
    """Fit a handful of scalars the way a lab calibrates: derivative-free."""
    evals = {"n": 0}

    def f(x):
        evals["n"] += 1
        return objective(x)

    res = minimize(f, x0, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-14, "maxfev": 2000})
    print(f"  {label}: {evals['n']} evals -> {res.fun:.6e}  at {np.round(res.x, 6)}")
    return res.x, float(res.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}

    with (
        Tesseract.from_image("electronics:latest", output_path=OUTDIR) as te,
        Tesseract.from_image("transmon:latest", output_path=OUTDIR) as tq,
    ):
        pipe = Pipeline(te, tq)

        # ---------------- arm 0 + 1: DRAG calibrated on the ideal model -----
        print("\n[arm 0/1] calibrating DRAG against a perfect line")

        def obj_ideal(x):
            amp, beta = x
            if not (0.0 < amp <= U_MAX):
                return 1.0
            ui, uq = P.place(*P.drag_pair(amp, beta))
            if np.abs(ui).max() > U_MAX or np.abs(uq).max() > U_MAX:
                return 1.0
            return pipe.infid_ideal(ui, uq)

        x_ideal, f_ideal = calibrate_drag(obj_ideal, np.array([0.35, 1.0]),
                                          "DRAG on perfect line")
        ui0, uq0 = P.place(*P.drag_pair(*x_ideal))
        f_arm1 = pipe.infid_real(ui0, uq0)
        print(f"  arm 0 (perfect line)          : {f_ideal:.6e}")
        print(f"  arm 1 (same pulse, real line) : {f_arm1:.6e}")

        # ---------------- arm 2: full classical pre-distortion --------------
        print("\n[arm 2] classical pre-distortion + recalibration")

        def build_arm2(x):
            amp, phase, beta = x
            i_sup, q_sup = P.drag_pair(amp, beta, phase=phase)
            di_des = P.KAPPA * P.zoh(P.place(i_sup, q_sup)[0])
            dq_des = P.KAPPA * P.zoh(P.place(i_sup, q_sup)[1])
            ui, uq = P.classical_predistort(di_des, dq_des)
            mask = np.zeros(P.N_AWG)
            mask[P.SUPPORT] = 1.0
            return np.clip(ui * mask, -U_MAX, U_MAX), np.clip(uq * mask, -U_MAX, U_MAX)

        def obj_arm2(x):
            if not (0.0 < x[0] <= U_MAX):
                return 1.0
            return pipe.infid_real(*build_arm2(x))

        x2, f_arm2 = calibrate_drag(obj_arm2,
                                    np.array([x_ideal[0], 0.0, x_ideal[1]]),
                                    "predistorted + recalibrated")
        ui2, uq2 = build_arm2(x2)

        # ---------------- arm 3: end-to-end gradients -----------------------
        print("\n[arm 3] end-to-end optimisation through both Tesseracts")
        loss = lambda th: pipe.real(*codes_from_theta(th))["infidelity"]  # noqa: E731
        vg = jax.value_and_grad(loss)

        theta0 = jnp.asarray(theta_from_codes(ui2, uq2))
        hist = []

        def fg(x):
            v, g = vg(jnp.asarray(x))
            hist.append(float(v))
            return float(v), np.asarray(g, dtype=np.float64)

        t0 = time.perf_counter()
        res = minimize(fg, np.asarray(theta0), jac=True, method="L-BFGS-B",
                       options={"maxiter": args.maxiter, "ftol": 1e-18,
                                "gtol": 1e-14})
        wall = time.perf_counter() - t0
        ui3, uq3 = (np.asarray(a) for a in codes_from_theta(jnp.asarray(res.x)))
        f_arm3 = pipe.infid_real(ui3, uq3)
        print(f"  {len(hist)} evals in {wall:.1f} s -> {f_arm3:.6e}")

        # ---------------- report --------------------------------------------
        leak3 = float(pipe.real(jnp.asarray(ui3), jnp.asarray(uq3))["leakage"])
        out = {
            "target": "X90",
            "metric": "virtual-Z-optimal average gate infidelity, leakage-aware",
            "arm0_drag_perfect_line": f_ideal,
            "arm1_drag_real_line": f_arm1,
            "arm2_classical_predistortion": f_arm2,
            "arm3_end_to_end": f_arm3,
            "gain_over_strong_baseline": f_arm2 / max(f_arm3, 1e-18),
            "gain_over_naive": f_arm1 / max(f_arm3, 1e-18),
            "arm3_leakage": leak3,
            "arm3_evals": len(hist),
            "arm3_wall_seconds": wall,
            "drag_cal_ideal": {"amp": float(x_ideal[0]), "beta": float(x_ideal[1])},
            "arm2_cal": {"amp": float(x2[0]), "phase": float(x2[1]),
                         "beta": float(x2[2])},
            "peak_dac_arm3": float(np.abs(np.concatenate([ui3, uq3])).max()),
            "u_max": U_MAX,
        }
        print("\n" + "=" * 62)
        for k in ("arm0_drag_perfect_line", "arm1_drag_real_line",
                  "arm2_classical_predistortion", "arm3_end_to_end"):
            print(f"  {k:32s} {out[k]:.6e}")
        print(f"  {'gain over STRONG baseline':32s} {out['gain_over_strong_baseline']:.1f}x")
        print("=" * 62)

        (RESULTS / "results.json").write_text(json.dumps(out, indent=2))
        np.savez(RESULTS / "waveforms.npz",
                 ui0=ui0, uq0=uq0, ui2=ui2, uq2=uq2, ui3=ui3, uq3=uq3,
                 hist=np.array(hist))
        print(f"\nwrote {RESULTS/'results.json'} and waveforms.npz")


if __name__ == "__main__":
    main()
