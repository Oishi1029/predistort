# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is the end-to-end solution fragile? Measure it rather than caveat it.

The bandwidth sweep showed end-to-end optimisation reaching the numerical floor
at 80 MHz where the classical stack fails. But it gets there with aggressive
pre-emphasis -- five samples pinned at the DAC rail, large swings between
neighbours -- and a waveform like that is exactly what you would expect to be
sensitive to error in the instrument model it was designed against.

So: design each arm's pulse against the NOMINAL model, then score it against a
PERTURBED one, and see how fast each degrades. Two perturbations, both of them
things a real lab actually gets wrong:

  * the line bandwidth is mis-estimated (your VNA sweep is not the fridge)
  * the drive scale kappa has drifted since calibration

A fair test requires that the pulses never see the perturbation during design.
That is the whole point, so it is enforced by construction here: designs come
from the nominal pipeline, scores come from the perturbed one.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.signal import bessel
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

import pulses as P
from run_sweep import (
    FS_SIM, TARGET_ANGLE, U_MAX, Pipeline, codes_from_theta, nm,
    theta_from_codes,
)

jax.config.update("jax_enable_x64", True)

OUTDIR = Path.home() / "dev" / "tesseract-hack" / "_outputs"
RESULTS = Path(__file__).resolve().parent.parent / "results"

DESIGN_BW_MHZ = 80.0        # the regime where the classical stack has failed
BW_ERRORS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
KAPPA_ERRORS = [-0.05, -0.02, 0.0, 0.02, 0.05]


def sos_for(bw_hz):
    return bessel(3, bw_hz, btype="low", analog=False, output="sos",
                  fs=FS_SIM, norm="mag")


class ScaledPipeline(Pipeline):
    """Pipeline whose drive scale can be perturbed away from nominal."""

    def __init__(self, te, tq, sos, xsat=P.XSAT, kappa=P.KAPPA):
        super().__init__(te, tq, sos, xsat)
        self.kappa = float(kappa)

    def real(self, ui, uq):
        d = apply_tesseract(self.te, {
            "envelope_i": ui, "envelope_q": uq,
            "sos": self.sos,
            "xsat": jnp.float64(self.xsat),
            "kappa": jnp.float64(self.kappa)})
        return self._qubit(d["drive_i"], d["drive_q"])


def design_arms(pipe, sos, maxiter):
    """Reproduce the three arms against the NOMINAL model."""
    def obj_ideal(x):
        amp, beta = x
        if not (0.0 < amp <= U_MAX):
            return 1.0
        ui, uq = P.place(*P.drag_pair(amp, beta))
        if max(np.abs(ui).max(), np.abs(uq).max()) > U_MAX:
            return 1.0
        return pipe.f_ideal(ui, uq)

    x_id, _ = nm(obj_ideal, np.array([0.60, 0.5]))
    ui1, uq1 = P.place(*P.drag_pair(*x_id))

    def build2(x):
        i_s, q_s = P.drag_pair(x[0], x[2], phase=x[1])
        ui, uq = P.place(i_s, q_s)
        pi_, pq_ = P.classical_predistort(P.KAPPA * P.zoh(ui), P.KAPPA * P.zoh(uq),
                                          sos=sos)
        mask = np.zeros(P.N_AWG)
        mask[P.SUPPORT] = 1.0
        return (np.clip(pi_ * mask, -U_MAX, U_MAX),
                np.clip(pq_ * mask, -U_MAX, U_MAX))

    x2, _ = nm(lambda x: 1.0 if not (0 < x[0] <= U_MAX) else pipe.f_real(*build2(x)),
               np.array([x_id[0], 0.0, x_id[1]]))
    ui2, uq2 = build2(x2)

    loss = lambda th: pipe.real(*codes_from_theta(th))["infidelity"]  # noqa: E731
    vg = jax.value_and_grad(loss)

    def fg(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(fg, theta_from_codes(ui2, uq2), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-14})
    ui3, uq3 = (np.asarray(a) for a in codes_from_theta(jnp.asarray(res.x)))

    return {"naive": (ui1, uq1), "classical": (ui2, uq2), "end_to_end": (ui3, uq3)}


KAPPA_ENSEMBLE = [-0.04, 0.0, 0.04]


def design_robust(te, tq, sos, maxiter, x0):
    """Arm 4: optimise the EXPECTED infidelity over drive-scale uncertainty.

    The measurement above shows the end-to-end pulse is tolerant of a
    mis-estimated line but sensitive to drive-amplitude drift, which is the one
    parameter a lab knows it cannot hold. The fix costs nothing structural: keep
    the same two Tesseracts, evaluate the loss at several kappa values, and
    average. The gradient flows through every member of the ensemble, so this is
    the same composition doing strictly more work.
    """
    pipes = [ScaledPipeline(te, tq, sos, kappa=P.KAPPA * (1.0 + e))
             for e in KAPPA_ENSEMBLE]

    def loss(th):
        ui, uq = codes_from_theta(th)
        return sum(p.real(ui, uq)["infidelity"] for p in pipes) / len(pipes)

    vg = jax.value_and_grad(loss)

    def fg(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-14})
    return tuple(np.asarray(a) for a in codes_from_theta(jnp.asarray(res.x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=250)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    nominal_sos = sos_for(DESIGN_BW_MHZ * 1e6)

    with (
        Tesseract.from_image("electronics:latest", output_path=OUTDIR) as te,
        Tesseract.from_image("transmon:latest", output_path=OUTDIR) as tq,
    ):
        nominal = ScaledPipeline(te, tq, nominal_sos)
        print(f"designing all arms against the nominal {DESIGN_BW_MHZ:.0f} MHz model")
        arms = design_arms(nominal, nominal_sos, args.maxiter)

        print(f"designing arm 4: robust over kappa in "
              f"{[f'{e:+.0%}' for e in KAPPA_ENSEMBLE]}")
        arms["e2e_robust"] = design_robust(
            te, tq, nominal_sos, args.maxiter,
            theta_from_codes(*arms["end_to_end"]))

        for k, (a, b) in arms.items():
            print(f"  {k:12s} nominal infidelity {nominal.f_real(a, b):.4e}")

        out = {"design_bw_mhz": DESIGN_BW_MHZ, "bandwidth_error": {},
               "kappa_error": {}}

        print(f"\n--- line bandwidth mis-estimated (designed for "
              f"{DESIGN_BW_MHZ:.0f} MHz) ---")
        print(f"{'error':>8} " + " ".join(f"{k:>13}" for k in arms))
        for e in BW_ERRORS:
            true_bw = DESIGN_BW_MHZ * (1.0 + e)
            pipe = ScaledPipeline(te, tq, sos_for(true_bw * 1e6))
            vals = {k: pipe.f_real(*v) for k, v in arms.items()}
            out["bandwidth_error"][f"{e:+.2f}"] = vals
            print(f"{e:>+7.0%} " + " ".join(f"{vals[k]:>13.3e}" for k in arms))

        print("\n--- drive scale kappa drifted since calibration ---")
        print(f"{'error':>8} " + " ".join(f"{k:>13}" for k in arms))
        for e in KAPPA_ERRORS:
            pipe = ScaledPipeline(te, tq, nominal_sos, kappa=P.KAPPA * (1.0 + e))
            vals = {k: pipe.f_real(*v) for k, v in arms.items()}
            out["kappa_error"][f"{e:+.2f}"] = vals
            print(f"{e:>+7.0%} " + " ".join(f"{vals[k]:>13.3e}" for k in arms))

    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    np.savez(RESULTS / "robustness_waveforms.npz",
             **{f"{k}_{ax}": v[i] for k, v in arms.items()
                for i, ax in enumerate(("i", "q"))})
    print(f"\nwrote {RESULTS/'robustness.json'}")


if __name__ == "__main__":
    main()
