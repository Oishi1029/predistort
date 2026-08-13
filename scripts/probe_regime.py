# SPDX-License-Identifier: Apache-2.0
"""Where does classical pre-distortion actually break down?

The first experiment showed the classical stack already reaching 1.2e-5 on a
12 ns gate -- below any realistic decoherence floor, so the residual the gradient
removes is physically irrelevant there. That is an honest finding, and it says
the interesting regime is elsewhere.

Physical hypothesis: the classical stack fails when linear pre-emphasis demands
DAC codes the DAC cannot produce. Inverting a low-pass line means boosting the
high-frequency content of the command, and the shorter the gate the more boost
is required. Once the required codes leave the box they get clipped, and a
clipped inverse is no inverse. End-to-end optimisation never has that problem,
because the box is built into its parameterisation rather than applied after it.

So sweep GATE DURATION and watch the two arms separate.
"""

import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import pulses as P

jax.config.update("jax_enable_x64", True)
OUT = Path.home() / "dev" / "tesseract-hack" / "_outputs"
U_MAX, TARGET = 0.90, np.pi / 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (
        Tesseract.from_image("electronics:latest", output_path=OUT) as te,
        Tesseract.from_image("transmon:latest", output_path=OUT) as tq,
    ):
        def qubit(di, dq):
            return apply_tesseract(tq, {
                "drive_i": di, "drive_q": dq,
                "dt": jnp.float64(P.DT_SIM),
                "anharmonicity": jnp.float64(P.ALPHA),
                "detunings": jnp.zeros(1),
                "target_angle": jnp.float64(TARGET)})

        def real(ui, uq):
            d = apply_tesseract(te, {"envelope_i": jnp.asarray(ui),
                                     "envelope_q": jnp.asarray(uq)})
            return qubit(d["drive_i"], d["drive_q"])

        print(f"{'n_sup':>6} {'t_gate':>7} {'peak DAC needed':>16} "
              f"{'clipped?':>9}   (classical pre-emphasis demand)")
        for n_sup in (24, 20, 16, 12, 10, 8, 6):
            t_gate = n_sup * P.DT_AWG
            # what does the classical stack ASK the DAC for, before clipping?
            env = P.gaussian_envelope(n_sup)
            amp = 0.55
            env = amp * env / env.max()
            denv = np.gradient(env, P.DT_AWG)
            i_sup, q_sup = env, -(0.5 / P.ALPHA) * denv

            ui, uq = np.zeros(P.N_AWG), np.zeros(P.N_AWG)
            sl = slice(1, 1 + n_sup)
            ui[sl], uq[sl] = i_sup, q_sup
            di_des, dq_des = P.KAPPA * P.zoh(ui), P.KAPPA * P.zoh(uq)
            pi, pq = P.classical_predistort(di_des, dq_des)
            peak = float(np.abs(np.concatenate([pi, pq])).max())
            print(f"{n_sup:>6} {t_gate:>6.1f}n {peak:>16.3f} "
                  f"{'YES' if peak > U_MAX else 'no':>9}")


if __name__ == "__main__":
    main()
