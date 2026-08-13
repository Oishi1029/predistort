# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that gradients cross the two-container boundary correctly.

    theta  --(u = u_max tanh, support mask)-->  DAC codes
                    |
                    v
        [ electronics ]   Julia core, hand-derived analytic adjoint, no AD framework
                    |  drive_i, drive_q at 16 GSa/s
                    v
        [ transmon ]      JAX autodiff, Taylor propagator
                    |  virtual-Z-optimal gate infidelity
                    v
                  loss

One ``jax.grad`` of ``loss`` runs back through the JAX propagator, across an HTTP
boundary, and into a Julia analytic adjoint. We check it against central finite
differences of the SAME composed forward pass, so the check exercises the whole
chain rather than either half.

Finite differences here are honest but expensive: each directional derivative
costs two full pipeline evaluations, so we check a random subset of coordinates
plus one random direction rather than all 64.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

# ---- grids, matching tesseracts/electronics (2 GSa/s AWG, x8 -> 16 GSa/s sim)
N_AWG = 32
UPSAMPLE = 8
N_SIM = N_AWG * UPSAMPLE
DT_SIM = 1.0 / 16.0  # ns
SUPPORT = slice(1, 25)  # 24 samples = 12 ns of drive inside a 16 ns window
U_MAX = 0.90
ALPHA = -2 * np.pi * 0.300  # rad/ns
TARGET_ANGLE = np.pi / 2  # X90

OUTDIR = Path.home() / "dev" / "tesseract-hack" / "_outputs"


def codes_from_theta(theta):
    """Unconstrained theta -> DAC codes obeying the box and the support window.

    The tanh enforces |u| <= U_MAX structurally rather than as a constraint the
    optimiser can sit against, and the mask pins the first and last samples to
    exactly zero because a real AWG waveform must start and end at zero.
    """
    u = U_MAX * jnp.tanh(theta)
    mask = jnp.zeros(N_AWG).at[SUPPORT].set(1.0)
    return u[:N_AWG] * mask, u[N_AWG:] * mask


def make_loss(t_elec, t_qubit):
    def loss(theta):
        ui, uq = codes_from_theta(theta)
        drive = apply_tesseract(t_elec, {"envelope_i": ui, "envelope_q": uq})
        out = apply_tesseract(
            t_qubit,
            {
                "drive_i": drive["drive_i"],
                "drive_q": drive["drive_q"],
                "dt": jnp.float64(DT_SIM),
                "anharmonicity": jnp.float64(ALPHA),
                "detunings": jnp.zeros(1),
                "target_angle": jnp.float64(TARGET_ANGLE),
            },
        )
        return out["infidelity"]

    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-coords", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    theta = jnp.asarray(0.5 * rng.normal(size=2 * N_AWG))

    with (
        Tesseract.from_image("electronics:latest", output_path=OUTDIR) as t_elec,
        Tesseract.from_image("transmon:latest", output_path=OUTDIR) as t_qubit,
    ):
        loss = make_loss(t_elec, t_qubit)

        t0 = time.perf_counter()
        l0 = float(loss(theta))
        t_fwd = time.perf_counter() - t0
        print(f"composed forward: infidelity = {l0:.8e}   ({t_fwd:.2f} s)")

        t0 = time.perf_counter()
        g = np.asarray(jax.grad(loss)(theta), dtype=np.float64)
        t_grad = time.perf_counter() - t0
        print(f"jax.grad across both containers: {t_grad:.2f} s")
        print(f"  |grad|_inf = {np.abs(g).max():.6e}   NaNs = {int(np.isnan(g).sum())}")

        # ---- coordinate-wise central differences on a random subset ----
        idx = rng.choice(2 * N_AWG, size=args.n_coords, replace=False)
        eps = args.eps
        rows = []
        for k in idx:
            tp = theta.at[k].add(eps)
            tm = theta.at[k].add(-eps)
            fd = (float(loss(tp)) - float(loss(tm))) / (2 * eps)
            den = max(abs(fd) + abs(g[k]), 1e-14)
            rows.append((int(k), g[k], fd, abs(g[k] - fd) / den))

        print(f"\n  {'coord':>6} {'jax.grad':>16} {'central diff':>16} {'rel err':>11}")
        for k, ad, fd, rel in rows:
            print(f"  {k:>6} {ad:>16.8e} {fd:>16.8e} {rel:>11.2e}")
        worst_coord = max(r[3] for r in rows)

        # ---- one random direction, which mixes every coordinate ----
        v = rng.normal(size=2 * N_AWG)
        v /= np.linalg.norm(v)
        fd_dir = (
            float(loss(theta + eps * v)) - float(loss(theta - eps * v))
        ) / (2 * eps)
        ad_dir = float(g @ v)
        rel_dir = abs(ad_dir - fd_dir) / max(abs(ad_dir) + abs(fd_dir), 1e-14)
        print(f"\n  random direction: analytic {ad_dir:.8e}  fd {fd_dir:.8e}  "
              f"rel {rel_dir:.2e}")

        tol = 1e-5
        ok = worst_coord < tol and rel_dir < tol
        print(f"\n{'PASS' if ok else 'FAIL'}: worst relative error {max(worst_coord, rel_dir):.2e} "
              f"< {tol:.0e} across the Julia/JAX container boundary")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
