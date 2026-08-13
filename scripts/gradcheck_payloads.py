# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit the JSON payloads `make verify-endpoints` feeds to check-gradients.

Both payloads deliberately pin the first and last AWG sample to zero, because
that is where the eigendecomposition propagator produced NaN gradients while its
forward pass still looked perfect. Checking anywhere else would miss it.
"""

import json
import sys

import numpy as np
from scipy.signal import bessel

FS_AWG = 2.4e9
FS_SIM = 16e9
N_AWG = 32
N_SIM = 256
DT_SIM = 1.0 / 16.0  # ns
ALPHA = -2 * np.pi * 0.300  # rad/ns


def electronics():
    rng = np.random.default_rng(0)
    ui = 0.65 * rng.normal(size=N_AWG)
    uq = 0.65 * rng.normal(size=N_AWG)
    ui[0] = ui[-1] = 0.0
    uq[0] = uq[-1] = 0.0
    sos = bessel(3, 250e6, btype="low", analog=False, output="sos",
                 fs=FS_SIM, norm="mag")
    return {
        "inputs": {
            "envelope_i": ui.tolist(),
            "envelope_q": uq.tolist(),
            "sos": sos.tolist(),
            "gain_imb": 0.020,
            "phase_imb": 0.017453,
            "lo_i": 0.003,
            "lo_q": -0.002,
            "xsat": 1.0,
            "rapp_p": 1.0,
            "kappa": 0.40,
        }
    }


def transmon():
    t_gate_ns = N_SIM * DT_SIM
    env = np.full(N_SIM, np.pi / 2 / t_gate_ns)
    env[0] = env[-1] = 0.0
    return {
        "inputs": {
            "drive_i": env.tolist(),
            "drive_q": (0.05 * env).tolist(),
            "dt": DT_SIM,
            "anharmonicity": ALPHA,
            "detunings": [0.0],
            "target_angle": float(np.pi / 2),
        }
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "electronics"
    print(json.dumps({"electronics": electronics, "transmon": transmon}[which]()))
