# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Textbook pulses and the classical pre-distortion a lab would actually apply.

This module is the BASELINE's toolbox, and it is deliberately good. An entry
that beats a weak baseline has proved nothing, so the comparison arm here gets
every correction a competent RF/control engineer applies before reaching for
gradients:

  * DRAG, the standard leakage-suppressing pulse for a weakly anharmonic qubit;
  * exact inversion of the IQ mixer, which is affine and therefore exactly
    invertible -- LO-leakage nulling and imbalance pre-correction are table
    stakes in every lab;
  * memoryless AM/AM pre-distortion, the standard fix for amplifier compression,
    which for the Rapp model has a closed-form inverse;
  * Tikhonov-regularised inversion of the measured LTI line response, i.e.
    linear pre-emphasis, which is what a VNA measurement buys you;
  * recalibration of amplitude, drive phase and the DRAG coefficient, fitted
    through the true chain -- a lab's Rabi and DRAG calibration.

What that stack CANNOT do is the reason this project exists. The compressor sits
*after* the line filter, so inverting the nonlinearity *before* the filter does
not commute with it. The residue is the classic memory effect, and removing it
requires a gradient through the composed, ordered chain rather than a cascade of
independent inverses.
"""

import numpy as np
from scipy.signal import sosfilt

# ---- grids, matching tesseracts/electronics -------------------------------
N_AWG = 32
UPSAMPLE = 8
N_SIM = N_AWG * UPSAMPLE
DT_AWG = 0.5  # ns, 2 GSa/s
DT_SIM = DT_AWG / UPSAMPLE  # ns, 16 GSa/s
SUPPORT = slice(1, 25)  # 24 samples = 12 ns of drive
N_SUPPORT = 24

SOS_CAL_A = np.array(
    [
        [0.00027851513734080725, 0.0005570302746816145,
         0.00027851513734080725, 1.0, -0.8779714873192201, 0.0],
        [1.0, 1.0, 0.0, 1.0, -1.7959744380069622, 0.8142334583766941],
    ]
)

# instrument parameters, mirroring the container defaults
GAIN_IMB = 0.020
PHASE_IMB = 0.017453
LO_I, LO_Q = 0.003, -0.002
XSAT = 1.0
RAPP_P = 1.0
KAPPA = 0.40  # rad/ns per DAC unit
ALPHA = -2 * np.pi * 0.300  # rad/ns


# ---------------------------------------------------------------- pulses
def gaussian_envelope(n_support=N_SUPPORT, sigma_frac=0.25):
    """Truncated, baseline-lifted Gaussian that starts and ends at zero."""
    t = (np.arange(n_support) + 0.5) / n_support - 0.5
    g = np.exp(-0.5 * (t / sigma_frac) ** 2)
    g = g - np.exp(-0.5 / (2 * sigma_frac) ** 2 * 1.0)
    return np.clip(g, 0.0, None)


def drag_pair(amp, beta, phase=0.0, n_support=N_SUPPORT, sigma_frac=0.25):
    """DRAG: Gaussian in-phase, derivative-shaped quadrature.

    Q(t) = -(beta / alpha) dI/dt, which cancels the leading leakage term. Note
    the whole mechanism is a derivative, i.e. exactly the high-frequency content
    a bandwidth-limited line destroys -- which is why the textbook answer
    degrades here.
    """
    env = gaussian_envelope(n_support, sigma_frac)
    env = amp * env / env.max()
    denv = np.gradient(env, DT_AWG)
    i_sup = env
    q_sup = -(beta / ALPHA) * denv
    c, s = np.cos(phase), np.sin(phase)
    return c * i_sup - s * q_sup, s * i_sup + c * q_sup


def place(i_sup, q_sup):
    """Drop support-length arrays into the full 32-sample AWG window."""
    ui, uq = np.zeros(N_AWG), np.zeros(N_AWG)
    ui[SUPPORT] = i_sup
    uq[SUPPORT] = q_sup
    return ui, uq


# ------------------------------------------- the linear part of the chain
def zoh(u, k=UPSAMPLE):
    return np.repeat(u, k)


def lti_forward(u, sos=SOS_CAL_A):
    """ZOH then the measured line response. The part a VNA gives you."""
    return sosfilt(sos, zoh(u))


def lti_matrix(sos=SOS_CAL_A):
    """Dense (N_SIM x N_AWG) matrix of the linear stage, by unit impulses."""
    m = np.zeros((N_SIM, N_AWG))
    for k in range(N_AWG):
        e = np.zeros(N_AWG)
        e[k] = 1.0
        m[:, k] = lti_forward(e, sos)
    return m


def tikhonov_inverse(target_sim, lam=1e-6, sos=SOS_CAL_A):
    """Linear pre-emphasis: argmin ||L u - target||^2 + lam ||u||^2."""
    ell = lti_matrix(sos)
    a = ell.T @ ell + lam * np.eye(N_AWG)
    return np.linalg.solve(a, ell.T @ target_sim)


# ------------------------------------------------ exact classical inverses
def mixer_inverse(i, q, gain_imb=GAIN_IMB, phase_imb=PHASE_IMB,
                  lo_i=LO_I, lo_q=LO_Q):
    """Undo the affine mixer exactly: subtract LO leakage, apply M^{-1}."""
    g = 1.0 + gain_imb
    s, c = np.sin(phase_imb), np.cos(phase_imb)
    yi, yq = i - lo_i, q - lo_q
    # M = [[1, -g s], [0, g c]]  ->  M^{-1} = [[1, s/c], [0, 1/(g c)]]
    return yi + (s / c) * yq, yq / (g * c)


def rapp_inverse(i, q, xsat=XSAT, p=RAPP_P):
    """Closed-form inverse of the memoryless Rapp compressor, for p = 1.

    Forward: |y| = r / (1 + (r/xsat)^2)^(1/2) with r = |x|.
    Solving for r with w = |y|/xsat gives r = xsat w / sqrt(1 - w^2), which
    exists only for |y| < xsat. Above that the amplifier is saturated and no
    pre-distortion can recover the commanded amplitude -- a real limit, not a
    numerical one, so we clamp and let the baseline suffer it honestly.
    """
    if p != 1.0:
        raise NotImplementedError("closed-form inverse implemented for p = 1")
    mag = np.hypot(i, q)
    w = np.clip(mag / xsat, 0.0, 0.999999)
    r = xsat * w / np.sqrt(1.0 - w**2)
    scale = np.where(mag > 1e-15, r / np.maximum(mag, 1e-15), 1.0)
    return i * scale, q * scale


def classical_predistort(i_des_sim, q_des_sim, lam=1e-6, sos=SOS_CAL_A, xsat=XSAT):
    """The full classical stack, applied in the only order it can be applied.

    Walk the model backwards: undo the drive scale, undo the compressor
    pointwise, undo the mixer exactly, then linearly invert the line response.

    The order is forced and it is *wrong* in one specific way: the true chain
    compresses AFTER filtering, so a pre-inverse of the compressor applied
    BEFORE the filter cannot cancel it. That residual is the memory effect this
    project removes with a gradient.
    """
    a_i, a_q = i_des_sim / KAPPA, q_des_sim / KAPPA
    b_i, b_q = rapp_inverse(a_i, a_q, xsat=xsat)
    c_i, c_q = mixer_inverse(b_i, b_q)
    return tikhonov_inverse(c_i, lam, sos), tikhonov_inverse(c_q, lam, sos)
