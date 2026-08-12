# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract A -- the control-electronics chain, with a HAND-DERIVED adjoint.

This container deliberately contains no automatic-differentiation framework.
Its only numerical dependency is NumPy. Every derivative below was derived on
paper and coded directly, which is what puts a real differentiation-strategy
boundary between this Tesseract and the JAX transmon Tesseract downstream.

THE FORWARD CHAIN
-----------------
A baseband drive (I, Q) leaves the AWG and arrives at the qubit having passed
through four stages:

  1. IQ mixer imbalance -- a static real 2x2 map. With gain error g and
     quadrature phase error phi:

         I1 = I - (1+g) sin(phi) Q
         Q1 =     (1+g) cos(phi) Q

  2. AWG + line finite bandwidth -- a causal FIR:

         y[n] = sum_k h[k] x[n-k]

  3. Output-amplifier compression -- a smooth saturating nonlinearity:

         y[n] = A tanh(x[n] / A)

  4. Bias-tee / coupling-capacitor droop -- a causal single-pole highpass:

         y[n] = a (y[n-1] + x[n] - x[n-1]),   y[-1] = x[-1] = 0

Stage 3 is what stops this chain from being a single fixed matrix. Without it a
reviewer could precompute the Jacobian once and fold it into JAX as a matmul,
and this container would be decoration. With it, the Jacobian depends on the
input and no fixed matrix exists.

THE ADJOINTS, DERIVED
---------------------
Stage 1 is linear, so its adjoint is the transpose of its matrix:

    Ibar = Ibar'
    Qbar = -(1+g) sin(phi) Ibar' + (1+g) cos(phi) Qbar'

Stage 2. Differentiating y[n] = sum_k h[k] x[n-k] with respect to x[m] gives
h[n-m], so

    dL/dx[m] = sum_n ybar[n] h[n-m]

The adjoint of a causal convolution with h is therefore an ANTI-CAUSAL
correlation with the same taps -- it looks forward in time exactly as far as the
forward pass looked back.

Stage 3 is elementwise, so its adjoint is a diagonal scaling by the local slope

    d/dx [A tanh(x/A)] = sech^2(x/A) = 1 - tanh^2(x/A)

evaluated at the stage-3 INPUT. This is the one stage that needs the forward
pass to stash an intermediate value; everything else is stateless.

Stage 4. Each x[n] enters y[n] with coefficient +a and y[n+1] with coefficient
-a, and the recursion carries y[n-1] forward with coefficient a. Introducing an
adjoint state lambda that runs BACKWARDS in time,

    lambda[N-1] = ybar[N-1]
    lambda[n]   = ybar[n] + a lambda[n+1]
    xbar[n]     = a lambda[n] - a lambda[n+1]        (lambda[N] := 0)

i.e. the adjoint of a causal recursion is the same recursion run in reverse
time. This is the classical adjoint-system result, and it is exact -- not a
finite-difference approximation of it.

Forward-mode (JVP) is simpler: stages 1, 2 and 4 are linear, so a tangent passes
through them unchanged in form, and stage 3 multiplies the tangent by the same
sech^2 factor.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class InputSchema(BaseModel):
    envelope_i: Differentiable[Array[(None,), Float64]] = Field(
        description="In-phase baseband envelope at the AWG output, shape (n,), "
        "in units of the drive amplitude scale."
    )
    envelope_q: Differentiable[Array[(None,), Float64]] = Field(
        description="Quadrature baseband envelope at the AWG output, shape (n,)."
    )
    fir_taps: Array[(None,), Float64] = Field(
        description="Causal FIR taps modelling the AWG and line bandwidth. "
        "Measured per instrument; not a fitted parameter here."
    )
    droop_alpha: Float64 = Field(
        default=0.999861,
        description="Bias-tee pole, a = exp(-dt/tau). 0 < a < 1; a -> 1 is no droop.",
    )
    sat_amplitude: Float64 = Field(
        default=3.0,
        description="Amplifier compression scale A in y = A tanh(x/A). "
        "Larger A is more linear.",
    )
    gain_imbalance: Float64 = Field(
        default=0.03, description="IQ mixer relative gain error g on the Q arm."
    )
    phase_error_rad: Float64 = Field(
        default=0.05236, description="IQ mixer quadrature phase error, radians."
    )


class OutputSchema(BaseModel):
    drive_i: Differentiable[Array[(None,), Float64]] = Field(
        description="In-phase drive actually delivered to the qubit, shape (n,)."
    )
    drive_q: Differentiable[Array[(None,), Float64]] = Field(
        description="Quadrature drive actually delivered to the qubit, shape (n,)."
    )


# --------------------------------------------------------------------------
# forward stages
# --------------------------------------------------------------------------


def _mixer(i, q, g, phi):
    gg = 1.0 + g
    return i - gg * np.sin(phi) * q, gg * np.cos(phi) * q


def _mixer_adj(bi, bq, g, phi):
    gg = 1.0 + g
    return bi, -gg * np.sin(phi) * bi + gg * np.cos(phi) * bq


def _fir(x, h):
    return np.convolve(x, h)[: len(x)]


def _fir_adj(ybar, h, n):
    padded = np.concatenate([ybar, np.zeros(len(h) - 1, dtype=ybar.dtype)])
    return np.correlate(padded, h, mode="valid")[:n]


def _compress(x, a_sat):
    return a_sat * np.tanh(x / a_sat)


def _compress_slope(x_in, a_sat):
    return 1.0 - np.tanh(x_in / a_sat) ** 2


def _droop(x, a):
    y = np.empty_like(x)
    y_prev = 0.0
    x_prev = 0.0
    for n in range(x.shape[0]):
        y_prev = a * (y_prev + x[n] - x_prev)
        x_prev = x[n]
        y[n] = y_prev
    return y


def _droop_adj(ybar, a):
    n = ybar.shape[0]
    lam = np.zeros(n + 1, dtype=ybar.dtype)
    for k in range(n - 1, -1, -1):
        lam[k] = ybar[k] + a * lam[k + 1]
    return a * lam[:n] - a * lam[1:]


def _forward(inputs: InputSchema, keep_tape: bool = False):
    i = np.ascontiguousarray(inputs.envelope_i, dtype=np.float64)
    q = np.ascontiguousarray(inputs.envelope_q, dtype=np.float64)
    h = np.ascontiguousarray(inputs.fir_taps, dtype=np.float64)

    i1, q1 = _mixer(i, q, inputs.gain_imbalance, inputs.phase_error_rad)
    i2, q2 = _fir(i1, h), _fir(q1, h)  # stage-3 input: the tape
    i3, q3 = _compress(i2, inputs.sat_amplitude), _compress(q2, inputs.sat_amplitude)
    out_i = _droop(i3, inputs.droop_alpha)
    out_q = _droop(q3, inputs.droop_alpha)

    if keep_tape:
        return (out_i, out_q), (i2, q2)
    return out_i, out_q


# --------------------------------------------------------------------------
# required endpoints
# --------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Push the commanded envelopes through the electronics chain."""
    out_i, out_q = _forward(inputs)
    return OutputSchema(drive_i=out_i, drive_q=out_q)


def abstract_eval(abstract_inputs):
    """Shapes out equal shapes in; the chain is length-preserving."""
    shp = abstract_inputs.envelope_i.shape
    return {
        "drive_i": ShapeDType(shape=shp, dtype="float64"),
        "drive_q": ShapeDType(shape=shp, dtype="float64"),
    }


# --------------------------------------------------------------------------
# gradient endpoints -- hand-derived, no AD framework
# --------------------------------------------------------------------------


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Forward mode. Linear stages pass the tangent through unchanged in form;
    the compression stage scales it by the local slope."""
    _, (i2, q2) = _forward(inputs, keep_tape=True)
    n = np.asarray(inputs.envelope_i).shape[0]
    h = np.ascontiguousarray(inputs.fir_taps, dtype=np.float64)

    ti = np.ascontiguousarray(
        tangent_vector.get("envelope_i", np.zeros(n)), dtype=np.float64
    )
    tq = np.ascontiguousarray(
        tangent_vector.get("envelope_q", np.zeros(n)), dtype=np.float64
    )
    if "envelope_i" not in jvp_inputs:
        ti = np.zeros(n)
    if "envelope_q" not in jvp_inputs:
        tq = np.zeros(n)

    ti, tq = _mixer(ti, tq, inputs.gain_imbalance, inputs.phase_error_rad)
    ti, tq = _fir(ti, h), _fir(tq, h)
    ti = ti * _compress_slope(i2, inputs.sat_amplitude)
    tq = tq * _compress_slope(q2, inputs.sat_amplitude)
    ti, tq = _droop(ti, inputs.droop_alpha), _droop(tq, inputs.droop_alpha)

    result = {}
    if "drive_i" in jvp_outputs:
        result["drive_i"] = ti
    if "drive_q" in jvp_outputs:
        result["drive_q"] = tq
    return result


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Reverse mode: walk the four stages backwards, adjointing each."""
    _, (i2, q2) = _forward(inputs, keep_tape=True)
    n = np.asarray(inputs.envelope_i).shape[0]
    h = np.ascontiguousarray(inputs.fir_taps, dtype=np.float64)

    bi = np.ascontiguousarray(
        cotangent_vector.get("drive_i", np.zeros(n)), dtype=np.float64
    )
    bq = np.ascontiguousarray(
        cotangent_vector.get("drive_q", np.zeros(n)), dtype=np.float64
    )
    if "drive_i" not in vjp_outputs:
        bi = np.zeros(n)
    if "drive_q" not in vjp_outputs:
        bq = np.zeros(n)

    bi, bq = _droop_adj(bi, inputs.droop_alpha), _droop_adj(bq, inputs.droop_alpha)
    bi = bi * _compress_slope(i2, inputs.sat_amplitude)
    bq = bq * _compress_slope(q2, inputs.sat_amplitude)
    bi, bq = _fir_adj(bi, h, n), _fir_adj(bq, h, n)
    bi, bq = _mixer_adj(bi, bq, inputs.gain_imbalance, inputs.phase_error_rad)

    result = {}
    if "envelope_i" in vjp_inputs:
        result["envelope_i"] = bi
    if "envelope_q" in vjp_inputs:
        result["envelope_q"] = bq
    return result


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Dense Jacobian, assembled column by column from the JVP.

    Only used for verification and for the figure showing the chain's Jacobian
    structure; the optimisation loop never calls this.
    """
    n = np.asarray(inputs.envelope_i).shape[0]
    cols = {inp: np.zeros((n, n)) for inp in jac_inputs}
    out = {o: {} for o in jac_outputs}

    for inp in jac_inputs:
        block = {o: np.zeros((n, n)) for o in jac_outputs}
        for k in range(n):
            e = {"envelope_i": np.zeros(n), "envelope_q": np.zeros(n)}
            e[inp] = np.zeros(n)
            e[inp][k] = 1.0
            jv = jacobian_vector_product(inputs, {inp}, jac_outputs, e)
            for o in jac_outputs:
                block[o][:, k] = jv[o]
        for o in jac_outputs:
            out[o][inp] = block[o]
        del cols[inp]

    return out
