# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract A -- the control-electronics line, with a Julia numerical core.

This file is a thin bridge. All physics and all derivatives live in
`julia/src/LineChain.jl`, and every one of those derivatives was derived on paper
rather than obtained from an autodiff framework.

WHY THE CORE IS JULIA, AND WHY THAT MATTERS
-------------------------------------------
The obvious objection to a two-container differentiable pipeline is that
`jax.custom_vjp` exists: write the forward pass, attach a hand-written backward
pass, keep everything in one process, and delete the second container. That
objection is correct whenever the component is Python.

It is not correct here. The numerical core is a Julia module. To collapse this
pipeline into one process you would have to embed a Julia runtime inside the JAX
process -- two garbage collectors, two package managers, two threading models,
and a multi-second interpreter start on every process launch. Nobody ships that.
Keeping the instrument model behind a served interface is the ordinary answer,
and Tesseract is what makes that interface differentiable.

The measured line response is an INPUT, of variable section count, so one image
can be swept across instruments. What the image pins is the *code* -- the Julia
implementation and its hand-derived adjoint -- not one particular calibration of
one particular fridge. A default calibration is supplied for reproducibility.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# --------------------------------------------------------------------------
# Julia bridge. Importing juliacall boots a Julia runtime in-process; the
# module is precompiled into an image layer at build time so this is a load,
# not a compile.
# --------------------------------------------------------------------------
from juliacall import Main as jl  # noqa: E402

jl.seval("using LineChain")

# Default calibration: 3rd-order Bessel low-pass, 250 MHz, at 16 GSa/s.
_SOS_DEFAULT = np.array(
    [
        [0.00027851513734080725, 0.0005570302746816145,
         0.00027851513734080725, 1.0, -0.8779714873192201, 0.0],
        [1.0, 1.0, 0.0, 1.0, -1.7959744380069622, 0.8142334583766941],
    ]
)

UPSAMPLE = 8


class InputSchema(BaseModel):
    envelope_i: Differentiable[Array[(None,), Float64]] = Field(
        description="Commanded in-phase DAC codes at the AWG rate, shape (n,)."
    )
    envelope_q: Differentiable[Array[(None,), Float64]] = Field(
        description="Commanded quadrature DAC codes at the AWG rate, shape (n,)."
    )
    sos: Array[(None, 6), Float64] = Field(
        default=_SOS_DEFAULT,
        description="Measured line response as second-order sections, shape (K, 6) "
        "with rows [b0 b1 b2 a0 a1 a2], at the 16 GSa/s simulation rate. Variable "
        "K, so one image covers instruments with different section counts.",
    )
    gain_imb: Float64 = Field(default=0.020, description="IQ gain imbalance on Q.")
    phase_imb: Float64 = Field(default=0.017453, description="Quadrature skew, rad.")
    lo_i: Float64 = Field(default=0.003, description="LO leakage on I, DAC units.")
    lo_q: Float64 = Field(default=-0.002, description="LO leakage on Q, DAC units.")
    xsat: Float64 = Field(default=1.0, description="Rapp saturation level, DAC units.")
    rapp_p: Float64 = Field(default=1.0, description="Rapp smoothness exponent.")
    kappa: Float64 = Field(default=0.40, description="Drive scale, rad/ns per DAC unit.")


class OutputSchema(BaseModel):
    drive_i: Differentiable[Array[(None,), Float64]] = Field(
        description="In-phase drive delivered to the qubit at the simulation rate, "
        "shape (8n,), rad/ns."
    )
    drive_q: Differentiable[Array[(None,), Float64]] = Field(
        description="Quadrature drive delivered to the qubit, shape (8n,), rad/ns."
    )


def _params(inputs: InputSchema):
    return jl.LineChain.ChainParams(
        upsample=UPSAMPLE,
        sos=np.ascontiguousarray(inputs.sos, dtype=np.float64),
        gain_imb=float(inputs.gain_imb),
        phase_imb=float(inputs.phase_imb),
        lo_i=float(inputs.lo_i),
        lo_q=float(inputs.lo_q),
        xsat=float(inputs.xsat),
        rapp_p=float(inputs.rapp_p),
        kappa=float(inputs.kappa),
    )


def _vec(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def _forward(inputs: InputSchema):
    out, tape = jl.LineChain.forward_with_tape(
        _vec(inputs.envelope_i), _vec(inputs.envelope_q), _params(inputs)
    )
    return out, tape


# --------------------------------------------------------------------------
# required endpoints
# --------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    out, _ = _forward(inputs)
    return OutputSchema(drive_i=np.asarray(out[0]), drive_q=np.asarray(out[1]))


def abstract_eval(abstract_inputs):
    (n,) = abstract_inputs.envelope_i.shape
    return {
        "drive_i": ShapeDType(shape=(n * UPSAMPLE,), dtype="float64"),
        "drive_q": ShapeDType(shape=(n * UPSAMPLE,), dtype="float64"),
    }


# --------------------------------------------------------------------------
# gradient endpoints -- hand-derived in Julia, no AD framework anywhere
# --------------------------------------------------------------------------


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    _, tape = _forward(inputs)
    n = np.asarray(inputs.envelope_i).shape[0]

    ti = _vec(tangent_vector.get("envelope_i", np.zeros(n)))
    tq = _vec(tangent_vector.get("envelope_q", np.zeros(n)))
    if "envelope_i" not in jvp_inputs:
        ti = np.zeros(n)
    if "envelope_q" not in jvp_inputs:
        tq = np.zeros(n)

    oi, oq = jl.LineChain.chain_jvp(ti, tq, tape, _params(inputs))

    result = {}
    if "drive_i" in jvp_outputs:
        result["drive_i"] = np.asarray(oi)
    if "drive_q" in jvp_outputs:
        result["drive_q"] = np.asarray(oq)
    return result


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    _, tape = _forward(inputs)
    n = np.asarray(inputs.envelope_i).shape[0]
    m = n * UPSAMPLE

    bi = _vec(cotangent_vector.get("drive_i", np.zeros(m)))
    bq = _vec(cotangent_vector.get("drive_q", np.zeros(m)))
    if "drive_i" not in vjp_outputs:
        bi = np.zeros(m)
    if "drive_q" not in vjp_outputs:
        bq = np.zeros(m)

    gi, gq = jl.LineChain.chain_vjp(bi, bq, tape, _params(inputs))

    result = {}
    if "envelope_i" in vjp_inputs:
        result["envelope_i"] = np.asarray(gi)
    if "envelope_q" in vjp_inputs:
        result["envelope_q"] = np.asarray(gq)
    return result


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Dense Jacobian, assembled column by column from the Julia JVP.

    Verification and figures only; the optimisation loop never calls this.
    """
    n = np.asarray(inputs.envelope_i).shape[0]
    m = n * UPSAMPLE
    out = {o: {} for o in jac_outputs}

    for inp in jac_inputs:
        blocks = {o: np.zeros((m, n)) for o in jac_outputs}
        for k in range(n):
            tang = {"envelope_i": np.zeros(n), "envelope_q": np.zeros(n)}
            tang[inp] = np.zeros(n)
            tang[inp][k] = 1.0
            jv = jacobian_vector_product(inputs, {inp}, jac_outputs, tang)
            for o in jac_outputs:
                blocks[o][:, k] = jv[o]
        for o in jac_outputs:
            out[o][inp] = blocks[o]

    return out
