# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract B -- the transmon, differentiated by JAX.

This is the other side of the boundary. Where Tesseract A's derivatives are
derived on paper and written out by hand, everything here comes from JAX's
automatic differentiation of the propagator.

PHYSICS
-------
A transmon truncated to three levels, in the frame rotating at the drive
frequency and under the rotating-wave approximation. With annihilation operator
a, anharmonicity alpha, and static detuning delta:

    H(t) = delta a^dag a
         + (alpha/2) a^dag a^dag a a
         + (Omega_I(t)/2) (a + a^dag)
         + (Omega_Q(t)/2) i (a^dag - a)

The third level is not decoration: a 20 ns gate has a bandwidth comparable to
the 250 MHz anharmonicity, so population leaks to |2> and the gate cannot be
made perfect by amplitude alone. That is exactly what makes the optimisation
non-trivial, and it is why DRAG exists.

The drive is piecewise constant over each AWG sample, so the propagator over
one slice is exact:

    U_k = exp(-i H_k dt)

evaluated by eigendecomposition of the 3x3 Hermitian H_k. The gate propagator is
the time-ordered product U = U_{N-1} ... U_1 U_0.

FIGURE OF MERIT
---------------
Projecting the 3x3 propagator onto the computational subspace span{|0>, |1>}
gives a sub-unitary 2x2 block M; the norm it is missing IS the leakage. The
average gate fidelity of a trace-decreasing map on a d-dimensional subspace is

    F_avg = ( |tr(M_target^dag M)|^2 + tr(M^dag M) ) / ( d (d+1) ),   d = 2

so a single number penalises both a wrong rotation and population lost to |2>.
The reported infidelity is 1 - F_avg, averaged over a supplied ensemble of
static detunings, which forces the solution to be robust rather than exact.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)

# three-level ladder operators
_A = jnp.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=jnp.complex128)
_AD = _A.conj().T
_N = _AD @ _A
_I3 = jnp.eye(3, dtype=jnp.complex128)
_P01 = jnp.diag(jnp.array([1.0, 1.0, 0.0])).astype(jnp.complex128)


class InputSchema(BaseModel):
    drive_i: Differentiable[Array[(None,), Float64]] = Field(
        description="In-phase drive delivered to the qubit, rad/s, shape (n,)."
    )
    drive_q: Differentiable[Array[(None,), Float64]] = Field(
        description="Quadrature drive delivered to the qubit, rad/s, shape (n,)."
    )
    dt: Float64 = Field(description="Duration of one piecewise-constant slice, seconds.")
    anharmonicity: Float64 = Field(
        default=-1.5707963267948966e9,
        description="Transmon anharmonicity alpha in rad/s (2*pi*-250 MHz).",
    )
    detunings: Array[(None,), Float64] = Field(
        description="Ensemble of static detunings in rad/s to average over. "
        "A single-element array gives the nominal, non-robust objective."
    )
    target_angle: Float64 = Field(
        default=3.141592653589793,
        description="Target rotation angle about x on the computational subspace.",
    )


class OutputSchema(BaseModel):
    infidelity: Differentiable[Float64] = Field(
        description="Ensemble-averaged average-gate infidelity on span{|0>,|1>}, "
        "leakage included."
    )
    leakage: Differentiable[Float64] = Field(
        description="Ensemble-averaged population leaving the computational "
        "subspace, averaged over the |0> and |1> initial states."
    )


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------


def _target(angle):
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    return jnp.array(
        [[c, -1j * s, 0], [-1j * s, c, 0], [0, 0, 1]], dtype=jnp.complex128
    )


_TAYLOR_ORDER = 12


def _expm_taylor(m, order=_TAYLOR_ORDER):
    """exp(m) by fixed-order Taylor series.

    Deliberately NOT eigendecomposition. `jnp.linalg.eigh` is accurate forward,
    but its derivative contains 1/(w_i - w_j) terms, and at zero drive the
    Hamiltonian is the drift alone, whose |0> and |1> levels are degenerate at
    zero detuning. The VJP is then NaN -- exactly at the samples where a real
    AWG waveform must start and end. That failure is silent: the forward pass
    looks perfect. Verified on this codebase: pinning the first and last sample
    to zero produced 2 NaNs in the gradient with eigh, and none with Taylor.

    ||H dt|| <~ 0.7 here, so order 12 truncates at ~1e-13, below float64 noise
    on the fidelity. `tests/test_propagator.py` pins that error.
    """
    term = jnp.eye(m.shape[0], dtype=m.dtype)
    out = term
    for k in range(1, order + 1):
        term = term @ m / k
        out = out + term
    return out


def _propagate(di, dq, dt, alpha, delta):
    h_static = delta * _N + 0.5 * alpha * (_AD @ _AD @ _A @ _A)

    def step(u, oiq):
        oi, oq = oiq
        hk = h_static + 0.5 * oi * (_A + _AD) + 0.5 * oq * 1j * (_AD - _A)
        uk = _expm_taylor(-1j * hk * dt)
        return uk @ u, None

    u, _ = jax.lax.scan(step, _I3, jnp.stack([di, dq], axis=1))
    return u


def _metrics_one(di, dq, dt, alpha, delta, angle):
    u = _propagate(di, dq, dt, alpha, delta)
    m = _P01 @ u @ _P01
    mt = _P01 @ _target(angle) @ _P01
    d = 2.0
    ov = jnp.abs(jnp.trace(mt.conj().T @ m)) ** 2
    tr_mm = jnp.real(jnp.trace(m.conj().T @ m))
    infid = 1.0 - (ov + tr_mm) / (d * (d + 1.0))
    # population leaving {|0>,|1>}, averaged over the two computational inputs
    leak = 1.0 - tr_mm / d
    return infid, leak


def _metrics(di, dq, dt, alpha, detunings, angle):
    infid, leak = jax.vmap(
        lambda de: _metrics_one(di, dq, dt, alpha, de, angle)
    )(detunings)
    return jnp.mean(infid), jnp.mean(leak)


def _as_arrays(inputs: InputSchema):
    return (
        jnp.asarray(inputs.drive_i, dtype=jnp.float64),
        jnp.asarray(inputs.drive_q, dtype=jnp.float64),
        float(inputs.dt),
        float(inputs.anharmonicity),
        jnp.asarray(inputs.detunings, dtype=jnp.float64),
        float(inputs.target_angle),
    )


# --------------------------------------------------------------------------
# required endpoints
# --------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    di, dq, dt, alpha, det, ang = _as_arrays(inputs)
    infid, leak = _metrics(di, dq, dt, alpha, det, ang)
    return OutputSchema(infidelity=float(infid), leakage=float(leak))


def abstract_eval(abstract_inputs):
    del abstract_inputs  # both outputs are always scalars
    return {
        "infidelity": ShapeDType(shape=(), dtype="float64"),
        "leakage": ShapeDType(shape=(), dtype="float64"),
    }


# --------------------------------------------------------------------------
# gradient endpoints -- supplied by JAX autodiff
# --------------------------------------------------------------------------

_OUTPUTS = ("infidelity", "leakage")


def _fn(di, dq, dt, alpha, det, ang):
    infid, leak = _metrics(di, dq, dt, alpha, det, ang)
    return {"infidelity": infid, "leakage": leak}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    di, dq, dt, alpha, det, ang = _as_arrays(inputs)
    n = di.shape[0]
    ti = jnp.asarray(tangent_vector.get("drive_i", np.zeros(n)), dtype=jnp.float64)
    tq = jnp.asarray(tangent_vector.get("drive_q", np.zeros(n)), dtype=jnp.float64)
    if "drive_i" not in jvp_inputs:
        ti = jnp.zeros(n)
    if "drive_q" not in jvp_inputs:
        tq = jnp.zeros(n)

    _, out_tan = jax.jvp(
        lambda a, b: _fn(a, b, dt, alpha, det, ang), (di, dq), (ti, tq)
    )
    return {k: float(out_tan[k]) for k in _OUTPUTS if k in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    di, dq, dt, alpha, det, ang = _as_arrays(inputs)
    _, vjp_fn = jax.vjp(lambda a, b: _fn(a, b, dt, alpha, det, ang), di, dq)

    ct = {
        k: jnp.asarray(cotangent_vector.get(k, 0.0), dtype=jnp.float64)
        if k in vjp_outputs
        else jnp.asarray(0.0)
        for k in _OUTPUTS
    }
    gi, gq = vjp_fn(ct)

    result = {}
    if "drive_i" in vjp_inputs:
        result["drive_i"] = np.asarray(gi)
    if "drive_q" in vjp_inputs:
        result["drive_q"] = np.asarray(gq)
    return result


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    di, dq, dt, alpha, det, ang = _as_arrays(inputs)
    jac = jax.jacrev(lambda a, b: _fn(a, b, dt, alpha, det, ang), argnums=(0, 1))(di, dq)
    # jacrev with a dict output and argnums=(0, 1) returns
    #   {output_name: (d/d drive_i, d/d drive_q)}
    names = ("drive_i", "drive_q")
    return {
        o: {
            nm: np.asarray(jac[o][i])
            for i, nm in enumerate(names)
            if nm in jac_inputs
        }
        for o in jac_outputs
    }
