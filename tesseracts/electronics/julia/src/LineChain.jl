# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
    LineChain

The control-electronics forward model and its **hand-derived analytic adjoint**.

There is no automatic-differentiation package in this module's dependency tree —
only `PythonCall` for the bridge and Julia's standard library. Every derivative
below was derived on paper. That is the point: this component's gradient cannot
be obtained by wrapping it in `jax.custom_vjp`, because it is not Python.

## The chain

Commanded DAC codes `u_i, u_q` (length `Nawg`) become the drive actually seen by
the qubit (length `Nsim = U * Nawg`) through five stages:

  S1  zero-order hold, factor `U`        what a DAC physically does
  S2  baseband line response             causal biquad (SOS) cascade
  S3  IQ mixer                           gain imbalance, quadrature skew, LO leakage
  S4  amplifier compression              Rapp model on the complex envelope MAGNITUDE
  S5  drive scale                        DAC units -> rad/s

## The adjoints

**S1.** `y[m] = u[div(m-1, U) + 1]`. Each input sample fans out to `U` outputs, so
the adjoint sums the cotangents within each block:
`ū[k] = Σ_{m in block k} ȳ[m]`.

**S2.** A causal LTI filter with zero initial state is a lower-triangular Toeplitz
matrix `H` whose first column is the impulse response. Its adjoint is `Hᵀ`, and

    (Hᵀ v)[n] = Σ_{m ≥ n} h[m-n] v[m] = reverse( H ( reverse(v) ) )[n]

so **the adjoint of a causal filter is the same filter run on the time-reversed
signal, reversed back**. A cascade of biquads is itself LTI, so this applies to
the whole cascade at once — no per-section adjoint bookkeeping is needed. This is
exact, not an approximation of an adjoint.

**S3.** Affine: `[I;Q] ↦ M [I;Q] + c`. Adjoint is `Mᵀ`; the LO-leakage offset `c`
is constant in the inputs and so contributes nothing to the input gradient.

**S4.** The Rapp compressor acts on the magnitude `r = √(I² + Q²)`:

    g(r) = (1 + (r/x_sat)^{2p})^{-1/(2p)},    (I', Q') = g(r) · (I, Q)

This does **not** have a diagonal Jacobian — the gain on I depends on Q. Writing
`h(r) ≡ g'(r)/r`, the per-sample Jacobian is the symmetric matrix

    ∂(I',Q')/∂(I,Q) = g(r) · Id + h(r) · [I²  IQ; IQ  Q²]

with

    g'(r) = -(1 + s)^{-1/(2p) - 1} · r^{2p-1} / x_sat^{2p},   s = (r/x_sat)^{2p}
    h(r)  = -(1 + s)^{-1/(2p) - 1} · r^{2p-2} / x_sat^{2p}

`h` is written directly rather than as `g'/r` so that `r = 0` is not a removable
singularity evaluated numerically; for `p = 1` it tends to `-1/x_sat²`. Because
the Jacobian is symmetric, forward and reverse mode share the same expression.

**S5.** Scalar multiply; adjoint is the same scalar.

Stage S4 is the only stage that needs the forward pass to retain an intermediate
value (its own input), and it is also the stage that makes the whole chain
nonlinear — which is what stops a reviewer from precomputing one Jacobian matrix
and folding this container into JAX as a matmul.
"""
module LineChain

export ChainParams, forward, forward_with_tape, chain_vjp, chain_jvp

"""Instrument parameters. These describe one physical control line."""
Base.@kwdef struct ChainParams
    upsample::Int = 8
    sos::Matrix{Float64}          # K x 6, rows [b0 b1 b2 a0 a1 a2]
    gain_imb::Float64 = 0.02      # relative gain error on the Q arm
    phase_imb::Float64 = 0.017453 # quadrature skew, radians
    lo_i::Float64 = 0.003         # LO leakage, DAC units
    lo_q::Float64 = -0.002
    xsat::Float64 = 1.0           # Rapp saturation level, DAC units
    rapp_p::Float64 = 1.0         # Rapp smoothness
    kappa::Float64 = 0.40         # rad/ns per DAC unit
end

# ---------------------------------------------------------------- S1: ZOH

function zoh(u::Vector{Float64}, U::Int)
    y = Vector{Float64}(undef, length(u) * U)
    @inbounds for k in eachindex(u), j in 1:U
        y[(k - 1) * U + j] = u[k]
    end
    return y
end

function zoh_adj(ybar::Vector{Float64}, U::Int)
    n = length(ybar) ÷ U
    ubar = zeros(Float64, n)
    @inbounds for k in 1:n, j in 1:U
        ubar[k] += ybar[(k - 1) * U + j]
    end
    return ubar
end

# ------------------------------------------------- S2: biquad SOS cascade

"""Direct-form-II transposed biquad cascade, zero initial state."""
function filt_sos(x::Vector{Float64}, sos::Matrix{Float64})
    y = copy(x)
    @inbounds for k in axes(sos, 1)
        b0, b1, b2, a0, a1, a2 = sos[k, 1], sos[k, 2], sos[k, 3],
                                 sos[k, 4], sos[k, 5], sos[k, 6]
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
        z1 = 0.0; z2 = 0.0
        for n in eachindex(y)
            xn = y[n]
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            y[n] = yn
        end
    end
    return y
end

"""Adjoint of `filt_sos`: reverse, filter with the SAME cascade, reverse back."""
filt_sos_adj(ybar::Vector{Float64}, sos::Matrix{Float64}) =
    reverse(filt_sos(reverse(ybar), sos))

# ---------------------------------------------------------- S3: IQ mixer

function mixer(i::Vector{Float64}, q::Vector{Float64}, p::ChainParams)
    g = 1.0 + p.gain_imb
    sφ, cφ = sin(p.phase_imb), cos(p.phase_imb)
    return (i .- (g * sφ) .* q .+ p.lo_i, (g * cφ) .* q .+ p.lo_q)
end

function mixer_adj(bi::Vector{Float64}, bq::Vector{Float64}, p::ChainParams)
    g = 1.0 + p.gain_imb
    sφ, cφ = sin(p.phase_imb), cos(p.phase_imb)
    return (bi, (-g * sφ) .* bi .+ (g * cφ) .* bq)
end

# ----------------------------------------------- S4: Rapp AM/AM compression

"""Return the gain `g(r)` and the curvature term `h(r) = g'(r)/r`."""
@inline function rapp_terms(r::Float64, xsat::Float64, p::Float64)
    s = (r / xsat)^(2p)
    g = (1.0 + s)^(-1.0 / (2p))
    common = (1.0 + s)^(-1.0 / (2p) - 1.0) / xsat^(2p)
    h = -common * r^(2p - 2.0)
    return g, h
end

function compress(i::Vector{Float64}, q::Vector{Float64}, p::ChainParams)
    oi = similar(i); oq = similar(q)
    @inbounds for n in eachindex(i)
        r = hypot(i[n], q[n])
        g, _ = rapp_terms(r, p.xsat, p.rapp_p)
        oi[n] = g * i[n]; oq[n] = g * q[n]
    end
    return oi, oq
end

"""Apply the (symmetric) per-sample Jacobian of S4 — serves both JVP and VJP."""
function compress_jac_apply(vi::Vector{Float64}, vq::Vector{Float64},
                            i::Vector{Float64}, q::Vector{Float64}, p::ChainParams)
    oi = similar(vi); oq = similar(vq)
    @inbounds for n in eachindex(vi)
        In, Qn = i[n], q[n]
        r = hypot(In, Qn)
        g, h = rapp_terms(r, p.xsat, p.rapp_p)
        dot = In * vi[n] + Qn * vq[n]
        oi[n] = g * vi[n] + h * In * dot
        oq[n] = g * vq[n] + h * Qn * dot
    end
    return oi, oq
end

# ------------------------------------------------------------- full chain

"""
Entry points accept any real vector and normalise to `Vector{Float64}`.

`juliacall` hands Julia a `PythonCall.PyArray`, which is an `AbstractVector` but
not a `Vector`, so signatures pinned to `Vector{Float64}` raise `MethodError` at
the bridge. Converting once here keeps the inner kernels concretely typed.
"""
forward_with_tape(ui, uq, p::ChainParams) =
    forward_with_tape(Vector{Float64}(ui), Vector{Float64}(uq), p)

function forward_with_tape(ui::Vector{Float64}, uq::Vector{Float64}, p::ChainParams)
    i1, q1 = zoh(ui, p.upsample), zoh(uq, p.upsample)
    i2, q2 = filt_sos(i1, p.sos), filt_sos(q1, p.sos)
    i3, q3 = mixer(i2, q2, p)              # S4 input: the tape
    i4, q4 = compress(i3, q3, p)
    return (p.kappa .* i4, p.kappa .* q4), (i3, q3)
end

forward(ui, uq, p) = forward_with_tape(ui, uq, p)[1]

chain_vjp(bi, bq, tape, p::ChainParams) =
    chain_vjp(Vector{Float64}(bi), Vector{Float64}(bq),
              (Vector{Float64}(tape[1]), Vector{Float64}(tape[2])), p)

"""Reverse mode: walk the five stages backwards, adjointing each."""
function chain_vjp(bi::Vector{Float64}, bq::Vector{Float64},
                   tape::Tuple{Vector{Float64},Vector{Float64}}, p::ChainParams)
    i3, q3 = tape
    bi = p.kappa .* bi;  bq = p.kappa .* bq            # S5
    bi, bq = compress_jac_apply(bi, bq, i3, q3, p)     # S4 (symmetric Jacobian)
    bi, bq = mixer_adj(bi, bq, p)                      # S3
    bi, bq = filt_sos_adj(bi, p.sos), filt_sos_adj(bq, p.sos)  # S2
    return zoh_adj(bi, p.upsample), zoh_adj(bq, p.upsample)    # S1
end

chain_jvp(ti, tq, tape, p::ChainParams) =
    chain_jvp(Vector{Float64}(ti), Vector{Float64}(tq),
              (Vector{Float64}(tape[1]), Vector{Float64}(tape[2])), p)

"""Forward mode: push a tangent through the same five stages."""
function chain_jvp(ti::Vector{Float64}, tq::Vector{Float64},
                   tape::Tuple{Vector{Float64},Vector{Float64}}, p::ChainParams)
    i3, q3 = tape
    ti, tq = zoh(ti, p.upsample), zoh(tq, p.upsample)
    ti, tq = filt_sos(ti, p.sos), filt_sos(tq, p.sos)
    # S3 is affine: the constant LO-leakage offset drops out of the tangent
    g = 1.0 + p.gain_imb
    sφ, cφ = sin(p.phase_imb), cos(p.phase_imb)
    ti, tq = ti .- (g * sφ) .* tq, (g * cφ) .* tq
    ti, tq = compress_jac_apply(ti, tq, i3, q3, p)
    return p.kappa .* ti, p.kappa .* tq
end

end # module
