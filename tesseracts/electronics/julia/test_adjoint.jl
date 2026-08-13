# SPDX-License-Identifier: Apache-2.0
# Standalone verification of the hand-derived adjoint. Pure Julia -- no Python,
# no autodiff. Run with:  julia --project=. test_adjoint.jl

include("src/LineChain.jl")
using .LineChain
using Printf
using Random

# 3rd-order Bessel low-pass, 250 MHz, at the 16 GSa/s simulation rate
const SOS = [
    0.00027851513734080725 0.0005570302746816145 0.00027851513734080725 1.0 -0.8779714873192201 0.0;
    1.0                    1.0                   0.0                    1.0 -1.7959744380069622 0.8142334583766941
]

function main()
    Random.seed!(0)
    rng = MersenneTwister(0)
    p = ChainParams(sos = SOS)

    nawg = 32
    # amplitudes large enough to sit well inside compression
    ui = 0.65 .* randn(rng, nawg)
    uq = 0.65 .* randn(rng, nawg)
    ui[1] = 0.0; ui[end] = 0.0        # a real AWG starts and ends at zero
    uq[1] = 0.0; uq[end] = 0.0

    (yi, yq), tape = forward_with_tape(ui, uq, p)
    nsim = length(yi)
    @printf("Nawg=%d -> Nsim=%d\n", nawg, nsim)

    peak = maximum(hypot.(tape[1], tape[2]))
    _, hh = (nothing, nothing)
    @printf("peak |envelope| into compressor = %.4f  (xsat = %.2f)\n", peak, p.xsat)

    # ---- 1. adjoint consistency: <ybar, J v> == <J^T ybar, v> ----
    vi, vq = randn(rng, nawg), randn(rng, nawg)
    bi, bq = randn(rng, nsim), randn(rng, nsim)

    jvi, jvq = chain_jvp(vi, vq, tape, p)
    gi, gq   = chain_vjp(bi, bq, tape, p)

    lhs = sum(bi .* jvi) + sum(bq .* jvq)
    rhs = sum(gi .* vi)  + sum(gq .* vq)
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-30)
    @printf("\ndot-product test\n  <ybar,Jv>   = %.12f\n  <J'ybar,v>  = %.12f\n  rel err     = %.3e\n",
            lhs, rhs, rel)

    # ---- 2. JVP against central finite differences ----
    eps = 1e-6
    (fp_i, fp_q) = forward(ui .+ eps .* vi, uq .+ eps .* vq, p)
    (fm_i, fm_q) = forward(ui .- eps .* vi, uq .- eps .* vq, p)
    fd_i = (fp_i .- fm_i) ./ (2eps)
    fd_q = (fp_q .- fm_q) ./ (2eps)
    err_i = maximum(abs.(fd_i .- jvi)) / max(maximum(abs.(jvi)), 1e-30)
    err_q = maximum(abs.(fd_q .- jvq)) / max(maximum(abs.(jvq)), 1e-30)
    @printf("\nJVP vs central differences\n  max rel err  I %.3e   Q %.3e\n", err_i, err_q)

    # ---- 3. prove the chain is genuinely nonlinear ----
    (s2i, _) = forward(2 .* ui, 2 .* uq, p)
    (s1i, _) = forward(ui, uq, p)
    homog = maximum(abs.(s2i .- 2 .* s1i))
    @printf("\nhomogeneity violation |f(2u)-2f(u)|inf = %.4f  (0 would mean linear)\n", homog)

    ok = rel < 1e-10 && err_i < 1e-6 && err_q < 1e-6 && homog > 1e-3
    println("\n", ok ? "PASS" : "FAIL")
    exit(ok ? 0 : 1)
end

main()
