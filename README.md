# predistort

**Distortion-aware pulse shaping for superconducting-qubit gates, by composing two Tesseracts across a differentiation-strategy boundary.**

> **Tesseract Hackathon 2026 — Track 03: Hybrid ML + mechanistic models**
> Solo entry. Licensed under Apache-2.0.

---

## The problem

A control pulse designed for a qubit is not the pulse the qubit receives. Between the two sit real
control electronics: a finite-bandwidth AWG, an imperfect IQ mixer, and a bias-tee that droops the
low-frequency content away. The pulse arrives smeared, rotated, and sagging, and the gate misses.

The textbook fix for leakage in a transmon is DRAG, which works by adding a *derivative-shaped*
quadrature component. A derivative is precisely the high-frequency content a bandwidth-limited AWG
destroys — so the standard solution is specifically the thing the electronics break.

The right answer is to optimise the pulse **through** the distortion: deliberately pre-distort it,
so that what arrives at the qubit is what you meant.

## Why this needs two Tesseracts

The pipeline is one differentiable function assembled from two components that cannot share a
runtime:

```
pulse parameters
      │
      ▼
┌─────────────────────────────┐
│  Tesseract A — electronics  │   causal FIR + causal IIR + IQ mixing
│  hand-derived analytic      │   NO autodiff framework inside
│  adjoint, NumPy only        │
└─────────────────────────────┘
      │  distorted (I, Q)
      ▼
┌─────────────────────────────┐
│  Tesseract B — transmon     │   time-ordered propagator, three levels
│  JAX autodiff               │   returns gate infidelity
└─────────────────────────────┘
      │
      ▼
   infidelity  ──►  a single jax.grad flows back through BOTH
```

The boundary is one the organisers name explicitly: **differentiation strategy**. Tesseract A's
gradient is derived on paper and coded by hand — the adjoint of a causal convolution is an
anti-causal correlation, and the adjoint of a causal recursion is that recursion run backwards in
time. Tesseract B's gradient comes from JAX. Neither side's method works on the other side's
problem.

## Status

Work in progress during the 2026-08-03 → 2026-08-31 build window.

Verified so far, on an Apple M1 Pro (arm64, CPU only):

| Check | Result |
|---|---|
| Hand-derived electronics adjoint vs autodiff reference | max abs error `2.2e-16` |
| Adjoint dot-product consistency test | relative error `5.4e-10` |
| Propagator vs closed form (0.7π rotation against an X(π) target) | `0.137405` vs `0.137405` |
| Resonant π pulse infidelity | `1.19e-14` |
| Two Tesseracts under one `jax.grad`, vs finite differences taken through both containers | max relative error `6.7e-05` |

## Licence

Apache License 2.0 — see [LICENSE](LICENSE).

Tesseract is a registered trademark of Pasteur Labs, Inc.
