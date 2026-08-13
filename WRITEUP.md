# Pre-distortion by composition

**Designing a transmon X90 gate through a Julia instrument model and a JAX propagator, with one gradient.**

**Track 03 — Hybrid ML + mechanistic models.**
Repository: https://github.com/Oishi1029/predistort · Apache-2.0 · solo entry.

---

## 1. The problem, and why it is not a toy

The pulse a quantum engineer designs is not the pulse the qubit receives. Between the two sit a
DAC holding samples at 2 GSa/s, a baseband line with a few hundred MHz of bandwidth, an IQ mixer
with a percent of gain imbalance and a degree of quadrature skew, and an output amplifier that
compresses near full scale. The pulse arrives smeared, rotated and clipped, and the gate misses.

There is a particular cruelty in how this interacts with the standard solution. A transmon is a
weakly anharmonic oscillator, so a short pulse drives population out of the computational subspace
into $|2\rangle$. The textbook fix is DRAG: add a quadrature component proportional to the
*derivative* of the in-phase envelope. But a derivative is precisely the high-frequency content a
bandwidth-limited line destroys. **The standard leakage fix is specifically the thing the
electronics break.**

The right answer is to stop treating the instrument as an error to be corrected afterwards and
start treating it as part of the system being optimised — to differentiate through it, and design
a pulse that is deliberately wrong at the DAC so that it is right at the qubit.

## 2. Composition across a real boundary

The pipeline is one differentiable function assembled from two containers.

```
theta  ──►  u = u_max·tanh(theta), masked to a 12 ns support
                    │  32 DAC codes per quadrature
                    ▼
        ┌───────────────────────────────────┐
        │  Tesseract A — electronics        │   JULIA numerical core
        │  hand-derived analytic adjoint    │   no AD framework in the image
        └───────────────────────────────────┘
                    │  256 samples per quadrature at 16 GSa/s
                    ▼
        ┌───────────────────────────────────┐
        │  Tesseract B — transmon           │   JAX autodiff
        │  three levels, Taylor propagator  │   virtual-Z-optimal infidelity
        └───────────────────────────────────┘
                    │
                    ▼
              gate infidelity
```

A single `jax.grad` of the infidelity runs backwards through the JAX propagator, across an HTTP
boundary, and into a Julia analytic adjoint. The boundary is **both** of the kinds the organisers
name: *language* (Julia ↔ Python/JAX) and *differentiation strategy* (adjoint derived on paper ↔
automatic differentiation).

**Verification.** The composed gradient was checked against central finite differences of the same
two-container forward pass, so the check exercises the whole chain rather than either half:

```
worst coordinate-wise relative error   1.14e-06
random-direction relative error        1.64e-08
NaNs in the gradient                   0
```

Each Tesseract also passes the framework's own `check-gradients` at `rtol 0.02`, five times tighter
than its default: **0 failures / 2000 checks** on each of `jacobian`,
`jacobian_vector_product` and `vector_jacobian_product`, for both containers.

## 3. Why this genuinely needs Tesseract

The honest objection to any two-container differentiable pipeline is `jax.custom_vjp`. That
decorator exists precisely to attach a hand-written backward pass to a function JAX cannot
differentiate. If component A were a NumPy function with a hand-derived adjoint, a competent JAX
programmer would paste it under `@jax.custom_vjp`, delete both Dockerfiles, save four HTTP
round-trips per iteration, and get bit-identical gradients. The containers would be decoration.

**Component A is not Python.** Its numerical core is a Julia module, and its derivatives are
written in Julia. To collapse this pipeline into one process you would embed a Julia runtime inside
the JAX process: two garbage collectors, two package managers, two threading models, and a
multi-second interpreter start on every process launch. That is not an engineering trade anyone
makes for a component that is called over a socket perfectly well.

Two further facts make the separation real rather than rhetorical:

**The image forbids autodiff.** The build fails if `jax`, `torch`, `tensorflow` or `autograd` can be
imported inside the electronics container. The claim "this component's gradient is not machine-
generated" is enforced by the build, not asserted in prose.

**The calibration is structural, not numerical.** The line response is baked into the image rather
than passed as an input, because two calibrations of the same instrument differ in *shape* — a
third-order Bessel is a 2×6 second-order-section array, and a calibration with an added reflection
resonance is 3×6. A fixed-shape schema field cannot carry that difference; an image tag can. The
instrument model is a versioned artifact measured per instrument by the people who own the
hardware, and consumed by the people who own the physics. That is what a container is for.

## 4. Gradients doing real work

The optimiser controls {{N_DESIGN}} design variables — the in-phase and quadrature DAC codes over a
12 ns support inside a 16 ns window. Amplitude limits are structural, not penalties:
$u = u_{\max}\tanh\theta$ with $u_{\max} = 0.90$, so the DAC box cannot be violated, and the first
and last samples are pinned to exactly zero because a real AWG waveform must start and end there.

Four arms, one metric, identical constraints:

| arm | what it is | infidelity |
|---|---|---|
| 0 | DRAG on a **perfect** line — the unreachable floor | {{ARM0}} |
| 1 | DRAG calibrated ignoring the electronics, played through the real line | {{ARM1}} |
| 2 | **strong baseline**: classical pre-distortion + recalibration | {{ARM2}} |
| 3 | **this work**: end-to-end gradients through both Tesseracts | {{ARM3}} |

**Arm 2 is deliberately strong.** An entry that beats a weak baseline has proved nothing, so the
comparison arm receives every correction a competent engineer applies before reaching for
gradients: exact inversion of the affine IQ mixer including LO-leakage nulling; closed-form
memoryless AM/AM pre-distortion of the Rapp compressor; Tikhonov-regularised inversion of the
measured LTI line response; and recalibration of amplitude, drive phase and the DRAG coefficient,
all fitted through the true chain and scored with the same metric.

What arm 2 cannot do is the entire point. **The compressor sits after the line filter, so inverting
the nonlinearity before the filter does not commute with it.** The residue is the classical memory
effect, and removing it requires a gradient through the composed, *ordered* chain — not a cascade
of independently inverted stages.

The headline is arm 3 against arm 2, not arm 3 against arm 1: **{{GAIN}}× lower infidelity than a
fully corrected classical baseline**, in {{EVALS}} objective evaluations and {{WALL}} of wall-clock
on a laptop CPU.

## 5. Two things that were nearly wrong

**The metric.** Fidelity is maximised in closed form over a free virtual Z. A physical lab
implements Z rotations by relabelling the phase of every subsequent pulse — it is free and
instantaneous — so a figure of merit that charges for Z error reports an error the lab does not
suffer, and would flatter or punish each arm according to how much Z drift it happened to
accumulate. With $c = \mathrm{diag}(M M_{\text{target}}^\dagger)$, the overlap
$|c_0 + e^{-i\varphi}c_1|$ is maximised at $|c_0| + |c_1|$, so no search is needed and the result
stays differentiable. Measured invariance across 13 Z angles: **0.000e+00** change.

**The propagator.** Building the piecewise-constant propagator with `eigh` is the obvious choice and
it is silently wrong. The eigendecomposition derivative carries $1/(w_i - w_j)$ terms, and the drift
Hamiltonian is degenerate at zero drive and zero detuning — so the VJP is NaN at exactly the samples
where a real AWG waveform must start and end, while the forward pass looks perfect. Measured on this
code: **2 NaNs with `eigh`, 0 with a fixed-order Taylor exponential.** Order 12 truncates at
6.9e-13 and the full-gate unitarity error is 3.8e-11.

## 6. The adjoint, derived

Component A's derivatives are the technical core, so they are stated in full. The chain is
zero-order hold → biquad line response → IQ mixer → Rapp compression → drive scale.

**Zero-order hold.** Each input fans out to $U$ outputs, so the adjoint sums cotangents within each
hold block.

**The line response.** A causal LTI filter with zero initial state is a lower-triangular Toeplitz
matrix $H$ whose first column is the impulse response. Its adjoint is $H^{\mathsf T}$, and

$$(H^{\mathsf T} v)[n] \;=\; \sum_{m \ge n} h[m-n]\, v[m] \;=\; \mathrm{reverse}\big(H(\mathrm{reverse}(v))\big)[n].$$

**The adjoint of a causal filter is the same filter run on the time-reversed signal, reversed back.**
A cascade of biquads is itself LTI, so this applies to the whole cascade at once — no per-section
bookkeeping. It is exact, not a numerical approximation of an adjoint.

**The mixer** is affine, so its adjoint is $M^{\mathsf T}$; the LO-leakage offset is constant in the
inputs and contributes nothing.

**The compressor** acts on the envelope magnitude $r = \sqrt{I^2+Q^2}$, with
$g(r) = (1 + (r/x_{\text{sat}})^{2p})^{-1/(2p)}$ and $(I',Q') = g(r)\,(I,Q)$. Its Jacobian is *not*
diagonal — the gain applied to $I$ depends on $Q$. Writing $h(r) \equiv g'(r)/r$,

$$\frac{\partial(I',Q')}{\partial(I,Q)} \;=\; g(r)\,\mathbb{1} \;+\; h(r)\begin{pmatrix} I^2 & IQ \\ IQ & Q^2\end{pmatrix},
\qquad h(r) = -\,\frac{(1+s)^{-\frac{1}{2p}-1}\, r^{2p-2}}{x_{\text{sat}}^{2p}},\quad s=(r/x_{\text{sat}})^{2p}.$$

$h$ is written in closed form rather than as $g'/r$ so that $r=0$ is never evaluated as a removable
singularity. The matrix is symmetric, so forward and reverse mode share one expression.

**This stage is also what makes the container load-bearing.** Without it the chain is a single fixed
matrix that could be precomputed once and folded into JAX as a matmul. With it, the Jacobian depends
on the input: the measured homogeneity violation $\|f(2u) - 2f(u)\|_\infty = 0.173$, where any
linear chain gives exactly zero.

Standalone verification of the Julia adjoint, independent of the containers
(`tesseracts/electronics/julia/test_adjoint.jl`):

```
dot-product test   <ybar, Jv>  vs  <J^T ybar, v>     relative error  1.7e-15
JVP vs central finite differences                    relative error  8.0e-10
```

## 7. Reproducibility

Everything runs on a laptop CPU. `make verify` reproduces every check quoted above; `make reproduce`
regenerates the results table and the figure from scratch.

- Both Tesseracts pin their dependencies; the Julia environment ships a committed `Manifest.toml` so
  the image resolves to the exact package versions used here.
- All seeds are fixed. `results/results.json` is committed.
- Measured transport cost is 11 ms per `apply` and 26 ms per gradient call, flat from 16 to 4096
  elements, so one optimiser step through both containers costs 60–70 ms.
- Built and verified on an Apple M1 Pro (arm64, macOS) with Docker via colima. One portability note:
  the images are built for the host platform, and a `linux/amd64` build path is documented in the
  README.

## 8. Disclosure

Built solo with AI-assisted development (Anthropic Claude) for code generation, drafting and
documentation. The problem framing, composition design, gradient verification and all reported
results are my own and were validated by me.
