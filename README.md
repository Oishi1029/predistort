# predistort

**Designing a transmon X90 gate through a Julia instrument model and a JAX propagator, with one gradient.**

> **Tesseract Hackathon 2026 — Track 03: Hybrid ML + mechanistic models**
> Solo entry · Apache-2.0 · runs on a laptop CPU
> Technical write-up: **[WRITEUP.md](WRITEUP.md)**

---

## The one-paragraph version

The pulse you design is not the pulse the qubit receives. A DAC holds samples, a
band-limited line smears them, an IQ mixer rotates them, and an output amplifier compresses
them. Worse, the textbook fix for leakage in a transmon — DRAG — works by adding a
*derivative-shaped* quadrature component, and a derivative is precisely the high-frequency
content a band-limited line destroys. So the standard solution is specifically the thing the
electronics break. This project stops treating the instrument as an error to correct afterwards
and differentiates *through* it: one `jax.grad` runs from gate infidelity, back through a JAX
propagator, across a container boundary, into a **hand-derived analytic adjoint written in Julia**,
and out onto the DAC codes.

## The composition

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

The boundary is **both** kinds the organisers name: *language* (Julia ↔ Python/JAX) and
*differentiation strategy* (derived on paper ↔ automatic differentiation).

## Why this needs Tesseract and not `jax.custom_vjp`

The honest objection to any two-container differentiable pipeline is that `jax.custom_vjp`
exists — attach a hand-written backward pass to a function JAX cannot differentiate, keep it in
one process, delete the containers. **That objection is correct whenever the component is
Python.** It is not correct here:

- **The core is Julia.** Collapsing this into one process means embedding a Julia runtime inside
  the JAX process: two garbage collectors, two package managers, two threading models, and a
  multi-second interpreter start per launch. Nobody ships that for a component that is called
  over a socket perfectly well.
- **The image forbids autodiff.** The build **fails** if `jax`, `torch`, `tensorflow` or
  `autograd` can be imported inside the electronics container. The claim "these gradients are not
  machine-generated" is enforced by the build, not asserted in prose.
- **The chain is genuinely nonlinear.** Amplifier compression makes the Jacobian input-dependent,
  so there is no fixed matrix to precompute and fold into JAX. Measured homogeneity violation
  ‖f(2u) − 2f(u)‖∞ = **0.173**, where any linear chain gives exactly zero.

## Verification

Every number below is measured by `make verify` on an Apple M1 Pro, CPU only.

| check | result |
|---|---|
| Julia adjoint, dot-product identity ⟨ȳ, Jv⟩ vs ⟨Jᵀȳ, v⟩ | rel **1.7e-15** |
| Julia adjoint vs central finite differences | rel **8.0e-10** |
| `check-gradients`, electronics, 3 endpoints @ rtol 0.02 | **0 failures / 2000 checks** each |
| `check-gradients`, transmon, 3 endpoints @ rtol 0.02 | **0 failures / 2000 checks** each |
| composed Julia↔JAX gradient vs finite differences **through both containers** | worst rel **1.09e-06** |
| virtual-Z invariance of the metric, 13 angles | **0.000e+00** change |
| propagator vs closed form (0.7π rotation vs X(π) target) | 0.137405 vs **0.137405** |

## Reproducing

```bash
make env        # Python 3.12 venv via uv
make build      # both Tesseract images (~2 min transmon, ~5 min electronics)
make verify     # every check in the table above
make reproduce  # all three experiments and all three figures (~15 min)
```

Requires Docker. **Julia is not needed on the host to build or run anything** — 1.12.6 aarch64 is
baked into the electronics image, precompiled into a layer, with a committed `Manifest.toml`. The
endpoint checks run *inside* the built images for that reason. The one exception is
`make verify-julia`, the standalone adjoint proof, which deliberately runs the Julia test outside
any container and so needs a host Julia; skip it with `make verify-endpoints verify-composition`
if you would rather not install one.

### Two things that will bite you on macOS

1. **colima only mounts `$HOME`.** `Tesseract.from_image()` defaults its output directory to
   `tempfile.mkdtemp()`, which on macOS is `/var/folders/...` — outside the VM's mounts. The bind
   mount then lands as a root-owned empty directory and every `apply` dies with
   `PermissionError`. Pass an explicit `output_path=` under `$HOME`.
2. **`docker buildx` is required** and is not installed with Docker via Homebrew.
   `brew install docker-buildx`, then symlink it into `~/.docker/cli-plugins/`.

### Architecture

Both Tesseracts build for the host platform (`target_platform: "native"`), and the electronics
image picks its Julia toolchain from `uname -m` at build time, so an x86_64 host gets an x86_64
Julia without touching anything. Built and verified here on arm64; the x86 branch is the same code
path but has not been run on x86 hardware.

This repo was checked by cloning it into an empty directory and running
`make env && make build && make verify` from scratch — both images build and every check above
passes, with the composed-gradient error reproducing to the same 1.09e-06.

## Layout

```
tesseracts/electronics/   Tesseract A — Julia core + juliacall bridge
  julia/src/LineChain.jl    the chain and every hand-derived adjoint
  julia/test_adjoint.jl     standalone verification, no Python, no containers
tesseracts/transmon/      Tesseract B — three-level transmon in JAX
scripts/pulses.py         DRAG and the classical pre-distortion baseline
scripts/run_sweep.py      the two hardness axes (--axis bandwidth | compression)
scripts/run_robustness.py designed-vs-perturbed model, and the kappa-robust arm
scripts/verify_*.py       the checks
figures/spine.png         the threshold
figures/axes.png          both axes, one shared cause
figures/robustness.png    the robustness trade
```

## Licence

Apache License 2.0 — see [LICENSE](LICENSE). Tesseract is a registered trademark of
Pasteur Labs, Inc.
