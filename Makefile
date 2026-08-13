# Copyright 2026 Bong. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Everything here runs on a laptop CPU. No GPU, no cluster, no accounts.

PY      ?= python
TESS    ?= tesseract
SCRIPTS := scripts
ELEC    := tesseracts/electronics
QUBIT   := tesseracts/transmon

.PHONY: help env build build-electronics build-transmon \
        verify verify-julia verify-endpoints verify-composition \
        reproduce figures clean-images

help:
	@echo "make env         install the Python environment (uv)"
	@echo "make build       build both Tesseract images"
	@echo "make verify      every correctness check quoted in WRITEUP.md"
	@echo "make reproduce   run the bandwidth sweep and regenerate the figure"

# ---------------------------------------------------------------- setup
env:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=$(PWD)/.venv uv pip install \
		'tesseract-core[runtime]==1.11.0' 'tesseract-jax==0.4.1' \
		'jax[cpu]==0.11.0' numpy scipy matplotlib juliacall

build: build-electronics build-transmon

build-electronics:
	$(TESS) build $(ELEC)

build-transmon:
	$(TESS) build $(QUBIT)

# ---------------------------------------------------------------- verify
verify: verify-julia verify-endpoints verify-composition
	@echo
	@echo "all verification passed"

## The hand-derived Julia adjoint, checked without Python, containers or autodiff:
## an adjoint dot-product identity and central finite differences.
verify-julia:
	@echo "== Julia adjoint (standalone) =="
	cd $(ELEC)/julia && julia --startup-file=no --project=. test_adjoint.jl

## Each BUILT IMAGE against the framework's own finite-difference checker, at
## rtol 0.02 -- five times tighter than its default. Deliberately run inside the
## containers, not on the host: that verifies the artifact we actually ship, and
## it means a reviewer needs no Julia, no juliacall and no environment variables.
verify-endpoints:
	@echo "== electronics endpoints (inside the image) =="
	$(PY) $(SCRIPTS)/gradcheck_payloads.py electronics > /tmp/pl_e.json
	$(TESS) run electronics:latest check-gradients "$$(cat /tmp/pl_e.json)" \
		--runtime-args '--eps=1e-5 --rtol=0.02 --seed=0'
	@echo "== transmon endpoints (inside the image) =="
	$(PY) $(SCRIPTS)/gradcheck_payloads.py transmon > /tmp/pl_t.json
	$(TESS) run transmon:latest check-gradients "$$(cat /tmp/pl_t.json)" \
		--runtime-args '--eps=1e-6 --rtol=0.02 --seed=0'

## The composed gradient against central differences taken through BOTH
## containers -- the check that actually exercises the boundary.
verify-composition:
	@echo "== composed Julia/JAX gradient =="
	cd $(SCRIPTS) && $(PY) verify_composition.py

# ------------------------------------------------------------- reproduce
reproduce:
	cd $(SCRIPTS) && $(PY) -u run_sweep.py --axis bandwidth   --maxiter 250 --out sweep
	cd $(SCRIPTS) && $(PY) -u run_sweep.py --axis compression --maxiter 250
	cd $(SCRIPTS) && $(PY) -u run_robustness.py --maxiter 250
	$(MAKE) figures

figures:
	cd $(SCRIPTS) && $(PY) make_figures.py
	cd $(SCRIPTS) && $(PY) make_axes_figure.py
	cd $(SCRIPTS) && $(PY) make_robustness_figure.py

clean-images:
	-docker rmi electronics:latest electronics:0.2.0 transmon:latest transmon:0.1.0
