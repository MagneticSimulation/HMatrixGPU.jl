```@meta
CurrentModule = HMatrixGPU
```

# HMatrixGPU.jl

**HMatrixGPU.jl** compresses dense matrices that arise from kernel interactions
(boundary element matrices, dipolar/demagnetization tensors, covariance
kernels, log-potentials, ...) into **hierarchical matrices (𝓗-matrices)** and
accelerates the matrix–vector product on **GPUs**. It is written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so
the same code runs on NVIDIA, AMD, Intel and Apple GPUs as well as the CPU.

The package is vendor-neutral: its only non-stdlib dependency is
KernelAbstractions, and its only package extension is an optional plotting
one (Plots.jl). You install and load whichever vendor package you want;
instances are placed by their `backend=` keyword alone (nothing specified
means CPU), so CPU and GPU matrices — even from different vendors — coexist
in one process and can be multiplied in alternation.

Typical applications include boundary-element matrices, covariance and
spatial-statistics kernels, and the demagnetization tensors of micromagnetic
simulations — for example the fast dense matrix–vector product of the
finite-element demagnetization field in
[MicroMagnetic.jl](https://github.com/MagneticSimulation/MicroMagnetic.jl).

## At a glance

A kernel matrix from a function `g(x, y)` and a set of points — the library
clusters, compresses and hands you a fast matrix–vector product; the same code
runs on the CPU and on GPUs:

```julia
using HMatrixGPU, LinearAlgebra

N = 2000
pts = [(sin(2π*i/N), cos(2π*i/N), 0.0) for i in 1:N]  # point vector: one point per element

# 2D Laplace single-layer potential (log kernel)
H = HMatrix(pts; eta = 1.5, eps = 1e-6, max_points_per_leaf = 64) do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)      # the singular self-pair is g's responsibility
end

info(H)
y = H * rand(N)      # compressed matrix–vector product
```

```@contents
Pages = ["getting-started.md", "manual.md", "examples.md", "internals.md", "api.md", "known-issues.md"]
Depth = 1
```

## Package contents

| Page | Content |
| :--- | :------ |
| [Getting started](getting-started.md) | Installation, the backend model, first example |
| [Manual](manual.md) | Point sets, kernels, trees, backends & devices, tuning |
| [Examples](examples.md) | Application gallery: scalar BEM, covariance, demag tensor, micromagnetics |
| [Internals](internals.md) | Design of the stable parts: trees, CSR layout, kernels, ACA+, GPU dense assembly |
| [API reference](api.md) | Documented public interface |
| [Known issues](known-issues.md) | Current limitations and fixed issues |
