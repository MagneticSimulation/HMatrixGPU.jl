# HMatrixGPU.jl

[![Build Status](https://github.com/MagneticSimulation/HMatrixGPU.jl/workflows/CI/badge.svg)](https://github.com/MagneticSimulation/HMatrixGPU.jl/actions)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://magneticsimulation.github.io/HMatrixGPU.jl/dev)
[![Coverage](https://codecov.io/gh/magneticsimulation/HMatrixGPU.jl/graph/badge.svg?token=3A2M8U8TYE)](https://codecov.io/gh/magneticsimulation/HMatrixGPU.jl)

HMatrixGPU.jl compresses dense matrices that arise from kernel interactions (boundary
element matrices, dipolar/demagnetization tensors, log-potentials, ...) into
**hierarchical matrices (𝓗-matrices)** and accelerates the matrix–vector product on
**GPUs**. It is written with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
so the same code runs on NVIDIA, AMD, Intel and Apple GPUs as well as the CPU.

Typical applications include boundary-element matrices, covariance and
spatial-statistics kernels, and the demagnetization tensors of micromagnetic
simulations.

## Installation

```julia
using Pkg
Pkg.add("HMatrixGPU")
```

To use a GPU, install one of `CUDA`, `AMDGPU`, `oneAPI` or `Metal` alongside
the package and load it with `using`.

## Quick start (CPU)

Write the physics as a kernel function — the library does the bookkeeping:

```julia
using HMatrixGPU, LinearAlgebra

N = 2000
pts = [(sin(2π*i/N), cos(2π*i/N), 0.0) for i in 1:N]  # one point per element
                                                      # (a d×N matrix works too)
# 2D Laplace single-layer potential (log kernel); the same code runs on CPU and GPU
H = HMatrix(pts; eta = 1.5, eps = 1e-6, max_points_per_leaf = 64) do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)          # the self-term guard is the kernel's job
end

info(H)                     # compression statistics
y = H * rand(N)             # compressed matrix-vector product
```

A vector-valued kernel — `g` returning a `d×d` matrix, e.g. the demagnetization
tensor — needs one extra keyword, `dims = d`; the cluster trees, the interleaved
DOF layout (`K[d(p-1)+c, d(q-1)+e] = g(x_p, y_q)[c, e]`) and grouped ACA
pivoting all follow from it, and the matvec keeps flat `dN`-vectors in/out.

On the example above the compression ratio is ≈ 8.6× (276 leaves, ranks 4–5) and
the relative error of the matvec with respect to the exact kernel is ≈ 2.1e-8.
`eps` is a *relative* per-block tolerance (the ACA stopping criterion and the
SVD recompression truncate at fractions of each block's norm), so the achieved
accuracy is independent of the scale of the kernel. Need index-level control,
custom storage or device-side block evaluation? The explicit low-level mode —
a hand-written lazy `AbstractMatrix` plus `HMatrix(K, X, Y; ...)` — covers these cases (see [`examples/scalar_laplace3d.jl`](examples/scalar_laplace3d.jl)
and [`examples/hmatrix_vector.jl`](examples/hmatrix_vector.jl)).

## GPU usage

```julia
using HMatrixGPU, CUDA   # the vendor package is loaded explicitly by the user

H = HMatrix(pts; eta = 1.5, eps = 1e-6, backend = "cuda") do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)
end
y = H * CuArray(x)   # x must already live on the GPU (e.g. CuArray)
```

`backend="cuda"` resolves the loaded CUDA package at construction time. The CPU and GPU
matrices above are independent instances and can be used interleaved.

The `HMatrix` stores all blocks in flat device arrays so that the matvec runs
as three kernels (a permutation, `V*x`, and a fused near-field/`U` kernel).

> [!NOTE]
> The compressed matrix is stored as three CSR operators (near field, far-field
> `V` and `U` factors) with Int32 indices, and the product runs as three fused
> thread-group kernels — no shared-memory barriers, no atomics and no kernel
> arguments that depend on the group size.

## Backend options

The landing backend of an `HMatrix` is decided per construction by the
`backend=` keyword — a name or a KernelAbstractions backend object; the
default is `CPU()`. Device-resident inputs (`K` or point sets on a GPU) are
legal: they are downloaded once during construction but do not decide the
placement, and a cross-device matvec errors out as the safety net. There is
no global backend state and no auto-detection — the user chooses the backend,
the package implements the functionality.

| `backend=` name     | Hardware  | KernelAbstractions backend |
| :------------------ | :-------- | :------------------------- |
| `"cpu"`             | CPU       | `KernelAbstractions.CPU()` |
| `"cuda"` / `"nvidia"`| NVIDIA GPU| `CUDA.CUDABackend()`       |
| `"amd"` / `"roc"`    | AMD GPU   | `AMDGPU.ROCBackend()`      |
| `"oneapi"` / `"intel"` | Intel GPU | `oneAPI.oneAPIBackend()` |
| `"metal"` / `"apple"` | Apple GPU | `Metal.MetalBackend()`   |

`backend=` also accepts a KernelAbstractions backend object directly, and
`backend = KernelAbstractions.get_backend(x)` places the factors next to `x`.
Requesting a GPU backend whose vendor package is not loaded errors with
"run `using CUDA` first"; a loaded package without a functional device errors
as well. For a soft fallback in scripts, wrap the request in `try`/`catch` —
see the examples.

## Related packages

[HMatrices.jl](https://github.com/IntegralEquations/HMatrices.jl) covers the same
problem family on the CPU — a recursive block tree with per-block BLAS and
`Threads`/`Distributed` parallelism. HMatrixGPU.jl is the GPU counterpart: the
𝓗-matrix is flattened into CSR device arrays so the matvec runs as a few fused
KernelAbstractions kernels — pick whichever matches your hardware.

## Documentation

- [Getting started](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/getting-started/)
- [User manual](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/manual/)
- [Examples](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/examples/)
- [Internals](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/internals/)
- [API reference](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/api/)
- [Known issues](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/known-issues/)

## License

MIT — see [LICENSE](LICENSE).
