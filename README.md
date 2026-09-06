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
simulations — for example the fast dense matrix–vector product of the
finite-element demagnetization field in
[MicroMagnetic.jl](https://github.com/MagneticSimulation/MicroMagnetic.jl).

## Features

- **Cluster tree** (median/bbox split) and **block tree** with the standard
  admissibility condition `dist > η·(r_X + r_Y)`.
- **Adaptive Cross Approximation (ACA+)** with partial pivoting and optional
  **SVD recompression** of the low-rank blocks.
- **Matrix-free assembly**: the matrix never needs to exist as a dense array —
  only block-wise `getindex` queries are required, so `K` can be any
  `AbstractMatrix` (including a lazy kernel evaluation on the GPU).
- **Flat, Structure-of-Arrays layout** of the 𝓗-matrix so the matvec runs as a
  handful of fused GPU kernels instead of recursive tree traversals.
- **Multi-backend** via KernelAbstractions: CPU, CUDA, AMDGPU, oneAPI and Metal.
  The package itself has zero GPU dependencies — the vendor package is the
  user's choice and is simply loaded with `using` (loading it has zero side
  effects on HMatrixGPU). Where an `HMatrix` lands is decided per instance
  (`backend=` or `like=` keyword, else the device of the data), and with no
  global backend state, CPU and GPU instances — even from different vendors —
  coexist in one process and interleave freely.

Runnable application examples (scalar BEM, covariance, vector demagnetization
kernels) are in [examples/](examples/README.md).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/MagneticSimulation/HMatrixGPU.jl")
```

The package is not yet registered in the General registry. To use a GPU, install
one of `CUDA`, `AMDGPU`, `oneAPI` or `Metal` alongside the package and load it
with `using`.

## Quick start (CPU)

```julia
using HMatrixGPU, LinearAlgebra

# 2000 points on a ring
N = 2000
pts = reduce(hcat, [[sin(2π*i/N), cos(2π*i/N), 0.0] for i in 1:N])

# Matrix-free kernel: 2D Laplace single-layer potential (log kernel)
struct Laplace2D <: AbstractMatrix{Float64}
    X::Matrix{Float64}
    Y::Matrix{Float64}
end
Base.size(K::Laplace2D) = size(K.X, 2), size(K.Y, 2)
Base.getindex(K::Laplace2D, i::Int, j::Int) =
    let d = norm(K.X[:, i] .- K.Y[:, j]); d < 1e-12 ? 0.0 : -0.5/π*log(d) end

K = Laplace2D(pts, pts)
X = ClusterTree(pts; max_points_per_leaf = 64)
Y = ClusterTree(pts; max_points_per_leaf = 64)

# Build the compressed matrix (the same structure runs on CPU and GPU)
H = HMatrix(K, X, Y; eta = 1.5, eps = 1e-6)

info(H)                     # compression statistics
y = H * rand(N)             # compressed matrix-vector product
```

On the example above the compression ratio is ≈ 8.7× (268 leaves, ranks 4–5) and
the relative error of the matvec with respect to the exact kernel is ≈ 3.1e-8.
`eps` is a *relative* per-block tolerance (the ACA stopping criterion and the
SVD recompression truncate at fractions of each block's norm), so the achieved
accuracy is independent of the scale of the kernel.

## GPU usage

```julia
using HMatrixGPU, CUDA   # the vendor package is loaded explicitly by the user

H = HMatrix(K, X, Y; eta = 1.0, eps = 1e-6, backend = "cuda")
y = H * x            # x must already live on the GPU (e.g. CuArray)
```

`backend="cuda"` resolves the loaded CUDA package at construction time — there
is no auto-detection and no global state: the user (or a higher-level package)
chooses the backend, the library implements the functionality. The CPU and GPU
matrices above are independent instances and can be used interleaved.

The `HMatrix` stores all blocks in flat device arrays so that the matvec runs
as three kernels (a permutation, `V*x`, and a fused near-field/`U` kernel).

For large matrices you typically do not want to materialize `K` at all: define a
batched `getindex(K, I::Vector{Int}, J::Vector{Int})` that evaluates your kernel
with a KernelAbstractions kernel and only the required blocks are ever computed.
A complete, runnable example (the dipolar demag tensor on a GPU) is provided in
[`examples/hmatrix_vector.jl`](examples/hmatrix_vector.jl).

> [!NOTE]
> The compressed matrix is stored as three CSR operators (near field, far-field
> `V` and `U` factors) with Int32 indices, and the product runs as three fused
> thread-group kernels — no shared-memory barriers, no atomics and no kernel
> arguments that depend on the group size.

## Backend options

The landing backend of an `HMatrix` is resolved per construction:
`like=` > `backend=` > the device of the primary data (`K`) > `CPU()`. There is
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
`like=<array>` places the factors next to `like`. Requesting a GPU backend
whose vendor package is not loaded errors with "run `using CUDA` first"; a
loaded package without a functional device errors as well. For a soft fallback
in scripts, wrap the request in `try`/`catch` — see the examples.

## Documentation

- [Getting started](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/getting-started/)
- [User manual](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/manual/)
- [Micromagnetics (FEM demag)](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/micromagnetics/)
- [API reference](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/api/)
- [Known issues](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/known-issues/)

## License

MIT — see [LICENSE](LICENSE).
