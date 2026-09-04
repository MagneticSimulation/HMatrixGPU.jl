# HMatrixGPU.jl

[![Build Status](https://github.com/MagneticSimulation/HMatrixGPU.jl/workflows/CI/badge.svg)](https://github.com/MagneticSimulation/HMatrixGPU.jl/actions)
[![Coverage](https://codecov.io/gh/magneticsimulation/HMatrixGPU.jl/graph/badge.svg?token=3A2M8U8TYE)](https://codecov.io/gh/magneticsimulation/HMatrixGPU.jl)

HMatrixGPU.jl compresses dense matrices that arise from kernel interactions (boundary
element matrices, dipolar/demagnetization tensors, log-potentials, ...) into
**hierarchical matrices (𝓗-matrices)** and accelerates the matrix–vector product on
**GPUs**. It is written with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
so the same code runs on NVIDIA, AMD, Intel and Apple GPUs as well as the CPU.

The package is primarily developed for micromagnetic simulations: it provides the fast
dense matrix–vector product for the finite-element demagnetization field in
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
- **Multi-backend** via KernelAbstractions: CPU, CUDA, AMDGPU, oneAPI and Metal
  (enabled by simply loading the corresponding package).
- A pure-Julia `HMatrixCPU` reference implementation for validation.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/MagneticSimulation/HMatrixGPU.jl")
```

The package is not yet registered in the General registry. To use a GPU, install
one of `CUDA`, `AMDGPU`, `oneAPI` or `Metal` alongside the package.

## Quick start (CPU)

```julia
using HMatrixGPU, LinearAlgebra

set_backend("cpu")

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

# Build the compressed matrix (flatten=false keeps a CPU reference structure)
H = HMatrix(K, X, Y; eta = 1.5, eps = 1e-6, flatten = false)

info(H)                     # compression statistics
y = H * rand(N)             # compressed matrix-vector product
```

On the example above the compression ratio is ≈ 8.6× (268 leaves, ranks 4–5) and
the relative error of the matvec with respect to the exact kernel is ≈ 1.6e-8.

## GPU usage

```julia
using HMatrixGPU, CUDA   # loading CUDA.jl activates the CUDA backend automatically

# Alternatively:
# set_backend("cuda")     # or "amd", "oneapi", "metal", "cpu"
```

The `HMatrix` should then be constructed with `flatten = true`, which stores all
blocks in flat device arrays so that the matvec runs as two kernels
(`V*x` followed by a fused dense/`U` kernel):

```julia
H = HMatrix(K, X, Y; eta = 1.0, eps = 1e-6, flatten = true)
y = H * x            # x must already live on the GPU (e.g. CuArray)
```

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

| `set_backend` option | Hardware  | KernelAbstractions backend |
| :------------------- | :-------- | :------------------------- |
| `"cpu"`              | CPU       | `KernelAbstractions.CPU()` |
| `"cuda"` / `"nvidia"`| NVIDIA GPU| `CUDA.CUDABackend()`       |
| `"amd"` / `"roc"`    | AMD GPU   | `AMDGPU.ROCBackend()`      |
| `"oneapi"` / `"intel"` | Intel GPU | `oneAPI.oneAPIBackend()` |
| `"metal"` / `"apple"` | Apple GPU | `Metal.MetalBackend()`   |

Loading one of `CUDA`, `AMDGPU`, `oneAPI` or `Metal` automatically selects the
corresponding backend.

## Documentation

- [Getting started](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/getting-started/)
- [User manual](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/manual/)
- [Micromagnetics (FEM demag)](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/micromagnetics/)
- [API reference](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/api/)
- [Known issues](https://magneticsimulation.github.io/HMatrixGPU.jl/dev/known-issues/)

## Acknowledgements

The cluster/block tree construction and the ACA implementation are inspired by the
excellent 𝓗-matrix tutorial in
[TDE: A Tutorial on the Boundary Element Method](https://tbenthompson.com/book/tdes/hmatrix.html)
by T. Ben Thompson.

## License

MIT — see [LICENSE](LICENSE).
