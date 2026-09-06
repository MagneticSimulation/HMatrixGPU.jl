# Examples

The repository ships five runnable, self-validating example scripts
([examples/](https://github.com/MagneticSimulation/HMatrixGPU.jl/tree/main/examples) —
scenario, kernel, mode and measured numbers below are synced from
`examples/README.md`). Every script validates the compressed matvec against
the exact kernel (`relerr < 1e-5`, non-zero exit on failure), takes the
problem size as its first command-line argument, and prints a common
measurement format (assembly time, block counts, ACA rank range, compression
ratio, matvec time, relative error).

## The gallery

| Script | Scenario | Kernel | Mode | Key parameters | Default N |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [`scalar_laplace2d.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/scalar_laplace2d.jl) | 2D Laplace single-layer BEM on a ring | `-log(r)/(2π)` | high-level | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`scalar_laplace3d.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/scalar_laplace3d.jl) | 3D Laplace single-layer BEM on a Fibonacci sphere | `1/(4πr)` | explicit low-level | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`covariance_gaussian.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/covariance_gaussian.jl) | Gaussian covariance, GP/kriging in the unit cube | `σ²·exp(-r²/(2ℓ²))`, σ=1, ℓ=0.1 | high-level | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`vector_demag.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/vector_demag.jl) | dipolar demag tensor, thin-film slab, host-loop kernel | `-(3R_cR_d - r²δ_cd)/(4πr⁵)` | high-level, `dims=3` | `eta=1.5, eps=1e-4, dims=3` (ACA block sizes follow `dims`) | 2000 (6000×6000) |
| [`hmatrix_vector.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/hmatrix_vector.jl) | same tensor, **device-side kernel evaluation** (advanced) — one work item per cell pair | `-(3R_cR_d - r²δ_cd)/(4πr⁵)` | explicit low-level, device block evaluation | `eta=1.0, eps=1e-6, dims=3, blocks 3×3` | 2000 (6000×6000) |

The three "high-level" scripts are the do-block form of the
[manual](manual.md); `scalar_laplace3d.jl` shows the same workflow in the
explicit low-level mode (custom struct + batched `getindex`), and
`hmatrix_vector.jl` moves the block *evaluation* itself onto the device with
a KernelAbstractions kernel: one work item per cell pair evaluates the
geometry once and writes the whole `3×3` block, and the tensor is computed as
`(3R_cR_d - r²δ_cd)·t` with `t = -1/(4π·r²·r²·√r²)` — one division per pair,
and `r⁵` decomposed into multiplications plus a single hardware `sqrt`
instead of a software `pow`.

## Measured numbers

Measured with the default `N` on a fresh Julia session (times include one-time
JIT compilation), an NVIDIA A100, Julia 1.12. Source: `examples/README.md`.
`relerr` is the relative error of the compressed matvec against the exact
kernel; the CUDA columns come from the closing GPU section of each script.

| Script | assembly | compression | rank | matvec | relerr | assembly (CUDA) | matvec (CUDA) | relerr (CUDA) |
| :-- | --: | --: | :-- | --: | --: | --: | --: | --: |
| `scalar_laplace2d.jl` | 8.1 s | 15.8× | 4–5 | 1.28 ms | 2.8e-08 | 7.1 s | 0.06 ms | 2.8e-08 |
| `scalar_laplace3d.jl` | 7.5 s | 2.4× | 9–14 | 10.82 ms | 2.5e-09 | 6.1 s | 0.12 ms | 2.5e-09 |
| `covariance_gaussian.jl` | 15.7 s | 1.3× | 10–34 | 21.57 ms | 5.5e-12 | 12.3 s | 0.17 ms | 5.5e-12 |
| `vector_demag.jl` | 12.6 s | 2.7× | 17–27 | 22.10 ms | 1.7e-10 | 10.8 s | 0.16 ms | 1.7e-10 |
| `hmatrix_vector.jl` (CPU path) | 14.7 s | 1.4× | 46–100 | 58.4 ms | 1.6e-10 | — | — | — |
| `hmatrix_vector.jl` (CUDA) | — | — | — | — | — | 20.5 s | 0.28 ms | 1.6e-10 |

Lazy kernels assemble through the CPU ACA in all examples (the queried blocks
are evaluated by the kernel, then the factors are placed on the requested
backend); only the matvec is device-side — except in `hmatrix_vector.jl`,
where the block evaluation itself already runs on the device. The three
high-level scripts pay the wrapper's ≈ 30% assembly-time overhead (see the
[manual](manual.md)); ranks, compression and accuracy are bit-identical to the
equivalent low-level construction. Notes: the 2D
log kernel on a ring is the smoothest case (rank 4–5, 15.8× compression); the
covariance example at these parameters is dominated by its near-field blocks
in 3D, so its compression is modest — the win is the O(N log N) matvec and
O(N) storage instead of the O(N²) dense product; the vector demag kernel
decays slowly and is anisotropic, so it compresses far less than smooth
scalar kernels even with grouped (3×3) pivoting.

## How to run

```sh
julia --project=. examples/scalar_laplace2d.jl        # default N
julia --project=. examples/scalar_laplace3d.jl 2000   # custom N
```

**CPU:** every script runs entirely on the CPU out of the box.

**GPU:** the GPU packages are not installed by HMatrixGPU.jl — run the scripts
in an environment that also has the vendor package for your hardware (`CUDA`,
`AMDGPU`, `oneAPI` or `Metal`) with a functional device. Each script's closing
GPU section resolves a GPU backend explicitly (`HMatrixGPU.@using_gpu()` to
load whatever the environment provides, then `HMatrixGPU.backend_from_name`)
and is skipped cleanly — with a printed notice — when none is available, so
the same script is a CPU smoke test and a GPU benchmark.

## Micromagnetics: the demagnetization field

A typical application is the demagnetization field in micromagnetic
simulations
([MicroMagnetic.jl](https://github.com/MagneticSimulation/MicroMagnetic.jl)):
each time step is one dense matrix–vector product, which is what the
𝓗-matrix accelerates. The tensor assembled in
[`vector_demag.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/vector_demag.jl)
and
[`hmatrix_vector.jl`](https://github.com/MagneticSimulation/HMatrixGPU.jl/blob/main/examples/hmatrix_vector.jl)
is the point-dipole tensor with the self term zeroed — a teaching demo
(`dims = 3`, grouped `3×3` pivoting). Production kernels follow the
discretization: finite-difference codes evaluate the Newell analytic
cell-pair integrals (regularizing the self term and the near field), and FEM
codes use the hybrid FEM–BEM (Fredkin–Koehler) scheme whose compressed
operator is the dense boundary matrix `B` on the mesh boundary (symmetric —
the forward product suffices); MicroMagnetic.jl wires this in as its
`bem_hmatrix` demag method. Either way the change is confined to the kernel
(or the batched `getindex` of a low-level `K`) — the 𝓗-matrix compression
interface is unchanged. Build once per mesh, reuse for every time step.
