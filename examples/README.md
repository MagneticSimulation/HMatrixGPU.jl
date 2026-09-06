# Application examples

Runnable, self-validating example scripts for the three typical applications
of HMatrixGPU.jl: scalar boundary-element kernels, covariance (Gaussian
process / spatial statistics) kernels, and the vector-valued (3 DOF per
point) demagnetization tensor of micromagnetics.

Every script follows the same contract:

- the kernel `K` is a **lazy, matrix-free** `AbstractMatrix` — the dense
  matrix never exists, only the blocks the H-matrix assembly queries are
  ever evaluated;
- the compressed matvec is validated against the exact kernel with a hard
  assertion (`relerr < 1e-5`; the script exits non-zero on failure, so any
  script doubles as a smoke test);
- the problem size is configurable as the first command-line argument;
- measurements are printed in a common format (assembly time, block counts,
  ACA rank range, compression ratio, matvec time, relerr).

## The examples

| Script | Scenario | Kernel | Key parameters | Default N (matrix) |
| :-- | :-- | :-- | :-- | :-- |
| [`scalar_laplace2d.jl`](scalar_laplace2d.jl) | 2D Laplace single-layer BEM on a ring | `-log(r)/(2π)` | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`scalar_laplace3d.jl`](scalar_laplace3d.jl) | 3D Laplace single-layer BEM on a Fibonacci sphere | `1/(4πr)` | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`covariance_gaussian.jl`](covariance_gaussian.jl) | Gaussian covariance, GP/kriging in the unit cube | `σ²·exp(-r²/(2ℓ²))`, σ=1, ℓ=0.1 | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`vector_demag.jl`](vector_demag.jl) | dipolar demag tensor, thin-film slab, host-loop kernel | `-(3R_cR_d - r²δ_cd)/(4πr⁵)` | `eta=1.5, eps=1e-4, dims=3` (block sizes default to the trees' dims: 3×3) | 2000 (6000×6000) |
| [`hmatrix_vector.jl`](hmatrix_vector.jl) | same tensor, **device-side kernel evaluation** (advanced) | `-(3R_cR_d - r²δ_cd)/(4πr⁵)` | `eta=1.0, eps=1e-6, dims=3` (block sizes default to the trees' dims: 3×3) | 2000 (6000×6000) |

## Measured numbers

Measured with the default `N` on a fresh Julia session (times include
one-time JIT compilation), 2× NVIDIA A100, Julia 1.12. `relerr` is the
relative error of the compressed matvec against the exact kernel. The CPU
columns are the scripts' main (CPU) section; the CUDA columns come from the
closing GPU section of each script — except `hmatrix_vector.jl`, which runs
on a single automatically selected backend: its CPU row is the script's
CPU-only path (what a CPU-only system or CI runs), its CUDA row the
GPU-device path.

| Script | assembly | compression | rank | matvec | relerr | assembly (CUDA) | matvec (CUDA) | relerr (CUDA) |
| :-- | --: | --: | :-- | --: | --: | --: | --: | --: |
| `scalar_laplace2d.jl` | 5.9 s | 15.8× | 4–5 | 1.24 ms | 2.8e-08 | 3.4 s | 0.06 ms | 2.8e-08 |
| `scalar_laplace3d.jl` | 7.5 s | 2.4× | 9–14 | 11.02 ms | 2.5e-09 | 4.8 s | 0.12 ms | 2.5e-09 |
| `covariance_gaussian.jl` | 13.5 s | 1.3× | 10–34 | 22.00 ms | 5.5e-12 | 8.7 s | 0.17 ms | 5.5e-12 |
| `vector_demag.jl` | 9.8 s | 2.7× | 17–27 | 22.61 ms | 1.7e-10 | 5.9 s | 0.16 ms | 1.7e-10 |
| `hmatrix_vector.jl` (CPU path) | 18.2 s | 1.4× | 46–100 | 58.4 ms | 1.6e-10 | — | — | — |
| `hmatrix_vector.jl` (CUDA) | — | — | — | — | — | 18.8 s | 0.26 ms | 1.6e-10 |

Lazy kernels assemble through the CPU ACA in all examples (the queried
blocks are evaluated by the kernel, then the factors are placed on the
requested backend); only the matvec is device-side — except in
`hmatrix_vector.jl`, where the block evaluation itself already runs on the
device. Notes: the 2D log kernel on a ring is the smoothest case (rank 4–5,
15.8× compression); the covariance example at these parameters is dominated
by its near-field blocks in 3D, so its compression is modest — the win is
the O(N log N) matvec and O(N) storage instead of the O(N²) dense product
(0.17 ms per compressed matvec on the GPU at this size); the vector demag
kernel decays slowly and is anisotropic, so it compresses far less than
smooth scalar kernels even with grouped (3×3) pivoting.

## How to run

```sh
julia --project=. examples/scalar_laplace2d.jl        # default N
julia --project=. examples/scalar_laplace3d.jl 2000   # custom N
```

**CPU:** every script runs entirely on the CPU out of the box. The library
declares the GPU packages as weak dependencies, so the project environment
does not install them and the GPU sections are skipped cleanly.

**GPU:** when the *current Julia environment* has one of `CUDA`, `AMDGPU`,
`oneAPI` or `Metal` installed (with a functional device), `HMatrixGPU.@using_gpu()`
loads it and the GPU section of each script activates automatically — rebuild
with `backend="cuda"`, run the matvec on the device, validate, and switch
back. To run on a GPU, use any environment that has the matching GPU package
installed alongside HMatrixGPU.jl. There is no `using CUDA` in any example;
the library's portable hooks (`create_zeros`, `to_backend`, `kernel_array`)
do the backend dispatch, and `Array(y)` is the only synchronization.

## Which example should I start from?

- **Scalar problem, one DOF per point** (BEM potentials, covariances): start
  with `scalar_laplace2d.jl`, then `scalar_laplace3d.jl` for a 3D geometry.
  No `dims`/block-size settings are needed — the defaults handle it.
- **Vector problem, d DOF per point** (magnetization, elasticity): use
  `vector_demag.jl`. Build the cluster trees with `dims=3` (tree nodes then
  cover whole cells) — the ACA block sizes default to the trees' `dims`, so
  the ACA pivots one cell at a time, which scalar per-column pivoting
  (starving the weak components of anisotropic kernels) would not; explicit
  `row_block_size`/`col_block_size` override the default.
- **Matrix-free vs. dense kernels:** all examples here are matrix-free (lazy
  `K`), which is the fully supported assembly path today — CPU ACA per block,
  factors on any backend. A dense `K::Matrix` with a CUDA backend has a
  batched GPU assembly fast path; that path is still being stabilized, so the
  examples intentionally stick to the matrix-free route. For large-scale
  matrix-free kernels, `hmatrix_vector.jl` shows how to evaluate the queried
  blocks with a KernelAbstractions kernel directly on the active backend.
