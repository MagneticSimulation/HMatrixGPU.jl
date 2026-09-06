# Application examples

Runnable, self-validating example scripts for the three typical applications
of HMatrixGPU.jl: scalar boundary-element kernels, covariance (Gaussian
process / spatial statistics) kernels, and the vector-valued (3 DOF per
point) demagnetization tensor of micromagnetics.

Every script follows the same contract:

- the physics is written as a **kernel function** `g(x, y)` that receives the
  *coordinates* of a target/source point pair (a scalar for the 1-DOF
  kernels, a `3×3` matrix for the demag tensor). In the **high-level mode**
  the library derives the lazy `KernelMatrix`, the cluster trees and the
  compressed `HMatrix` from `g` and the point set — `HMatrix(pts) do x, y
  ... end` — and the user writes no bookkeeping at all; in the **explicit
  low-level mode** the user hand-writes a lazy `AbstractMatrix` (struct +
  size + batched `getindex`) and builds `HMatrix(K, X, Y; ...)`, keeping full
  control over storage and block evaluation. Both modes use the same
  assembly and the same compressed representation;
- the kernel is matrix-free in both modes — the dense matrix never exists,
  only the blocks the H-matrix assembly queries are ever evaluated;
- the compressed matvec is validated against the exact kernel with a hard
  assertion (`relerr < 1e-5`; the script exits non-zero on failure, so any
  script doubles as a smoke test);
- the problem size is configurable as the first command-line argument;
- measurements are printed in a common format (assembly time, block counts,
  ACA rank range, compression ratio, matvec time, relerr).

## The examples

| Script | Mode | Scenario | Kernel | Key parameters | Default N (matrix) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [`scalar_laplace2d.jl`](scalar_laplace2d.jl) | high-level (do-block) | 2D Laplace single-layer BEM on a ring | `-log(r)/(2π)` | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`scalar_laplace3d.jl`](scalar_laplace3d.jl) | explicit low-level | 3D Laplace single-layer BEM on a Fibonacci sphere | `1/(4πr)` | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`covariance_gaussian.jl`](covariance_gaussian.jl) | high-level (do-block) | Gaussian covariance, GP/kriging in the unit cube | `σ²·exp(-r²/(2ℓ²))`, σ=1, ℓ=0.1 | `eta=1.5, eps=1e-6` | 4000 (4000×4000) |
| [`vector_demag.jl`](vector_demag.jl) | high-level (do-block, `dims=3`) | dipolar demag tensor, thin-film slab, host-loop kernel | `-(3R_cR_d - r²δ_cd)/(4πr⁵)` | `eta=1.5, eps=1e-4, dims=3` (ACA block sizes follow the `dims`: 3×3) | 2000 (6000×6000) |
| [`hmatrix_vector.jl`](hmatrix_vector.jl) | explicit low-level, **device-side kernel evaluation** (advanced) | same tensor, block queries run on the device | `-(3R_cR_d - r²δ_cd)/(4πr⁵)` | `eta=1.0, eps=1e-6, dims=3` (block sizes follow the `dims`: 3×3) | 2000 (6000×6000) |

## Measured numbers

Measured with the default `N` on a fresh Julia session (times include
one-time JIT compilation), 2× NVIDIA A100, Julia 1.12. `relerr` is the
relative error of the compressed matvec against the exact kernel. The CPU
columns are the scripts' main (CPU) section; the CUDA columns come from the
closing GPU section of each script — except `hmatrix_vector.jl`, which picks
one backend at startup (CUDA when available, CPU otherwise): its CPU row is
the script's CPU-only path (what a CPU-only system or CI runs), its CUDA row
the GPU-device path.

| Script | assembly | compression | rank | matvec | relerr | assembly (CUDA) | matvec (CUDA) | relerr (CUDA) |
| :-- | --: | --: | :-- | --: | --: | --: | --: | --: |
| `scalar_laplace2d.jl` | 8.1 s | 15.8× | 4–5 | 1.28 ms | 2.8e-08 | 7.1 s | 0.06 ms | 2.8e-08 |
| `scalar_laplace3d.jl` | 7.5 s | 2.4× | 9–14 | 10.82 ms | 2.5e-09 | 6.1 s | 0.12 ms | 2.5e-09 |
| `covariance_gaussian.jl` | 15.7 s | 1.3× | 10–34 | 21.57 ms | 5.5e-12 | 12.3 s | 0.17 ms | 5.5e-12 |
| `vector_demag.jl` | 12.6 s | 2.7× | 17–27 | 22.10 ms | 1.7e-10 | 10.8 s | 0.16 ms | 1.7e-10 |
| `hmatrix_vector.jl` (CPU path) | 18.2 s | 1.4× | 46–100 | 58.4 ms | 1.6e-10 | — | — | — |
| `hmatrix_vector.jl` (CUDA) | — | — | — | — | — | 20.5 s | 0.28 ms | 1.6e-10 |

The three high-level scripts and the explicit `scalar_laplace3d.jl` share the
same computation path: the kernel is evaluated by the assembly through CPU ACA
per block (the queried blocks call `g`, then the factors are placed on the
requested backend) and only the matvec is device-side — except in
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

**CPU:** every script runs entirely on the CPU out of the box — without a GPU
package in the environment, `backend_from_name("cuda")` fails and the scripts
fall back to the CPU cleanly (no GPU sections run).

**GPU:** when the *current Julia environment* has one of `CUDA`, `AMDGPU`,
`oneAPI` or `Metal` installed (with a functional device), `HMatrixGPU.@using_gpu()`
loads it and `HMatrixGPU.backend_from_name("cuda")` resolves the backend —
there is no auto-detection and no global state: the scripts pass the chosen
backend object explicitly to every construction (`backend=B`) and every
allocation (`create_zeros(B, ...)`), and fall back to the CPU in a
`try`/`catch` when no functional device is present. Rebuild with
`backend=B`, run the matvec on the device, validate, and continue on the CPU
independently. To run on a GPU, use any environment that has the matching GPU
package installed alongside HMatrixGPU.jl. There is no `using CUDA` in any
example; `Array(y)` is the only synchronization.

## Which example should I start from?

- **Scalar problem, one DOF per point** (BEM potentials, covariances): start
  with `scalar_laplace2d.jl` — the whole workflow is the do-block, no
  `dims`/block-size settings are needed (the defaults handle it). Then
  `scalar_laplace3d.jl` shows the same workflow in the explicit low-level
  mode for a 3D geometry.
- **Vector problem, d DOF per point** (magnetization, elasticity): use
  `vector_demag.jl`. `dims=3` is the single knob — the trees are built per
  cell, the DOFs are flattened automatically, and the ACA block sizes default
  to the cell (3×3): the ACA pivots one cell at a time, which scalar
  per-column pivoting (starving the weak components of anisotropic kernels)
  would not. Explicit `row_block_size`/`col_block_size` still override the
  default.
- **Scaling the block evaluation up:** all examples here evaluate the queried
  blocks on the host through the kernel function. When even that is the
  bottleneck, `hmatrix_vector.jl` shows the explicit low-level mode where the
  block queries run as a KernelAbstractions kernel directly on the device.
