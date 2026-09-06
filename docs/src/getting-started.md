# Getting started

## Installation

HMatrixGPU.jl is not yet registered in the Julia General registry. Install it
directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/MagneticSimulation/HMatrixGPU.jl")
```

The package itself has **zero GPU dependencies** — its only non-stdlib
dependency is [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
For GPU support, install one of the vendor packages as well: `CUDA` (NVIDIA),
`AMDGPU`, `oneAPI` (Intel) or `Metal` (Apple). You load the vendor package
yourself (`using CUDA`); loading it has no side effects on the library.

## The backend model

There is **no global backend** to switch and no automatic detection: the
package keeps no backend state, and the landing device is chosen *per
instance*, at construction time, through (in order of priority)

1. `like = <array>` — place the matrix next to the given array;
2. `backend = <name or object>` — `"cpu"`, `"cuda"`, `"amd"`, `"oneapi"`,
   `"metal"` (or a vendor alias such as `"nvidia"`), or a backend object such
   as `CUDA.CUDABackend()`;
3. **the device of the primary data** — device follows data: for the low-level
   constructor this is `K`, for the high-level constructors it is the point
   set (device point sets are downloaded once during construction);
4. `CPU()` — when none of the above says anything.

Explicit keywords always win over the data. Target and source data living on
*different* devices is an error. A `HMatrix` stays where it was built: its
factors and the matvec are fixed to that device.

| `backend=` name | Hardware | Vendor package | Backend object |
| :--------------------- | :--------- | :------------- | :------------------------- |
| `"cpu"` | CPU | — | `KernelAbstractions.CPU()` |
| `"cuda"` / `"nvidia"` | NVIDIA GPU | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | `CUDA.CUDABackend()` |
| `"amd"` / `"roc"` / `"amdgpu"` | AMD GPU | [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) | `AMDGPU.ROCBackend()` |
| `"oneapi"` / `"intel"` | Intel GPU | [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) | `oneAPI.oneAPIBackend()` |
| `"metal"` / `"apple"` | Apple GPU | [Metal.jl](https://github.com/JuliaGPU/Metal.jl) | `Metal.MetalBackend()` |

Because there is no global state, CPU and GPU instances — even from different
vendors — coexist in the same process and can be multiplied in alternation.

`HMatrixGPU.backend_from_name(name)` is the strict resolver that the `backend=`
keyword uses internally. It errors — rather than guessing — when the vendor
package is not loaded (``backend "cuda" requires CUDA.jl — run `using CUDA`
first``) or has no functional device (``backend "cuda" was requested but no
functional device was detected``). There is deliberately no `"gpu"` auto-choice
and no silent fallback: a script that wants a soft fallback wraps the call in
`try`/`catch` itself. The `@using_gpu()` macro is a plain convenience loader —
it loads whichever vendor package the environment provides and does nothing
else.

## A first example (CPU)

The following complete example compresses the 2D Laplace single-layer
potential (a log kernel) between points on a ring, builds the compressed
matrix from a kernel *function*, and validates the product against the exact
kernel:

<!-- PHASE2: 转 @example -->
```julia
using HMatrixGPU, LinearAlgebra

N = 800
pts = [(sin(2π*i/N), cos(2π*i/N), 0.0) for i in 1:N]   # one point per element

# 2D Laplace single-layer potential (log kernel)
H = HMatrix(pts; eta = 1.5, eps = 1e-6, max_points_per_leaf = 64) do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)    # the singular self-pair is g's responsibility
end

info(H)

# validate the compressed product against the exact kernel matrix
K = [let d = norm(pts[i] - pts[j]); d < 1e-12 ? 0.0 : -log(d)/(2π) end
     for i in 1:N, j in 1:N]
x = rand(N)
println("relative error: ", norm(H * x - K * x) / norm(K * x))   # ≈ 1e-8
```

What this shows:

- the kernel is a **function of coordinates**: `g` receives the column views
  of the i-th target and j-th source point (never indices), and the
  `do x, y ... end` form is exactly equivalent to passing a named function;
- `g` must be a **total function** — the coincident-pair guard (`d < 1e-12`)
  belongs to the kernel, which is why the example can build on a ring where
  target and source points coincide;
- `eps` is a *relative* per-block tolerance — the ACA stopping criterion and
  the SVD recompression truncate at fractions of each block's Frobenius norm —
  so the achieved accuracy is independent of the scale of the kernel. Blocks
  whose ε-rank exceeds the storage crossover are stored densely.

## A GPU variant

The same example on an NVIDIA GPU — you load the vendor package and name the
backend; everything else is identical (fenced, not executed here):

```julia
using HMatrixGPU, CUDA        # installing and loading CUDA is the user's choice

H = HMatrix(pts; eta = 1.5, eps = 1e-6, backend = "cuda") do x, y
    d = norm(x - y)
    d < 1e-12 ? 0.0 : -log(d)/(2π)
end

y = H * CuArray(x)   # the matvec runs on the device; x is never moved for you
```

Equivalent spellings for the same landing device: `backend = CUDA.CUDABackend()`,
`like = some_cuda_array`, or no keyword at all when the points themselves live
on the GPU (`HMatrix(g, CuArray(pts); ...)`). Vectors passed to `H * x` /
`mul!` must live on the same device as the matrix — the library does not
transfer them for you.

And because nothing is global, both worlds coexist in one process:

```julia
h_cpu = HMatrix(g, pts; eta = 1.5, eps = 1e-6, backend = "cpu")
h_gpu = HMatrix(g, pts; eta = 1.5, eps = 1e-6, backend = "cuda")
y1 = h_cpu * x                # CPU instance
y2 = h_gpu * CuArray(x)       # GPU instance — interleaved, no interference
```

## Where to go next

- The [manual](manual.md) explains the point-set formats, the kernel contract,
  the trees, the backend/device rules and every tuning parameter
  (`eta`, `eps`, `max_points_per_leaf`, block sizes, group size).
- The [examples](examples.md) gallery lists runnable scripts for scalar BEM,
  covariance and demag-tensor problems.
- Check [known issues](known-issues.md) for current limitations before running
  on a GPU.
