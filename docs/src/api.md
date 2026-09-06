# API reference

Public interface of HMatrixGPU.jl. Docstrings of the shipped code are rendered
below; every entry also appears in context on the [manual](manual.md) page.

```@meta
CurrentModule = HMatrixGPU
```

## Trees

```@docs
ClusterTree
```

## Kernel matrix (lazy, matrix-free)

```@docs
KernelMatrix
```

`KernelMatrix` is the lazy kernel behind the high-level [`HMatrix`](@ref)
constructors and can also be held and passed around on its own: every utility
that accepts an `AbstractMatrix` (e.g. `sparsify_hmatrix`) composes with it
directly (see the [manual](manual.md)).

## Hierarchical matrix

```@docs
HMatrix
```

In addition to the documented low-level constructor above, `HMatrix` has four
high-level methods — thin wrappers that build the cluster trees, wrap `g` in a
[`KernelMatrix`](@ref) and forward to the low-level constructor (no parallel
implementation):

| Form | Signature | Notes |
| :--- | :-------- | :---- |
| kernel + point set (square; do-block friendly) | `HMatrix(g::Function, pts; dims = 1, max_points_per_leaf = 32, kwargs...)` | targets = sources = `pts` |
| kernel + point sets (rectangular) | `HMatrix(g::Function, pts_t, pts_s; dims = 1, max_points_per_leaf = 32, backend = nothing, like = nothing, kwargs...)` | point sets on different devices error unless an explicit `backend=`/`like=` is given |
| kernel + custom trees | `HMatrix(g::Function, X::ClusterTree, Y::ClusterTree; dims = nothing, kwargs...)` | `dims` defaults to the trees' `dims`; a mismatch errors |
| own lazy `K` + point sets | `HMatrix(K::AbstractMatrix, pts_t, pts_s; dims = 1, max_points_per_leaf = 32, kwargs...)` | the library builds the trees; the landing device follows `K` |

`kwargs...` are the low-level keywords (`eta`, `eps`, `index_map_using_cpu`,
`svd_recompress`, `row_block_size`, `col_block_size`, `backend`, `like`).
`HMatrix(pts; ...) do x, y ... end` is the do-block spelling of the first form.
The contracts (kernel sentence, `dims` layout, backend resolution) are
described on the [manual](manual.md) page.

## Products

```@docs
mul!
*
```

## Diagnostics

```@docs
info
sparsify_hmatrix
```

## Backends and utilities

```@docs
to_backend
move_to_backend
backend_from_name
```

The remaining backend/allocation helpers are unexported shims over
KernelAbstractions (documented here by signature):

- `create_zeros(B, T, dims...)` / `create_ones(B, T, dims...)` —
  backend-generic allocation, i.e. `KernelAbstractions.zeros(B, T, dims...)`
  and `ones`.
- `kernel_array(B, a)` — move the array `a` to the backend `B` (an alias of
  `move_to_backend`).
- `HMatrixGPU.set_groupsize(n)` — set the workgroup size (default 512) used to
  launch the matvec's permutation kernel; a global performance knob that never
  changes results (see the [manual](manual.md)).
- `@using_gpu()` — try-load the four vendor packages (`CUDA`, `AMDGPU`,
  `oneAPI`, `Metal`); a plain convenience loader with no selection logic.

The backend resolution rules (the `like=` > `backend=` > data > `CPU()` chain,
strict name resolution, no global state) are described on the
[manual](manual.md) page; `backend_from_name` is the strict name resolver the
`backend=` keyword uses.

## Low-rank approximation (internal, documented for reference)

`ACA_plus` is the adaptive cross approximation used for every low-rank block
on the CPU assembly path. It is an internal algorithm — documented here for
reference and because the test suite relies on it.

```@docs
ACA_plus
```
