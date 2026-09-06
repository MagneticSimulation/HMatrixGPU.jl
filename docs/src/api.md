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
| kernel + point sets (rectangular) | `HMatrix(g::Function, pts_t, pts_s; dims = 1, max_points_per_leaf = 32, kwargs...)` | device point sets are legal (downloaded once); the placement is `backend=` (default `CPU()`) |
| kernel + custom trees | `HMatrix(g::Function, X::ClusterTree, Y::ClusterTree; dims = nothing, kwargs...)` | `dims` defaults to the trees' `dims`; a mismatch errors |
| own lazy `K` + point sets | `HMatrix(K::AbstractMatrix, pts_t, pts_s; dims = 1, max_points_per_leaf = 32, kwargs...)` | the library builds the trees; lands on `backend=` (default `CPU()`) |

`kwargs...` are the low-level keywords (`eta`, `eps`, `index_map_using_cpu`,
`svd_recompress`, `row_block_size`, `col_block_size`, `backend`).
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

## Plotting (optional Plots.jl extension)

```@docs
hmatrix_blocks
```

`plot_hmatrix(H::HMatrix; nx = 600, ny = 600, kwargs...) -> Plots.Plot` —
heatmap of the block distribution: teal = dense (near-field) blocks, amber =
low-rank (far-field) blocks, origin `(0, 0)` at the bottom left (matrix
entry `(1, 1)` sits there). No title or axis labels, so a caption can be
provided externally; `nx`/`ny` set the raster resolution and extra
`kwargs...` go to `Plots.heatmap`. Save the result with
`Plots.savefig(p, "hmatrix_pattern.png")`.

The method is provided by the `HMatrixGPUPlotsExt` package extension and
becomes available once [Plots.jl](https://github.com/JuliaPlots/Plots.jl) is
loaded in the session (`using Plots`). Both functions are exported.

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

The backend resolution rules (the explicit `backend=` keyword, default
`CPU()`, strict name resolution, no global state) are described on the
[manual](manual.md) page; `backend_from_name` is the strict name resolver the
`backend=` keyword uses.

## Low-rank approximation (internal, documented for reference)

`ACA_plus` is the adaptive cross approximation used for every low-rank block
on the CPU assembly path. It is an internal algorithm — documented here for
reference and because the test suite relies on it.

```@docs
ACA_plus
```
